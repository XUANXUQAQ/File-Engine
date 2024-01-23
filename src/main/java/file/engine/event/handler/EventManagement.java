package file.engine.event.handler;

import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.event.handler.impl.BuildEventRequestEvent;
import file.engine.event.handler.impl.plugin.EventProcessedBroadcastEvent;
import file.engine.event.handler.impl.plugin.PluginRegisterEvent;
import file.engine.event.handler.impl.stop.CloseEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.services.utils.DaemonUtil;
import file.engine.utils.ThreadPoolUtil;
import file.engine.utils.clazz.scan.ClassScannerUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

@Slf4j
public class EventManagement {
    private static volatile EventManagement instance = null;
    private static volatile boolean exit = false;
    private final ConcurrentLinkedQueue<Event> blockEventQueue = new ConcurrentLinkedQueue<>();
    private final ConcurrentHashMap<String, Method> EVENT_HANDLER_MAP = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, ConcurrentLinkedQueue<Method>> EVENT_LISTENER_MAP = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, BiConsumer<Class<?>, Object>> PLUGIN_EVENT_HANDLER_MAP = new ConcurrentHashMap<>();
    private final Map<String, Map<String, ConcurrentLinkedQueue<ListenerPair>>> PLUGIN_EVENT_LISTENER_MAP = new ConcurrentHashMap<>();
    private HashSet<String> classesList = new HashSet<>();

    @RequiredArgsConstructor
    private static class ListenerPair {
        private final String listenerName;
        private final BiConsumer<Class<?>, Object> listener;
    }

    private EventManagement() {
        startBlockEventHandler();
    }

    public static EventManagement getInstance() {
        if (instance == null) {
            synchronized (EventManagement.class) {
                if (instance == null) {
                    instance = new EventManagement();
                }
            }
        }
        return instance;
    }

    public void unregisterPluginHandler(String className) {
        PLUGIN_EVENT_HANDLER_MAP.remove(className);
    }

    public void removePluginListener(String className, String pluginIdentifier, String listenerName) {
        var pluginListenerMap = PLUGIN_EVENT_LISTENER_MAP.get(className);
        if (pluginListenerMap == null || pluginListenerMap.isEmpty()) {
            return;
        }
        var biConsumers = pluginListenerMap.get(pluginIdentifier);
        if (biConsumers == null || biConsumers.isEmpty()) {
            return;
        }
        biConsumers.removeIf(each -> each.listenerName.equals(listenerName));
    }

    public void registerPluginHandler(String className, BiConsumer<Class<?>, Object> handler) {
        PLUGIN_EVENT_HANDLER_MAP.put(className, handler);
    }

    public synchronized void registerPluginListener(String className, String pluginIdentifier, String listenerName, BiConsumer<Class<?>, Object> listener) {
        if (!PLUGIN_EVENT_LISTENER_MAP.containsKey(className)) {
            PLUGIN_EVENT_LISTENER_MAP.put(className, new ConcurrentHashMap<>());
        }
        var listenerMap = PLUGIN_EVENT_LISTENER_MAP.get(className);
        if (!listenerMap.containsKey(pluginIdentifier)) {
            listenerMap.put(pluginIdentifier, new ConcurrentLinkedQueue<>());
        }
        var listenersPerPlugin = listenerMap.get(pluginIdentifier);
        boolean isListenerNameRepeat = listenersPerPlugin.stream().anyMatch(eachListener -> listenerName.equals(eachListener.listenerName));
        if (isListenerNameRepeat) {
            log.error("插件" + pluginIdentifier + "注册事件 " + className + " 监听器重复，名称为：" + listenerName);
            return;
        }
        listenersPerPlugin.add(new ListenerPair(listenerName, listener));
    }

    /**
     * 等待任务
     *
     * @param event   任务实例
     * @param timeout 超时时间
     * @return true如果任务执行失败， false如果执行正常完成
     */
    public boolean waitForEvent(Event event, int timeout) {
        long startTime = System.currentTimeMillis();
        while (!event.allRetryFailed() && !event.isFinished()) {
            if (System.currentTimeMillis() - startTime > timeout) {
                log.error("等待" + event + "超时");
                break;
            }
            try {
                TimeUnit.MILLISECONDS.sleep(1);
            } catch (InterruptedException e) {
                log.error("error: {}", e.getMessage(), e);
            }
        }
        return event.allRetryFailed();
    }

    /**
     * 等待任务
     *
     * @param event 任务实例
     * @return true如果任务执行失败， false如果执行正常完成
     */
    public boolean waitForEvent(Event event) {
        return waitForEvent(event, 20_000);
    }

    /**
     * 处理重启和关闭事件
     *
     * @param event Restart
     */
    private void handleRestart(RestartEvent event) {
        exit = true;
        doAllMethod(RestartEvent.class.getName(), event);
        if (event instanceof CloseEvent) {
            DaemonUtil.stopDaemon();
        }
        event.setFinishedAndExecCallback();
        ThreadPoolUtil.getInstance().shutdown();
    }

    /**
     * 根据buildEventRequest中的信息创建任务，用于插件发送事件
     *
     * @param buildEventRequestEvent build event request
     * @return true如果构建失败，false构建成功
     */
    private boolean handleBuildEvent(BuildEventRequestEvent buildEventRequestEvent) {
        Object[] eventInfo = buildEventRequestEvent.eventInfo;
        String eventClassName = (String) eventInfo[0];
        Object[] eventParams = (Object[]) eventInfo[1];
        buildEventRequestEvent.setFinishedAndExecCallback();
        // 构建任务
        if (eventClassName == null || eventClassName.isEmpty()) {
            return true;
        }
        Event buildEvent = buildEvent(eventClassName, eventParams);
        if (buildEvent == null) {
            return true;
        }
        putEvent(buildEvent);
        return false;
    }

    private boolean handleNormalEvent(Event event) {
        String eventClassName;
        boolean isPluginEvent = false;
        if (event instanceof PluginRegisterEvent pluginRegisterEvent) {
            isPluginEvent = true;
            eventClassName = pluginRegisterEvent.getClassFullName();
        } else {
            eventClassName = event.getClass().getName();
        }
        BiConsumer<Class<?>, Object> pluginHandler = PLUGIN_EVENT_HANDLER_MAP.get(eventClassName);
        if (pluginHandler != null) {
            try {
                if (isPluginEvent) {
                    PluginRegisterEvent pluginRegisterEvent = (PluginRegisterEvent) event;
                    if (pluginRegisterEvent.getEventObjFromPlugin() == null) {
                        // 兼容以前版本
                        pluginHandler.accept(event.getClass(), event);
                    } else {
                        pluginHandler.accept(pluginRegisterEvent.getEventClass(), pluginRegisterEvent.getEventObjFromPlugin());
                        doAllMethod(eventClassName, event);
                    }
                } else {
                    pluginHandler.accept(event.getClass(), event);
                    doAllMethod(eventClassName, event);
                }
                event.setFinishedAndExecCallback();
                return false;
            } catch (Exception e) {
                log.error("error: {}", e.getMessage(), e);
                return true;
            }
        } else {
            Method eventHandler = EVENT_HANDLER_MAP.get(eventClassName);
            if (eventHandler != null) {
                try {
                    eventHandler.invoke(null, event);
                    doAllMethod(eventClassName, event);
                    event.setFinishedAndExecCallback();
                    return false;
                } catch (Exception e) {
                    log.error("error: {}", e.getMessage(), e);
                    return true;
                }
            } else {
                doAllMethod(eventClassName, event);
                event.setFinishedAndExecCallback();
            }
        }
        return false;
    }

    /**
     * 执行任务
     *
     * @param event 任务
     * @return true如果执行失败，false执行成功
     */
    private boolean executeTaskFailed(Event event) {
        event.incrementExecuteTimes();
        if (event instanceof RestartEvent) {
            handleRestart((RestartEvent) event);
            System.exit(0);
        } else if (event instanceof BuildEventRequestEvent) {
            return handleBuildEvent((BuildEventRequestEvent) event);
        } else {
            return handleNormalEvent(event);
        }
        return true;
    }

    /**
     * 根据类名构建类实例
     *
     * @param eventClassName 事件全类名
     * @param eventParams    事件所需参数
     * @return 事件实例
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private Event buildEvent(String eventClassName, Object[] eventParams) {
        try {
            Class<?> eventClass = Class.forName(eventClassName);
            outer:
            for (Constructor<?> constructor : eventClass.getConstructors()) {
                Class<?>[] parameterTypes = constructor.getParameterTypes();
                int parameterCount = constructor.getParameterCount();
                if (parameterCount == 0) {
                    // 构造方法无参数但传入了参数
                    if (eventParams != null && eventParams.length > 0) {
                        continue;
                    }
                } else {
                    // 构造有参数但没有传入参数
                    if (eventParams == null || eventParams.length != parameterCount) {
                        continue;
                    }
                }
                // 检查参数类型
                for (int i = 0; i < parameterCount; i++) {
                    Class<?> parameterType = parameterTypes[i];
                    Object eventParam = eventParams[i];
                    if (eventParam == null) {
                        continue;
                    }
                    Class<?> eventParamClass = eventParam.getClass();
                    if (!parameterType.isAssignableFrom(eventParamClass)) {
                        continue outer;
                    }
                }
                // 检查完成
                Object eventInstance = constructor.newInstance(eventParams);
                return (Event) eventInstance;
            }
        } catch (ClassNotFoundException | InvocationTargetException | InstantiationException |
                 IllegalAccessException e) {
            // 注入字段
            if (eventParams != null && eventParams.length >= 3) {
                try {
                    PluginRegisterEvent registerEvent = new PluginRegisterEvent();
                    registerEvent.setClassFullName(eventClassName);
                    AtomicBoolean isBlock = (AtomicBoolean) eventParams[0];
                    if (isBlock.get()) {
                        registerEvent.setBlock();
                    }
                    registerEvent.setCallback((Consumer) eventParams[1]);
                    registerEvent.setErrorHandler((Consumer) eventParams[2]);
                    if (eventParams.length > 4) {
                        // 老版本传入的参数为paramsMap，在该版本中已弃用，因此需要判断，否则会导致数组越界
                        registerEvent.setEventObjFromPlugin(eventParams[3]);
                        registerEvent.setEventClass((Class<?>) eventParams[4]);
                    } else if (eventParams.length == 4) {
                        registerEvent.setParams((LinkedHashMap<String, Object>) eventParams[3]);
                    }
                    return registerEvent;
                } catch (Exception ex) {
                    log.error("error: {}", ex.getMessage(), ex);
                }
            } else {
                log.error("error: {}", e.getMessage(), e);
            }
        }
        return null;
    }

    /**
     * 用于在debug时查看在哪个位置发出的任务
     * 由于执行任务的调用栈长度超过3，所以不会出现数组越界
     *
     * @return stackTraceElement
     */
    private StackTraceElement getStackTraceElement() {
        StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
        return stacktrace[3];
    }

    /**
     * 执行所有监听了该Event的任务链
     *
     * @param eventType 任务类型
     * @param event     任务
     */
    private void doAllMethod(String eventType, Event event) {
        var methodChains = EVENT_LISTENER_MAP.get(eventType);
        if (methodChains != null) {
            methodChains.forEach(each -> {
                try {
                    each.invoke(null, event);
                } catch (IllegalAccessException | InvocationTargetException e) {
                    log.error("error: {}", e.getMessage(), e);
                }
            });
        }
        var classObjectBiConsumer = PLUGIN_EVENT_LISTENER_MAP.get(eventType);
        if (classObjectBiConsumer != null) {
            if (event instanceof PluginRegisterEvent pluginRegisterEvent) {
                classObjectBiConsumer.forEach((pluginName, listeners) ->
                        listeners.forEach(listener ->
                                listener.listener.accept(pluginRegisterEvent.getEventClass(), pluginRegisterEvent.getEventObjFromPlugin())));
            } else {
                classObjectBiConsumer.forEach((pluginName, listeners) -> listeners.forEach(listener -> listener.listener.accept(event.getClass(), event)));
            }
        }
    }

    /**
     * 发送任务
     * 不要在构造函数中执行，单例模式下可能会导致死锁
     *
     * @param event 任务
     */
    public void putEvent(Event event) {
        final boolean isDebug = IsDebug.isDebug();
        if (isDebug) {
            log.info("尝试放入任务" + event.toString() + "---来自" + getStackTraceElement().toString());
        }
        if (notMainExit()) {
            if (event.isBlock()) {
                blockEventQueue.add(event);
            } else {
                ThreadPoolUtil.getInstance().executeTask(() -> eventHandle(event));
            }
        } else {
            if (isDebug) {
                log.warn("任务已被拒绝---" + event);
            }
        }
    }

    /**
     * 异步回调方法发送任务
     * 不要在构造函数中执行，单例模式下可能会导致死锁
     *
     * @param event        任务
     * @param callback     回调函数
     * @param errorHandler 错误处理
     */
    public void putEvent(Event event, Consumer<Event> callback, Consumer<Event> errorHandler) {
        event.setCallback(callback);
        event.setErrorHandler(errorHandler);
        putEvent(event);
    }

    /**
     * 全局系统退出标志，用于常驻循环判断标志
     *
     * @return true如果系统未退出
     */
    public boolean notMainExit() {
        return !exit;
    }

    /**
     * 注册所有事件处理器
     */
    public void registerAllHandler() {
        try {
            BiConsumer<Class<? extends Annotation>, Method> registerMethod = (annotationClass, method) -> {
                EventRegister annotation = (EventRegister) method.getAnnotation(annotationClass);
                if (IsDebug.isDebug()) {
                    Class<?>[] parameterTypes = method.getParameterTypes();
                    if (!Modifier.isStatic(method.getModifiers())) {
                        throw new RuntimeException("方法不是static " + method);
                    }
                    if (Arrays.stream(parameterTypes).noneMatch(each -> each.equals(Event.class)) || method.getParameterCount() != 1) {
                        throw new RuntimeException("注册handler方法参数错误 " + method);
                    }
                }
                String registerClassName = annotation.registerClass().getName();
                if (RestartEvent.class.getName().equals(registerClassName) || CloseEvent.class.getName().equals(registerClassName)) {
                    throw new RuntimeException("RestartEvent和CloseEvent不可被注册事件处理器");
                }
                registerHandler(registerClassName, method);
            };
            if (IsDebug.isDebug()) {
                ClassScannerUtil.searchAndRun(EventRegister.class, registerMethod);
            } else {
                Class<?> c;
                Method[] methods;
                for (String className : classesList) {
                    c = Class.forName(className);
                    methods = c.getDeclaredMethods();
                    for (Method eachMethod : methods) {
                        eachMethod.setAccessible(true);
                        if (eachMethod.isAnnotationPresent(EventRegister.class)) {
                            registerMethod.accept(EventRegister.class, eachMethod);
                        }
                    }
                }
            }
        } catch (ClassNotFoundException e) {
            log.error("error: {}", e.getMessage(), e);
        }
    }

    /**
     * 注册所有时间监听器
     */
    public void registerAllListener() {
        try {
            BiConsumer<Class<? extends Annotation>, Method> registerListenerMethod = (annotationClass, method) -> {
                EventListener annotation = (EventListener) method.getAnnotation(annotationClass);
                if (IsDebug.isDebug()) {
                    Class<?>[] parameterTypes = method.getParameterTypes();
                    if (!Modifier.isStatic(method.getModifiers())) {
                        throw new RuntimeException("方法不是static" + method);
                    }
                    if (Arrays.stream(parameterTypes).noneMatch(each -> each.equals(Event.class)) || method.getParameterCount() != 1) {
                        throw new RuntimeException("注册Listener方法参数错误" + method);
                    }
                }
                for (Class<? extends Event> aClass : annotation.listenClass()) {
                    registerListener(aClass.getName(), method);
                }
            };
            if (IsDebug.isDebug()) {
                ClassScannerUtil.searchAndRun(EventListener.class, registerListenerMethod);
            } else {
                Class<?> c;
                Method[] methods;
                for (String className : classesList) {
                    c = Class.forName(className);
                    methods = c.getDeclaredMethods();
                    for (Method eachMethod : methods) {
                        eachMethod.setAccessible(true);
                        if (eachMethod.isAnnotationPresent(EventListener.class)) {
                            registerListenerMethod.accept(EventListener.class, eachMethod);
                        }
                    }
                }
            }
        } catch (ClassNotFoundException e) {
            log.error("error: {}", e.getMessage(), e);
        }
    }

    public void readClassList() {
        try (var reader = new BufferedReader(
                new InputStreamReader(
                        Objects.requireNonNull(EventManagement.class.getResourceAsStream("/classes.list")),
                        StandardCharsets.UTF_8
                ))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.isBlank()) {
                    classesList.add(line);
                }
            }
        } catch (IOException e) {
            log.error("error: {}", e.getMessage(), e);
        }
    }

    public void releaseClassesList() {
        classesList = null;
    }

    /**
     * 注册任务监听器
     *
     * @param eventType 需要监听的任务类型
     * @param handler   需要执行的操作
     */
    private void registerHandler(String eventType, Method handler) {
        if (IsDebug.isDebug()) {
            log.info("注册处理器" + eventType);
        }
        if (EVENT_HANDLER_MAP.containsKey(eventType)) {
            throw new RuntimeException("重复的监听器：" + eventType + "方法：" + handler);
        }
        EVENT_HANDLER_MAP.put(eventType, handler);
    }

    /**
     * 监听某个任务被发出，并不是执行任务
     *
     * @param eventType 需要监听的任务类型
     */
    private void registerListener(String eventType, Method listenerMethod) {
        if (IsDebug.isDebug()) {
            log.info("注册监听器" + eventType);
        }
        ConcurrentLinkedQueue<Method> queue = EVENT_LISTENER_MAP.get(eventType);
        if (queue == null) {
            queue = new ConcurrentLinkedQueue<>();
            queue.add(listenerMethod);
            EVENT_LISTENER_MAP.put(eventType, queue);
        } else {
            queue.add(listenerMethod);
        }
    }

    /**
     * 检查是否所有任务执行完毕再推出
     *
     * @return boolean
     */
    private boolean isEventHandlerNotExit() {
        return !(exit && blockEventQueue.isEmpty());
    }

    /**
     * 开启同步任务事件处理中心
     */
    private void startBlockEventHandler() {
        new Thread(() -> eventHandle(blockEventQueue)).start();
    }

    /**
     * 事件处理器
     * 注意，容器不能使用SynchronousQueue，因为事件处理的过程中可能会放入其他事件，会导致putEvent和eventHandle互相等待的问题
     *
     * @param eventQueue eventQueue
     */
    private void eventHandle(ConcurrentLinkedQueue<Event> eventQueue) {
        final boolean isDebug = IsDebug.isDebug();
        while (isEventHandlerNotExit()) {
            //取出任务
            Event event = eventQueue.poll();
            if (event == null) {
                try {
                    TimeUnit.MILLISECONDS.sleep(5);
                } catch (InterruptedException e) {
                    log.error("error: {}", e.getMessage(), e);
                }
                continue;
            }
            //判断任务是否执行完成或者失败
            if (event.isFinished()) {
                continue;
            }
            if (event.allRetryFailed()) {
                //判断是否超过最大次数
                event.execErrorHandler();
                event.setFinished();
                if (isDebug) {
                    log.error("任务超时---" + event);
                }
            } else {
                if (executeTaskFailed(event)) {
                    log.error("任务执行失败---" + event);
                    eventQueue.add(event);
                } else {
                    // 任务执行成功
                    if (!(event instanceof EventProcessedBroadcastEvent)) {
                        putEvent(new EventProcessedBroadcastEvent(event.getClass(), event));
                    }
                }
            }
        }
    }

    private void eventHandle(@NonNull Event event) {
        final boolean isDebug = IsDebug.isDebug();
        while (!exit) {
            //判断任务是否执行完成或者失败
            if (event.isFinished()) {
                return;
            }
            if (event.allRetryFailed()) {
                //判断是否超过最大次数
                event.execErrorHandler();
                event.setFinished();
                if (isDebug) {
                    log.error("任务超时---" + event);
                }
            } else {
                if (executeTaskFailed(event)) {
                    log.error("任务执行失败---" + event);
                } else {
                    // 任务执行成功
                    if (!(event instanceof EventProcessedBroadcastEvent)) {
                        putEvent(new EventProcessedBroadcastEvent(event.getClass(), event));
                    }
                }
            }
        }
    }
}
