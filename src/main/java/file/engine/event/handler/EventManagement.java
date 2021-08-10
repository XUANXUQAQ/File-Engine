package file.engine.event.handler;

import file.engine.utils.system.properties.IsDebug;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.event.handler.impl.stop.CloseEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.utils.clazz.scan.ClassScannerUtil;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class EventManagement {
    private static volatile EventManagement instance = null;
    private final AtomicBoolean exit = new AtomicBoolean(false);
    private final ConcurrentLinkedQueue<Event> blockEventQueue = new ConcurrentLinkedQueue<>();
    private final ConcurrentLinkedQueue<Event> asyncEventQueue = new ConcurrentLinkedQueue<>();
    private final ConcurrentHashMap<Class<? extends Event>, Method> EVENT_HANDLER_MAP = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Class<? extends Event>, ConcurrentLinkedQueue<Method>> EVENT_LISTENER_MAP = new ConcurrentHashMap<>();
    private final AtomicInteger failureEventNum = new AtomicInteger(0);

    private final int MAX_TASK_RETRY_TIME = 20;

    private EventManagement() {
        startBlockEventHandler();
        startAsyncEventHandler();
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

    /**
     * 等待任务
     *
     * @param event 任务实例
     * @return true如果任务执行失败， false如果执行正常完成
     */
    public boolean waitForEvent(Event event) {
        try {
            int timeout = 400;
            int count = 0;
            while (!event.isFailed() && !event.isFinished()) {
                count++;
                if (count > timeout) {
                    System.err.println("等待" + event + "超时");
                    break;
                }
                TimeUnit.MILLISECONDS.sleep(50);
            }
        } catch (InterruptedException ignored) {
        }
        return event.isFailed();
    }

    /**
     * 关闭守护进程
     */
    private void stopDaemon() {
        File closeSignal = new File("tmp/closeDaemon");
        if (!closeSignal.exists()) {
            boolean isCreated = false;
            try {
                isCreated = closeSignal.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (!isCreated) {
                System.err.println("创建守护进程关闭标志文件失败");
            }
        }
    }

    /**
     * 执行任务
     *
     * @param event 任务
     * @return true如果执行失败，false执行成功
     */
    private boolean executeTaskFailed(Event event) {
        event.incrementExecuteTimes();
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        if (event instanceof RestartEvent) {
            exit.set(true);
            cachedThreadPoolUtil.executeTaskNoRejection(() -> {
                doAllMethod(EVENT_LISTENER_MAP.get(RestartEvent.class));
                if (event instanceof CloseEvent) {
                    stopDaemon();
                }
            });
            event.setFinished();
            try {
                cachedThreadPoolUtil.shutdown();
            } catch (InterruptedException ignored) {
            }
            return false;
        } else {
            Method eventHandler = EVENT_HANDLER_MAP.get(event.getClass());
            if (eventHandler != null) {
                try {
                    eventHandler.invoke(null, event);
                } catch (IllegalAccessException | InvocationTargetException e) {
                    e.printStackTrace();
                }
                event.setFinished();
                cachedThreadPoolUtil.executeTaskNoRejection(() ->
                        doAllMethod(EVENT_LISTENER_MAP.get(event.getClass())));
                return false;
            }
            //当前无可以接该任务的handler
            return true;
        }
    }

    private StackTraceElement getStackTraceElement() {
        StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
        return stacktrace[3];
    }

    /**
     * 执行所有监听了该Event的任务链
     * @param methodChains 任务链
     */
    private void doAllMethod(ConcurrentLinkedQueue<Method> methodChains) {
        if (methodChains == null) {
            return;
        }
        for (Method each : methodChains) {
            try {
                each.invoke(null);
            } catch (IllegalAccessException | InvocationTargetException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 发送任务
     * 不要在构造函数中执行，可能会导致死锁
     * @param event 任务
     */
    public void putEvent(Event event) {
        boolean isDebug = IsDebug.isDebug();
        if (isDebug) {
            System.err.println("尝试放入任务" + event.toString() + "---来自" + getStackTraceElement().toString());
        }
        if (!exit.get()) {
            if (event.isBlock()) {
                if (!blockEventQueue.contains(event)) {
                    blockEventQueue.add(event);
                }
            } else {
                if (!asyncEventQueue.contains(event)) {
                    asyncEventQueue.add(event);
                }
            }
        } else {
            if (isDebug) {
                System.err.println("任务已被拒绝---" + event);
            }
        }
    }

    public boolean isNotMainExit() {
        return !exit.get();
    }

    /**
     * 注册所有事件处理器
     */
    public void registerAllHandler() {
        try {
            ClassScannerUtil.searchAndRun(EventRegister.class, (annotationClass, method) -> {
                EventRegister annotation = (EventRegister) method.getAnnotation(annotationClass);
                if (IsDebug.isDebug()) {
                    Class<?>[] parameterTypes = method.getParameterTypes();
                    int parameterCount = method.getParameterCount();
                    if (Arrays.stream(parameterTypes).noneMatch(each -> each.equals(Event.class)) || parameterCount != 1) {
                        throw new RuntimeException("注册handler方法参数错误" + method);
                    }
                }
                register(annotation.registerClass(), method);
            });
        } catch (ClassNotFoundException | InvocationTargetException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    /**
     * 注册所有时间监听器
     */
    public void registerAllListener() {
        try {
            ClassScannerUtil.searchAndRun(EventListener.class, (annotationClass, method) -> {
                EventListener annotation = (EventListener) method.getAnnotation(annotationClass);
                if (IsDebug.isDebug()) {
                    int parameterCount = method.getParameterCount();
                    if (parameterCount != 0) {
                        throw new RuntimeException("注册Listener方法参数错误" + method);
                    }
                }
                registerListener(annotation.registerClass(), method);
            });
        } catch (ClassNotFoundException | InvocationTargetException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    /**
     * 注册任务监听器
     *
     * @param eventType 需要监听的任务类型
     * @param handler   需要执行的操作
     */
    private void register(Class<? extends Event> eventType, Method handler) {
        if (IsDebug.isDebug()) {
            System.err.println("注册监听器" + eventType.toString());
        }
        EVENT_HANDLER_MAP.put(eventType, handler);
    }

    /**
     * 监听某个任务被发出，并不是执行任务
     *
     * @param eventType 需要监听的任务类型
     */
    private void registerListener(Class<? extends Event> eventType, Method todo) {
        ConcurrentLinkedQueue<Method> queue = EVENT_LISTENER_MAP.get(eventType);
        if (queue == null) {
            queue = new ConcurrentLinkedQueue<>();
            queue.add(todo);
            EVENT_LISTENER_MAP.put(eventType, queue);
        } else {
            queue.add(todo);
        }
    }

    /**
     * 开启异步任务处理中心
     */
    private void startAsyncEventHandler() {
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        for (int i = 0; i < 4; i++) {
            cachedThreadPoolUtil.executeTaskNoRejection(() -> {
                try {
                    final boolean isDebug = IsDebug.isDebug();
                    Event event;
                    while (isEventHandlerNotExit()) {
                        //取出任务
                        if ((event = asyncEventQueue.poll()) == null) {
                            TimeUnit.MILLISECONDS.sleep(5);
                            continue;
                        }
                        //判断任务是否执行完成或者失败
                        if (event.isFinished() || event.isFailed()) {
                            continue;
                        } else if (event.getExecuteTimes() < MAX_TASK_RETRY_TIME) {
                            //判断是否超过最大次数
                            if (executeTaskFailed(event)) {
                                System.err.println("异步任务执行失败---" + event);
                                asyncEventQueue.add(event);
                            }
                        } else {
                            event.setFailed();
                            failureEventNum.incrementAndGet();
                            if (isDebug) {
                                System.err.println("任务超时---" + event);
                            }
                        }
                        TimeUnit.MILLISECONDS.sleep(5);
                    }
                    if (isDebug) {
                        System.err.println("******异步任务执行线程退出******");
                    }
                } catch (InterruptedException ignored) {
                }
            });
        }
    }

    /**
     * 检查是否所有任务执行完毕再推出
     * @return boolean
     */
    private boolean isEventHandlerNotExit() {
        return (!exit.get() || !blockEventQueue.isEmpty() || !asyncEventQueue.isEmpty());
    }

    /**
     * 开启同步任务事件处理中心
     */
    private void startBlockEventHandler() {
        CachedThreadPoolUtil.getInstance().executeTaskNoRejection(() -> {
            try {
                Event event;
                final boolean isDebug = IsDebug.isDebug();
                while (isEventHandlerNotExit()) {
                    //取出任务
                    if ((event = blockEventQueue.poll()) == null) {
                        TimeUnit.MILLISECONDS.sleep(5);
                        continue;
                    }
                    //判断任务是否已经被执行或者失败
                    if (event.isFinished() || event.isFailed()) {
                        continue;
                    }
                    //判断任务是否超过最大执行次数
                    if (event.getExecuteTimes() < MAX_TASK_RETRY_TIME) {
                        if (executeTaskFailed(event)) {
                            if (failureEventNum.get() > 20) {
                                System.err.println("超过20个任务失败，自动重启");
                                putEvent(new RestartEvent());
                            }
                            System.err.println("同步任务执行失败---" + event);
                            blockEventQueue.add(event);
                        }
                    } else {
                        event.setFailed();
                        failureEventNum.incrementAndGet();
                        if (isDebug) {
                            System.err.println("任务超时---" + event);
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(5);
                }
                if (isDebug) {
                    System.err.println("******同步任务执行线程退出******");
                }
            } catch (InterruptedException ignored) {
            }
        });
    }
}
