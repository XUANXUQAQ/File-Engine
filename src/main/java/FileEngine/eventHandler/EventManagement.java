package FileEngine.eventHandler;

import FileEngine.IsDebug;
import FileEngine.eventHandler.impl.stop.CloseEvent;
import FileEngine.eventHandler.impl.stop.RestartEvent;
import FileEngine.utils.CachedThreadPoolUtil;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class EventManagement {
    private static volatile EventManagement instance = null;
    private final AtomicBoolean exit = new AtomicBoolean(false);
    private final ConcurrentLinkedQueue<Event> blockEventQueue = new ConcurrentLinkedQueue<>();
    private final ConcurrentLinkedQueue<Event> asyncEventQueue = new ConcurrentLinkedQueue<>();
    private final ConcurrentHashMap<Class<? extends Event>, EventHandler> EVENT_HANDLER_MAP = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Class<? extends Event>, ConcurrentLinkedQueue<Runnable>> EVENT_LISTENER_MAP = new ConcurrentHashMap<>();
    private final AtomicBoolean isRejectTask = new AtomicBoolean(false);

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
     * @param event 任务实例
     * @return true如果任务执行失败， false如果执行正常完成
     */
    public boolean waitForEvent(Event event) {
        try {
            int timeout = 200;
            int count = 0;
            while (!event.isFailed() && !event.isFinished()) {
                count++;
                if (count > timeout){
                    System.err.println("等待" + event.toString() + "超时");
                    break;
                }
                TimeUnit.MILLISECONDS.sleep(50);
            }
        }catch (InterruptedException ignored){
        }
        return event.isFailed();
    }

    /**
     * 执行任务
     * @param event 任务
     * @return true如果执行失败，false执行成功
     */
    private boolean executeTaskFailed(Event event) {
        event.incrementExecuteTimes();
        if (event instanceof RestartEvent) {
            event.setFinished();
            exit.set(true);
            CachedThreadPoolUtil.getInstance().executeTask(new Thread(() -> {
                doAllMethod(EVENT_LISTENER_MAP.get(RestartEvent.class));
                if (event instanceof CloseEvent) {
                    doAllMethod(EVENT_LISTENER_MAP.get(CloseEvent.class));
                }
            }, "close register func"));
            isRejectTask.set(true);
            try {
                CachedThreadPoolUtil.getInstance().shutdown();
            } catch (InterruptedException ignored) {
            }
            return false;
        } else {
            EventHandler eventHandler = EVENT_HANDLER_MAP.get(event.getClass());
            if (eventHandler != null) {
                eventHandler.doEvent(event);
                CachedThreadPoolUtil.getInstance().executeTask(new Thread(() ->
                        doAllMethod(EVENT_LISTENER_MAP.get(event.getClass())), "do event listener func"));
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

    private void doAllMethod(ConcurrentLinkedQueue<Runnable> todo) {
        if (todo == null) {
            return;
        }
        for (Runnable each : todo) {
            each.run();
        }
    }

    /**
     * 发送任务
     * @param event 任务
     */
    public void putEvent(Event event) {
        boolean isDebug = IsDebug.isDebug();
        if (isDebug) {
            System.err.println("尝试放入任务" + event.toString() + "---来自" + getStackTraceElement().toString());
        }
        if (!isRejectTask.get()) {
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
                System.err.println("任务已被拒绝---" + event.toString());
            }
        }
    }

    public boolean isNotMainExit() {
        return !exit.get();
    }

    /**
     * 注册任务监听器
     * @param eventType 需要监听的任务类型
     * @param handler 需要执行的操作
     */
    public void register(Class<? extends Event> eventType, EventHandler handler) {
        if (IsDebug.isDebug()) {
            System.err.println("注册监听器" + eventType.toString());
        }
        EVENT_HANDLER_MAP.put(eventType, handler);
    }

    /**
     * 监听某个任务被发出，并不是执行任务
     * @param eventType 需要监听的任务类型
     */
    public void registerListener(Class<? extends Event> eventType, Runnable todo) {
        ConcurrentLinkedQueue<Runnable> queue = EVENT_LISTENER_MAP.get(eventType);
        if (queue == null) {
            queue = new ConcurrentLinkedQueue<>();
            queue.add(todo);
            EVENT_LISTENER_MAP.put(eventType, queue);
        } else {
            queue.add(todo);
        }
    }

    private void startAsyncEventHandler() {
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        for (int i = 0; i < 4; i++) {
            cachedThreadPoolUtil.executeTask(new Thread(() -> {
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
                                System.err.println("异步任务执行失败---" + event.toString());
                                asyncEventQueue.add(event);
                            }
                        } else {
                            event.setFailed();
                            if (isDebug) {
                                System.err.println("任务超时---" + event.toString());
                            }
                        }
                        TimeUnit.MILLISECONDS.sleep(5);
                    }
                    if (isDebug) {
                        System.err.println("******异步任务执行线程退出******");
                    }
                } catch (InterruptedException ignored) {
                }
            }, "async event handler"));
        }
    }

    private boolean isEventHandlerNotExit() {
        return (!exit.get() || !blockEventQueue.isEmpty() || !asyncEventQueue.isEmpty());
    }

    private void startBlockEventHandler() {
        CachedThreadPoolUtil.getInstance().executeTask(new Thread(() -> {
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
                            System.err.println("同步任务执行失败---" + event.toString());
                            blockEventQueue.add(event);
                        }
                    } else {
                        event.setFailed();
                        if (isDebug) {
                            System.err.println("任务超时---" + event.toString());
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(5);
                }
                if (isDebug) {
                    System.err.println("******同步任务执行线程退出******");
                }
            } catch (InterruptedException ignored) {
            }
        }, "block event handler"));
    }
}
