package FileEngine.eventHandler;

import FileEngine.IsDebug;
import FileEngine.database.SQLiteUtil;
import FileEngine.eventHandler.impl.daemon.StopDaemonEvent;
import FileEngine.eventHandler.impl.frame.pluginMarket.HidePluginMarketEvent;
import FileEngine.eventHandler.impl.frame.searchBar.HideSearchBarEvent;
import FileEngine.eventHandler.impl.frame.settingsFrame.HideSettingsFrameEvent;
import FileEngine.eventHandler.impl.hotkey.StopListenHotkeyEvent;
import FileEngine.eventHandler.impl.monitorDisk.StopMonitorDiskEvent;
import FileEngine.eventHandler.impl.plugin.UnloadAllPluginsEvent;
import FileEngine.eventHandler.impl.stop.CloseEvent;
import FileEngine.eventHandler.impl.stop.RestartEvent;
import FileEngine.eventHandler.impl.stop.StopEvent;
import FileEngine.threadPool.CachedThreadPool;
import com.sun.istack.internal.NotNull;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class EventUtil {
    private static volatile EventUtil instance = null;
    private final AtomicBoolean exit = new AtomicBoolean(false);
    private final ConcurrentLinkedQueue<Event> eventQueue = new ConcurrentLinkedQueue<>();
    private final ConcurrentLinkedQueue<Event> asyncEventQueue = new ConcurrentLinkedQueue<>();
    private final ConcurrentHashMap<Class<? extends Event>, EventHandler> TASK_HANDLER_MAP = new ConcurrentHashMap<>();
    private final AtomicBoolean isRejectTask = new AtomicBoolean(false);

    private final int MAX_TASK_RETRY_TIME = 20;

    private EventUtil() {
        handleTask();
        startAsyncTaskHandleThread();
    }

    public static EventUtil getInstance() {
        if (instance == null) {
            synchronized (EventUtil.class) {
                if (instance == null) {
                    instance = new EventUtil();
                }
            }
        }
        return instance;
    }

    private void close() {
        restart();
        putTask(new StopDaemonEvent());
    }

    private void restart() {
        putTask(new HideSettingsFrameEvent());
        putTask(new HidePluginMarketEvent());
        putTask(new HideSearchBarEvent());
        putTask(new StopListenHotkeyEvent());
        putTask(new StopMonitorDiskEvent());
        putTask(new UnloadAllPluginsEvent());
        SQLiteUtil.closeAll();
        exit.set(true);
    }

    public void waitForTask(@NotNull Event event) {
        try {
            while (hasTaskByInstance(event)) {
                TimeUnit.MILLISECONDS.sleep(50);
            }
        }catch (InterruptedException ignored){
        }
    }

    private boolean hasTaskByInstance(@NotNull Event event) {
        for (Event each : eventQueue) {
            if (each.equals(event)) {
                return true;
            }
        }
        return false;
    }

    private boolean executeTaskFailed(Event event) {
        if (IsDebug.isDebug()) {
            System.err.println("正在尝试执行任务---" + event.toString());
        }
        if (event instanceof StopEvent) {
            if (event instanceof RestartEvent) {
                restart();
            } else if (event instanceof CloseEvent) {
                close();
            }
            event.setFinished();
            isRejectTask.set(true);
            return false;
        }
        event.incrementExecuteTimes();
        for (Class<? extends Event> eachTaskType : TASK_HANDLER_MAP.keySet()) {
            if (eachTaskType.isInstance(event)) {
                TASK_HANDLER_MAP.get(eachTaskType).doTask(event);
                return false;
            }
        }
        //当前无可以接该任务的handler
        return true;
    }

    /**
     * 发送任务
     * @param event 任务
     */
    public void putTask(Event event) {
        boolean isDebug = IsDebug.isDebug();
        if (isDebug) {
            System.err.println("尝试放入任务---" + event.toString());
        }
        if (!isRejectTask.get()) {
            eventQueue.add(event);
        } else {
            if (isDebug) {
                System.err.println("任务已被拒绝---" + event.toString());
            }
        }
    }

    public boolean isNotMainExit() {
        return !exit.get();
    }

    public void register(@NotNull Class<? extends Event> taskType, @NotNull EventHandler handler) {
        if (IsDebug.isDebug()) {
            System.err.println("注册监听器" + taskType.toString());
        }
        TASK_HANDLER_MAP.put(taskType, handler);
    }

    private void startAsyncTaskHandleThread() {
        for (int i = 0; i < 2; i++) {
            CachedThreadPool.getInstance().executeTask(() -> {
                try {
                    while (isEventHandlerNotExit()) {
                        Event event = asyncEventQueue.poll();
                        if (event != null && !event.isFinished()) {
                            if (event.getExecuteTimes() > MAX_TASK_RETRY_TIME) {
                                if (IsDebug.isDebug()) {
                                    System.err.println("任务超时---" + event.toString());
                                }
                                event.setFailed();
                                continue;
                            }
                            if (executeTaskFailed(event)) {
                                System.err.println("异步任务执行失败---" + event.toString());
                                asyncEventQueue.add(event);
                            }
                        }
                        TimeUnit.MILLISECONDS.sleep(5);
                    }
                } catch (InterruptedException ignored) {
                }
            });
        }
    }

    private boolean isEventHandlerNotExit() {
        return (!exit.get() || !eventQueue.isEmpty() || !asyncEventQueue.isEmpty());
    }

    private void handleTask() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (isEventHandlerNotExit()) {
                    //移除失败和已执行的任务
                    eventQueue.removeIf(task -> task.isFailed() || task.isFinished());
                    //取出任务
                    for (Event event : eventQueue) {

                        if (event.getExecuteTimes() > MAX_TASK_RETRY_TIME) {
                            if (IsDebug.isDebug()) {
                                System.err.println("任务超时---" + event.toString());
                            }
                            event.setFailed();
                            continue;
                        }

                        if (event.isBlock()) {
                            if (executeTaskFailed(event)) {
                                System.err.println("当前任务执行失败---" + event.toString());
                            }
                        } else {
                            asyncEventQueue.add(event);
                            eventQueue.remove(event);
                        }

                    }
                    TimeUnit.MILLISECONDS.sleep(5);
                }
                if (IsDebug.isDebug()) {
                    System.err.println("******取任务线程退出******");
                }
            } catch (InterruptedException ignored) {
            }
        });
    }
}
