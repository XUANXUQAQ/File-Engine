package FileEngine.taskHandler;

import FileEngine.IsDebug;
import FileEngine.SQLiteConfig.SQLiteUtil;
import FileEngine.taskHandler.impl.stop.StopTask;
import FileEngine.taskHandler.impl.stop.CloseTask;
import FileEngine.taskHandler.impl.stop.RestartTask;
import FileEngine.taskHandler.impl.daemon.StopDaemonTask;
import FileEngine.taskHandler.impl.frame.pluginMarket.HidePluginMarketTask;
import FileEngine.taskHandler.impl.frame.searchBar.HideSearchBarTask;
import FileEngine.taskHandler.impl.frame.settingsFrame.HideSettingsFrameTask;
import FileEngine.taskHandler.impl.hotkey.StopListenHotkeyTask;
import FileEngine.taskHandler.impl.monitorDisk.StopMonitorDiskTask;
import FileEngine.taskHandler.impl.plugin.UnloadAllPluginsTask;
import FileEngine.threadPool.CachedThreadPool;
import com.sun.istack.internal.NotNull;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class TaskUtil {
    private static volatile TaskUtil instance = null;
    private final AtomicBoolean exit = new AtomicBoolean(false);
    private final ConcurrentLinkedQueue<Task> TASK_QUEUE = new ConcurrentLinkedQueue<>();
    private final ConcurrentHashMap<Class<? extends Task>, TaskHandler> taskHandlerMap = new ConcurrentHashMap<>();
    private final AtomicBoolean isRejectTask = new AtomicBoolean(false);

    private TaskUtil() {
        handleTask();
    }

    public static TaskUtil getInstance() {
        if (instance == null) {
            synchronized (TaskUtil.class) {
                if (instance == null) {
                    instance = new TaskUtil();
                }
            }
        }
        return instance;
    }

    private void close() {
        restart();
        putTask(new StopDaemonTask());
    }

    private void restart() {
        putTask(new HideSettingsFrameTask());
        putTask(new HidePluginMarketTask());
        putTask(new HideSearchBarTask());
        putTask(new StopListenHotkeyTask());
        putTask(new StopMonitorDiskTask());
        putTask(new UnloadAllPluginsTask());
        SQLiteUtil.closeAll();
        exit.set(true);
    }

    public void waitForTask(@NotNull Task task) {
        try {
            while (hasTaskByInstance(task)) {
                TimeUnit.MILLISECONDS.sleep(50);
            }
        }catch (InterruptedException ignored){
        }
    }

    private boolean hasTaskByInstance(@NotNull Task task) {
        for (Task each : TASK_QUEUE) {
            if (each.equals(task)) {
                return true;
            }
        }
        return false;
    }

    private boolean executeTask(Task task) {
        if (task instanceof StopTask) {
            if (task instanceof RestartTask) {
                restart();
            } else if (task instanceof CloseTask) {
                close();
            }
            task.setFinished();
            isRejectTask.set(true);
            return true;
        }
        for (Class<? extends Task> eachTaskType : taskHandlerMap.keySet()) {
            if (eachTaskType.isInstance(task)) {
                if (task.isBlock()) {
                    taskHandlerMap.get(eachTaskType).doTask(task);
                } else {
                    task.setFinished();
                    CachedThreadPool.getInstance().executeTask(() -> taskHandlerMap.get(eachTaskType).doTask(task));
                }
                return true;
            }
        }
        //当前无可以接该任务的handler
        return false;
    }

    /**
     * 发送任务
     * @param task 任务
     */
    public void putTask(Task task) {
        boolean isDebug = IsDebug.isDebug();
        if (isDebug) {
            System.err.println("尝试放入任务---" + task.toString());
        }
        if (!isRejectTask.get()) {
            TASK_QUEUE.add(task);
        } else {
            if (isDebug) {
                System.err.println("任务已被拒绝---" + task.toString());
            }
        }
    }

    public boolean isNotMainExit() {
        return !exit.get();
    }

    public void registerTaskHandler(@NotNull Class<? extends Task> taskType, @NotNull TaskHandler handler) {
        if (IsDebug.isDebug()) {
            System.err.println("注册监听器" + taskType.toString());
        }
        taskHandlerMap.put(taskType, handler);
    }

    private void handleTask() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                boolean isExecuted;
                while (!exit.get() || !TASK_QUEUE.isEmpty()) {
                    //取出任务
                    for (Task task : TASK_QUEUE) {
                        isExecuted = executeTask(task);
                        if (!isExecuted) {
                            System.err.println("当前任务执行失败---" + task.toString());
                            task.incrementExecuteTimes();
                        }
                        if (task.getExecuteTimes() > 500) {
                            if (IsDebug.isDebug()) {
                                System.err.println("任务超时---" + task.toString());
                                task.setFailed();
                            }
                        }
                    }
                    TASK_QUEUE.removeIf(task -> task.isFailed() || task.isFinished());
                    TimeUnit.MILLISECONDS.sleep(10);
                }
                if (IsDebug.isDebug()) {
                    System.err.println("******取任务线程退出******");
                }
            } catch (InterruptedException ignored) {
            }
        });
    }
}
