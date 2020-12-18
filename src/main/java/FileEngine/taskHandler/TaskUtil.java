package FileEngine.taskHandler;

import FileEngine.IsDebug;
import FileEngine.database.SQLiteUtil;
import FileEngine.taskHandler.impl.daemon.StopDaemonTask;
import FileEngine.taskHandler.impl.frame.pluginMarket.HidePluginMarketTask;
import FileEngine.taskHandler.impl.frame.searchBar.HideSearchBarTask;
import FileEngine.taskHandler.impl.frame.settingsFrame.HideSettingsFrameTask;
import FileEngine.taskHandler.impl.hotkey.StopListenHotkeyTask;
import FileEngine.taskHandler.impl.monitorDisk.StopMonitorDiskTask;
import FileEngine.taskHandler.impl.plugin.UnloadAllPluginsTask;
import FileEngine.taskHandler.impl.stop.CloseTask;
import FileEngine.taskHandler.impl.stop.RestartTask;
import FileEngine.taskHandler.impl.stop.StopTask;
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
    private final ConcurrentLinkedQueue<Task> ASYNC_TASK_QUEUE = new ConcurrentLinkedQueue<>();
    private final ConcurrentHashMap<Class<? extends Task>, TaskHandler> TASK_HANDLER_MAP = new ConcurrentHashMap<>();
    private final AtomicBoolean isRejectTask = new AtomicBoolean(false);

    private final int MAX_TASK_RETRY_TIME = 20;

    private TaskUtil() {
        handleTask();
        startAsyncTaskHandleThread();
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

    private boolean executeTaskFailed(Task task) {
        if (IsDebug.isDebug()) {
            System.err.println("正在尝试执行任务---" + task.toString());
        }
        if (task instanceof StopTask) {
            if (task instanceof RestartTask) {
                restart();
            } else if (task instanceof CloseTask) {
                close();
            }
            task.setFinished();
            isRejectTask.set(true);
            return false;
        }
        task.incrementExecuteTimes();
        for (Class<? extends Task> eachTaskType : TASK_HANDLER_MAP.keySet()) {
            if (eachTaskType.isInstance(task)) {
                TASK_HANDLER_MAP.get(eachTaskType).doTask(task);
                return false;
            }
        }
        //当前无可以接该任务的handler
        return true;
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
        TASK_HANDLER_MAP.put(taskType, handler);
    }

    private void startAsyncTaskHandleThread() {
        for (int i = 0; i < 2; i++) {
            CachedThreadPool.getInstance().executeTask(() -> {
                try {
                    while (isThreadNotExit()) {
                        Task task = ASYNC_TASK_QUEUE.poll();
                        if (task != null && !task.isFinished()) {
                            if (task.getExecuteTimes() > MAX_TASK_RETRY_TIME) {
                                if (IsDebug.isDebug()) {
                                    System.err.println("任务超时---" + task.toString());
                                }
                                task.setFailed();
                                continue;
                            }
                            if (executeTaskFailed(task)) {
                                System.err.println("异步任务执行失败---" + task.toString());
                                ASYNC_TASK_QUEUE.add(task);
                            }
                        }
                        TimeUnit.MILLISECONDS.sleep(5);
                    }
                } catch (InterruptedException ignored) {
                }
            });
        }
    }

    private boolean isThreadNotExit() {
        return (!exit.get() || !TASK_QUEUE.isEmpty() || !ASYNC_TASK_QUEUE.isEmpty());
    }

    private void handleTask() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (isThreadNotExit()) {
                    //移除失败和已执行的任务
                    TASK_QUEUE.removeIf(task -> task.isFailed() || task.isFinished());
                    //取出任务
                    for (Task task : TASK_QUEUE) {

                        if (task.getExecuteTimes() > MAX_TASK_RETRY_TIME) {
                            if (IsDebug.isDebug()) {
                                System.err.println("任务超时---" + task.toString());
                            }
                            task.setFailed();
                            continue;
                        }

                        if (task.isBlock()) {
                            if (executeTaskFailed(task)) {
                                System.err.println("当前任务执行失败---" + task.toString());
                            }
                        } else {
                            ASYNC_TASK_QUEUE.add(task);
                            TASK_QUEUE.remove(task);
                        }

                    }
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
