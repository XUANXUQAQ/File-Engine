package file.engine.utils;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public enum CachedThreadPoolUtil {
    INSTANCE;
    private static final int THREAD_POOL_AWAIT_TIMEOUT = 5;
    private final ExecutorService platformThreadPool;
    private final AtomicBoolean isShutdown = new AtomicBoolean(false);

    CachedThreadPoolUtil() {
        platformThreadPool = new ThreadPoolExecutor(
                0,
                100,
                60,
                TimeUnit.SECONDS,
                new SynchronousQueue<>()
        );
    }

    public static CachedThreadPoolUtil getInstance() {
        return INSTANCE;
    }

    public boolean isShutdown() {
        return isShutdown.get();
    }

    public <T> Future<T> executeTask(Callable<T> task) {
        if (isShutdown.get()) {
            return null;
        }
        return platformThreadPool.submit(task);
    }

    public <T> Future<T> executeTask(Callable<T> todo, boolean isVirtualThread) {
        if (isShutdown.get()) {
            return null;
        }
        if (isVirtualThread) {
            return executeTask(todo);
        } else {
            return platformThreadPool.submit(todo);
        }
    }

    public void executeTask(Runnable todo, boolean isVirtualThread) {
        if (isShutdown.get()) {
            return;
        }
        if (isVirtualThread) {
            executeTask(todo);
        } else {
            platformThreadPool.submit(todo);
        }
    }

    public void executeTask(Runnable task) {
        if (isShutdown.get()) {
            return;
        }
        platformThreadPool.submit(task);
    }

    public void shutdown() {
        isShutdown.set(true);
        platformThreadPool.shutdown();
        printInfo((ThreadPoolExecutor) platformThreadPool);
    }

    private void printInfo(ThreadPoolExecutor threadPoolExecutor) {
        try {
            if (!threadPoolExecutor.awaitTermination(THREAD_POOL_AWAIT_TIMEOUT, TimeUnit.SECONDS)) {
                System.err.println("线程池等待超时");
                int queueSize = threadPoolExecutor.getQueue().size();
                System.err.println("当前排队线程数：" + queueSize);

                int activeCount = threadPoolExecutor.getActiveCount();
                System.err.println("当前活动线程数：" + activeCount);

                long completedTaskCount = threadPoolExecutor.getCompletedTaskCount();
                System.err.println("执行完成线程数：" + completedTaskCount);

                long taskCount = threadPoolExecutor.getTaskCount();
                System.err.println("总线程数：" + taskCount);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
