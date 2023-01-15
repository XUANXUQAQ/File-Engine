package file.engine.utils;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public enum CachedThreadPoolUtil {
    INSTANCE;
    private static final int THREAD_POOL_AWAIT_TIMEOUT = 10;
    private final ExecutorService platformThreadPool;
    private final AtomicBoolean isShutdown = new AtomicBoolean(false);

    CachedThreadPoolUtil() {
        platformThreadPool = new ThreadPoolExecutor(
                0,
                100,
                60,
                TimeUnit.SECONDS,
                new SynchronousQueue<>(),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy()
        );
    }

    public static CachedThreadPoolUtil getInstance() {
        return INSTANCE;
    }

    public boolean isShutdown() {
        return isShutdown.get();
    }

    private <T> Future<T> executeTaskPlatform(Callable<T> task) {
        return platformThreadPool.submit(task);
    }

    private void executeTaskPlatform(Runnable task) {
        platformThreadPool.submit(task);
    }

    /**
     * 提交任务
     *
     * @param task            任务
     * @param isVirtualThread 是否使用虚拟线程
     * @return Future
     */
    @SuppressWarnings("unused")
    public <T> Future<T> executeTask(Callable<T> task, boolean isVirtualThread) {
        if (isShutdown.get()) {
            return null;
        }
        return executeTaskPlatform(task);
    }

    /**
     * 提交任务
     *
     * @param task            任务
     * @param isVirtualThread 是否使用虚拟线程
     */
    @SuppressWarnings("unused")
    public void executeTask(Runnable task, boolean isVirtualThread) {
        if (isShutdown.get()) {
            return;
        }
        executeTaskPlatform(task);
    }

    /**
     * 使用虚拟线程提交任务
     *
     * @param task 任务
     */
    public void executeTask(Runnable task) {
        if (isShutdown.get()) {
            return;
        }
        executeTaskPlatform(task);
    }

    /**
     * 关闭线程池病等待
     */
    public void shutdown() {
        isShutdown.set(true);
        platformThreadPool.shutdown();
        printInfo((ThreadPoolExecutor) platformThreadPool);
    }

    /**
     * 等待线程池关闭并打印线程池信息
     *
     * @param threadPoolExecutor 线程池
     */
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
