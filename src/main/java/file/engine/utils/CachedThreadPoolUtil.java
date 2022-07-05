package file.engine.utils;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public enum CachedThreadPoolUtil {
    INSTANCE;
    private static final int THREAD_POOL_AWAIT_TIMEOUT = 5;
    private final ExecutorService virtualThreadPool;
    private final ExecutorService platformThreadPool;
    private final AtomicBoolean isShutdown = new AtomicBoolean(false);

    CachedThreadPoolUtil() {
        virtualThreadPool = new ThreadPoolExecutor(
                0,
                1000,
                0,
                TimeUnit.SECONDS,
                new SynchronousQueue<>(),
                Thread.ofVirtual().factory()
        );
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

    /**
     * 提交任务
     * @param todo 任务
     * @return Future
     */
    public <T> Future<T> executeTask(Callable<T> todo) {
        if (isShutdown.get()) {
            return null;
        }
        return virtualThreadPool.submit(todo);
    }

    /**
     * 提交任务
     * @param todo 任务
     * @param isVirtualThread 是否使用虚拟线程
     * @return Future
     */
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

    /**
     * 提交任务
     * @param todo 任务
     * @param isVirtualThread 是否使用虚拟线程
     */
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

    /**
     * 使用虚拟线程提交任务
     * @param todo 任务
     */
    public void executeTask(Runnable todo) {
        if (isShutdown.get()) {
            return;
        }
        virtualThreadPool.submit(todo);
    }

    /**
     * 关闭线程池病等待
     */
    public void shutdown() {
        isShutdown.set(true);
        virtualThreadPool.shutdown();
        platformThreadPool.shutdown();
        printInfo((ThreadPoolExecutor) virtualThreadPool);
        printInfo((ThreadPoolExecutor) platformThreadPool);
    }

    /**
     * 等待线程池关闭并打印线程池信息
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
