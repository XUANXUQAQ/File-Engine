package file.engine.utils;

import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
public enum ThreadPoolUtil {
    INSTANCE;
    private static final int THREAD_POOL_AWAIT_TIMEOUT = 60;
    private final ExecutorService cachedThreadPool;
    private final ExecutorService virtualThreadPool;
    private final AtomicBoolean isShutdown = new AtomicBoolean(false);

    ThreadPoolUtil() {
        cachedThreadPool = new ThreadPoolExecutor(
                0,
                1000,
                60,
                TimeUnit.SECONDS,
                new SynchronousQueue<>(),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.CallerRunsPolicy()
        );
        virtualThreadPool = Executors.newVirtualThreadPerTaskExecutor();
    }

    public static ThreadPoolUtil getInstance() {
        return INSTANCE;
    }

    public boolean isShutdown() {
        return isShutdown.get();
    }

    private <T> Future<T> executeTaskPlatform(Callable<T> task) {
        return cachedThreadPool.submit(task);
    }

    private <T> Future<T> executeTaskVirtual(Callable<T> task) {
        return virtualThreadPool.submit(task);
    }

    private void executeTaskPlatform(Runnable task) {
        cachedThreadPool.submit(task);
    }

    private void executeTaskVirtual(Runnable task) {
        virtualThreadPool.submit(task);
    }

    /**
     * 提交任务
     *
     * @param task            任务
     * @param isVirtualThread 是否使用虚拟线程
     * @return Future
     */
    public <T> Future<T> executeTask(Callable<T> task, boolean isVirtualThread) {
        if (isShutdown.get()) {
            return null;
        }
        if (isVirtualThread) {
            return executeTaskVirtual(task);
        } else {
            return executeTaskPlatform(task);
        }
    }

    public <T> Future<T> executeTask(Callable<T> task) {
        return executeTask(task, true);
    }

    /**
     * 提交任务
     *
     * @param task            任务
     * @param isVirtualThread 是否使用虚拟线程
     */
    public void executeTask(Runnable task, boolean isVirtualThread) {
        if (isShutdown.get()) {
            return;
        }
        if (isVirtualThread) {
            executeTaskVirtual(task);
        } else {
            executeTaskPlatform(task);
        }
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
        executeTask(task, true);
    }

    /**
     * 关闭线程池病等待
     */
    public void shutdown() {
        isShutdown.set(true);
        cachedThreadPool.shutdown();
        virtualThreadPool.shutdown();
        printInfo((ThreadPoolExecutor) cachedThreadPool);
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
            log.error("error: {}", e.getMessage(), e);
        }
    }
}
