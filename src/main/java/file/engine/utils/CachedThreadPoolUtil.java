package file.engine.utils;

import file.engine.configs.Constants;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public enum CachedThreadPoolUtil {
    INSTANCE;
    private final ExecutorService cachedThreadPool;
    private final AtomicBoolean isShutdown = new AtomicBoolean(false);

    CachedThreadPoolUtil() {
        cachedThreadPool = new ThreadPoolExecutor(
                0,
                200,
                0,
                TimeUnit.SECONDS,
                new SynchronousQueue<>(),
                Thread.ofVirtual().factory());
    }

    public static CachedThreadPoolUtil getInstance() {
        return INSTANCE;
    }

    public boolean isShutdown() {
        return isShutdown.get();
    }

    public <T> Future<T> executeTask(Callable<T> todo) {
        if (isShutdown.get()) {
            return null;
        }
        return cachedThreadPool.submit(todo);
    }

    public void executeTask(Runnable todo) {
        if (isShutdown.get()) {
            return;
        }
        cachedThreadPool.submit(todo);
    }

    public void shutdown() {
        isShutdown.set(true);
        cachedThreadPool.shutdown();
        try {
            if (!cachedThreadPool.awaitTermination(Constants.THREAD_POOL_AWAIT_TIMEOUT, TimeUnit.SECONDS)) {
                System.err.println("线程池等待超时");
                ThreadPoolExecutor tpe = (ThreadPoolExecutor) cachedThreadPool;
                int queueSize = tpe.getQueue().size();
                System.err.println("当前排队线程数：" + queueSize);

                int activeCount = tpe.getActiveCount();
                System.err.println("当前活动线程数：" + activeCount);

                long completedTaskCount = tpe.getCompletedTaskCount();
                System.err.println("执行完成线程数：" + completedTaskCount);

                long taskCount = tpe.getTaskCount();
                System.err.println("总线程数：" + taskCount);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
