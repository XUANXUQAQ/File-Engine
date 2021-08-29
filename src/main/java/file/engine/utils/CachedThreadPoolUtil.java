package file.engine.utils;

import file.engine.configs.Constants;
import file.engine.utils.system.properties.IsDebug;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public enum CachedThreadPoolUtil {
    INSTANCE;
    private final ExecutorService cachedThreadPool = new ThreadPoolExecutor(
            0,
            200,
            60L,
            TimeUnit.SECONDS,
            new SynchronousQueue<>(),
            new NamedThreadFactory());
    private final AtomicBoolean isShutdown = new AtomicBoolean(false);

    public static CachedThreadPoolUtil getInstance() {
        return INSTANCE;
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

    private static class NamedThreadFactory implements ThreadFactory {
        private final ThreadGroup group;
        private static final AtomicInteger poolNumber = new AtomicInteger(1);
        private final AtomicInteger threadNumber = new AtomicInteger(1);

        NamedThreadFactory() {
            SecurityManager s = System.getSecurityManager();
            group = (s != null) ? s.getThreadGroup() : Thread.currentThread().getThreadGroup();
        }

        @Override
        public Thread newThread(Runnable r) {
            String name = "pool-" + poolNumber.incrementAndGet() + "-thread-" + threadNumber.getAndIncrement();
            if (IsDebug.isDebug()) {
                name = getStackTraceElement().toString() + threadNumber.getAndIncrement();
            }
            Thread t = new Thread(group, r, name, 0);
            if (t.isDaemon())
                t.setDaemon(false);
            if (t.getPriority() != Thread.NORM_PRIORITY)
                t.setPriority(Thread.NORM_PRIORITY);
            return t;
        }

        /**
         * 用于在debug时查看在哪个位置发出的任务
         * 由于执行任务的调用栈长度超过3，所以不会出现数组越界
         *
         * @return stackTraceElement
         */
        private StackTraceElement getStackTraceElement() {
            StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
            return stacktrace[8];
        }
    }
}
