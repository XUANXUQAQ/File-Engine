package FileEngine.threadPool;

import java.util.concurrent.*;

public class CachedThreadPool {
    private final ExecutorService cachedThreadPool = new ThreadPoolExecutor(
            0,
            100,
            60L,
            TimeUnit.SECONDS,
            new SynchronousQueue<>());


    private static class CachedThreadPoolBuilder {
        private static final CachedThreadPool INSTANCE = new CachedThreadPool();
    }

    private CachedThreadPool() {
    }

    public static CachedThreadPool getInstance() {
        return CachedThreadPoolBuilder.INSTANCE;
    }

    public void executeTask(Runnable todo) {
        cachedThreadPool.execute(todo);
    }
}
