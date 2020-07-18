package FileEngine.threadPool;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CachedThreadPool {
    private final ExecutorService cachedThreadPool = Executors.newCachedThreadPool();

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
