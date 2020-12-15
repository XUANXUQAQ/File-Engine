package FileEngine.threadPool;

import java.util.concurrent.*;

public class CachedThreadPool {
    private final ExecutorService cachedThreadPool = new ThreadPoolExecutor(
            0,
            200,
            60L,
            TimeUnit.SECONDS,
            new SynchronousQueue<>());

    private static volatile CachedThreadPool INSTANCE = null;

    private CachedThreadPool() {}

    public static CachedThreadPool getInstance() {
        if (INSTANCE == null) {
            synchronized (CachedThreadPool.class) {
                if (INSTANCE == null) {
                    INSTANCE = new CachedThreadPool();
                }
            }
        }
        return INSTANCE;
    }

    public void executeTask(Runnable todo) {
        cachedThreadPool.execute(todo);
    }
}
