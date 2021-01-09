package FileEngine.utils;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class CachedThreadPoolUtil {
    private final ExecutorService cachedThreadPool = new ThreadPoolExecutor(
            0,
            200,
            15L,
            TimeUnit.SECONDS,
            new SynchronousQueue<>());

    private static volatile CachedThreadPoolUtil INSTANCE = null;

    private CachedThreadPoolUtil() {}

    public static CachedThreadPoolUtil getInstance() {
        if (INSTANCE == null) {
            synchronized (CachedThreadPoolUtil.class) {
                if (INSTANCE == null) {
                    INSTANCE = new CachedThreadPoolUtil();
                }
            }
        }
        return INSTANCE;
    }

    public void executeTask(Runnable todo) {
        cachedThreadPool.execute(todo);
    }

    public void shutdown() {
        cachedThreadPool.shutdown();
    }
}
