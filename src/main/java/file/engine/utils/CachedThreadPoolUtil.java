package file.engine.utils;

import file.engine.constant.Constants;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class CachedThreadPoolUtil {
    private final ExecutorService cachedThreadPool = new ThreadPoolExecutor(
            0,
            200,
            60L,
            TimeUnit.SECONDS,
            new SynchronousQueue<>(),
            Executors.defaultThreadFactory());
    private final AtomicBoolean isShutdown = new AtomicBoolean(false);

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
        if (isShutdown.get()) {
            return;
        }
        cachedThreadPool.execute(todo);
    }

    public void shutdown() throws InterruptedException {
        isShutdown.set(true);
        cachedThreadPool.shutdown();
        if (!cachedThreadPool.awaitTermination(Constants.THREAD_POOL_AWAIT_TIMEOUT, TimeUnit.SECONDS)) {
            System.err.println("线程池等待超时");
        }
    }
}
