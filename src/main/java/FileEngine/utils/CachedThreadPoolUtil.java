package FileEngine.utils;

import FileEngine.IsDebug;
import FileEngine.annotation.EventRegister;
import FileEngine.eventHandler.EventManagement;
import FileEngine.eventHandler.impl.stop.StopEvent;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class CachedThreadPoolUtil {
    private final ExecutorService cachedThreadPool = new ThreadPoolExecutor(
            0,
            200,
            60L,
            TimeUnit.SECONDS,
            new SynchronousQueue<>());
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

    private void shutdown() throws InterruptedException {
        isShutdown.set(true);
        cachedThreadPool.shutdown();
        int count = 0;
        final int maxWaitTime = 100 * 5;   //最大等待5s
        while (!cachedThreadPool.isTerminated() && count < maxWaitTime) {
            count++;
            TimeUnit.MILLISECONDS.sleep(10);
        }
        if (IsDebug.isDebug()) {
            System.err.println("线程池已关闭");
        }
    }

    @EventRegister
    @SuppressWarnings("unused")
    public static void registerShutdownEvent() {
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.registerListener(StopEvent.class, () -> {
            try {
                getInstance().shutdown();
            } catch (InterruptedException ignored) {
            }
        });
    }
}
