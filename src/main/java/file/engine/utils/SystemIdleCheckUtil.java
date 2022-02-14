package file.engine.utils;

import file.engine.event.handler.EventManagement;

import java.awt.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class SystemIdleCheckUtil {

    private static final int THRESHOLD = 120;

    static class CursorCount {
        private static final AtomicInteger count = new AtomicInteger();
    }

    /**
     * 检测鼠标是否在两分钟内都未移动
     *
     * @return true if cursor not move in 2 minutes
     */
    public static boolean isCursorLongTimeNotMove() {
        return CursorCount.count.get() > THRESHOLD;
    }

    /**
     * 持续检测鼠标位置，如果在一秒内移动过，则重置CursorCount.count
     */
    private static void startGetCursorPosTimer() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            try {
                Point lastPoint = new Point();
                while (eventManagement.isNotMainExit()) {
                    Point point = getCursorPoint();
                    if (!point.equals(lastPoint)) {
                        CursorCount.count.set(0);
                    }
                    lastPoint.setLocation(point);
                    TimeUnit.SECONDS.sleep(1);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * 获取鼠标位置
     * @return Point
     */
    private static Point getCursorPoint() {
        return java.awt.MouseInfo.getPointerInfo().getLocation();
    }

    /**
     * 开启检测
     */
    public static void start() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement instance = EventManagement.getInstance();
            try {
                while (instance.isNotMainExit()) {
                    if (CursorCount.count.get() <= THRESHOLD) {
                        CursorCount.count.incrementAndGet();
                    }
                    TimeUnit.SECONDS.sleep(1);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        startGetCursorPosTimer();
    }
}
