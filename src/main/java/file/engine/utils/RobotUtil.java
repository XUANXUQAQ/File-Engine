package file.engine.utils;

import java.awt.*;

public class RobotUtil {
    private static Robot robot;

    private static volatile RobotUtil INSTANCE = null;

    private static final int delayMills = 5;

    private RobotUtil() {
        try {
            robot = new Robot();
        } catch (AWTException e) {
            e.printStackTrace();
        }
    }

    public static RobotUtil getInstance() {
        if (INSTANCE == null) {
            synchronized (RobotUtil.class) {
                if (INSTANCE == null) {
                    INSTANCE = new RobotUtil();
                }
            }
        }
        return INSTANCE;
    }

    public void mouseClicked(int x, int y, int count, int mouseButtonNum) {
        if (robot != null) {
            double originX, originY;
            Point point = MouseInfo.getPointerInfo().getLocation();
            originX = point.getX();
            originY = point.getY();
            robot.mouseMove(x, y);
            for (int i = 0; i < count; ++i) {
                robot.mousePress(mouseButtonNum);
                robot.delay(delayMills);
                robot.mouseRelease(mouseButtonNum);
            }
            //鼠标归位
            robot.mouseMove((int) originX, (int) originY);
        }
    }

    public void keyTyped(int... keyCodes) {
        if (robot != null) {
            //全部点击
            for (int each : keyCodes) {
                robot.keyPress(each);
                robot.delay(delayMills);
            }
            //全部释放
            for (int each : keyCodes) {
                robot.keyRelease(each);
                robot.delay(delayMills);
            }
        }
    }
}
