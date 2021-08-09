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

    /**
     * 移动鼠标点击
     * @param x x坐标
     * @param y y坐标
     * @param count 次数
     * @param mouseButtonNum 需要点击的案件
     * @see java.awt.event.InputEvent
     */
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

    /**
     * 键盘点击
     * @param keyCodes 键盘案件
     * @see java.awt.event.KeyEvent
     */
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
