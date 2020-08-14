package FileEngine.robotUtil;

import java.awt.*;
import java.util.concurrent.TimeUnit;

public class RobotUtil {
    private static Robot robot = null;

    private static class MouseClickBuilder {
        private static final RobotUtil INSTANCE = new RobotUtil();
    }

    private RobotUtil() {
        try {
            robot = new Robot();
        } catch (AWTException e) {
            e.printStackTrace();
        }
    }

    public static RobotUtil getInstance() {
        return MouseClickBuilder.INSTANCE;
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
            }
            //保持按下状态100毫秒
            try {
                TimeUnit.MILLISECONDS.sleep(100);
            } catch (InterruptedException ignored) {
            }
            //全部释放
            for (int each : keyCodes) {
                robot.keyRelease(each);
            }
        }
    }
}
