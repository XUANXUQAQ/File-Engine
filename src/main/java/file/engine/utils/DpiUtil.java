package file.engine.utils;

import java.awt.*;

public class DpiUtil {
    public static double getDpi() {
        return Toolkit.getDefaultToolkit().getScreenResolution() / 96f;
    }
}
