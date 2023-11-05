package file.engine.utils;

import lombok.SneakyThrows;

import java.awt.*;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Path;

public class DpiUtil {

    private static volatile float dpi = 1;
    private static volatile Dimension screenDimension = new Dimension();
    private static volatile long getDpiTime = 0;
    private static final String DPI_CHECK_PROC = "user/getDpi.exe";

    public static float getDpi() {
        return getDpi(null);
    }

    @SneakyThrows
    public static float getDpi(Dimension dimension) {
        if (System.currentTimeMillis() - getDpiTime <= 1000) {
            if (dimension != null) {
                dimension.setSize(screenDimension);
            }
            return dpi;
        }
        getDpiTime = System.currentTimeMillis();
        Process exec = Runtime.getRuntime().exec(new String[]{Path.of(DPI_CHECK_PROC).toAbsolutePath().toString()});
        ProcessUtil.waitForProcess("getDpi.exe", 10);
        try (var reader = new BufferedReader(new InputStreamReader(exec.getInputStream()))) {
            String dpiStr = reader.readLine();
            if (dpiStr != null) {
                dpi = Float.parseFloat(dpiStr);
            }
            if (dimension != null) {
                String screenWidth = reader.readLine();
                int width = Integer.parseInt(screenWidth);
                String ScreenHeight = reader.readLine();
                int height = Integer.parseInt(ScreenHeight);
                dimension.setSize(width, height);
                screenDimension = new Dimension(width, height);
            }
        }
        return dpi;
    }
}
