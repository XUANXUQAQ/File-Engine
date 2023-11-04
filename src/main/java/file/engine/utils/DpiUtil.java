package file.engine.utils;

import lombok.SneakyThrows;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Path;

public class DpiUtil {

    private static volatile double dpi = 1;
    private static volatile long getDpiTime = 0;
    private static final String DPI_CHECK_PROC = "user/getDpi.exe";

    @SneakyThrows
    public static double getDpi() {
        if (System.currentTimeMillis() - getDpiTime <= 1000) {
            return dpi;
        }
        getDpiTime = System.currentTimeMillis();
        Process exec = Runtime.getRuntime().exec(new String[]{Path.of(DPI_CHECK_PROC).toAbsolutePath().toString()});
        ProcessUtil.waitForProcess("getDpi.exe", 10);
        try (var reader = new BufferedReader(new InputStreamReader(exec.getInputStream()))) {
            String dpiStr = reader.readLine();
            if (dpiStr != null) {
                dpi = Double.parseDouble(dpiStr);
            }
        }
        return dpi;
    }
}
