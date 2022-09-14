package file.engine.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class DpiUtil {

    private static volatile double dpi = 1;
    private static volatile long getDpiTime = 0;

    public static double getDpi() {
        if (System.currentTimeMillis() - getDpiTime <= 5000) {
            return dpi;
        }
        getDpiTime = System.currentTimeMillis();
        String proc = "user/getDpi.exe";
        try {
            Process exec = Runtime.getRuntime().exec(proc);
            ProcessUtil.waitForProcess("getDpi.exe", 10);
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(exec.getInputStream()))) {
                String dpiStr = reader.readLine();
                if (dpiStr != null) {
                    dpi = Double.parseDouble(dpiStr);
                }
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return dpi;
    }
}
