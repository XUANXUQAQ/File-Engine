package file.engine.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class DpiUtil {

    public static double getDpi() {
        String proc = "user/getDpi.exe";
        double dpi = 1;
        try {
            Process exec = Runtime.getRuntime().exec(proc);
            ProcessUtil.waitForProcess("getDpi.exe");
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(exec.getInputStream()))) {
                String dpiStr = reader.readLine();
                dpi = Double.parseDouble(dpiStr);
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return dpi;
    }
}
