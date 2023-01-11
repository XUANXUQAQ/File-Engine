package file.engine.services.utils;

import java.nio.charset.StandardCharsets;

public class StringUtf8SumUtil {

    public static int getStringSum(String fileName) {
        if (fileName == null || fileName.isEmpty()) {
            return 0;
        }
        var bytes = fileName.getBytes(StandardCharsets.UTF_8);
        int sum = 0;
        for (byte aByte : bytes) {
            if (aByte > 0) {
                sum += aByte;
            }
        }
        return sum;
    }
}
