package FileEngine.md5;

import java.io.FileInputStream;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class Md5Util {
    public static String getMD5(String filePath) {
        byte[] buffer = new byte[8192];
        int len;
        try (FileInputStream fis = new FileInputStream(filePath)) {
            MessageDigest md = MessageDigest.getInstance("MD5");
            while ((len = fis.read(buffer)) != -1) {
                md.update(buffer, 0, len);
            }
            byte[] b = md.digest();
            StringBuilder hexValue = new StringBuilder();
            for (byte value : b) {
                int val = ((int) value) & 0xff;
                if (val < 16) {
                    hexValue.append("0");
                }
                //这里借助了Integer类的方法实现16进制的转换
                hexValue.append(Integer.toHexString(val));
            }
            return hexValue.toString();
        } catch (NoSuchAlgorithmException | IOException e) {
            return null;
        }
    }
}
