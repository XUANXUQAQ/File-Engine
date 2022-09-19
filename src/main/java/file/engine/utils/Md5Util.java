package file.engine.utils;

import java.io.FileInputStream;
import java.io.InputStream;
import java.security.MessageDigest;

public class Md5Util {

    /**
     * 获取文件MD5值，固定为32位
     *
     * @param filePath 文件路径
     * @return MD5字符串
     */
    public static String getMD5(String filePath) {
        try (var stream = new FileInputStream(filePath)) {
            return getMD5(stream);
        } catch (Exception e) {
            return "";
        }
    }

    public static String getMD5(InputStream fileStream) {
        byte[] buffer = new byte[8192];
        int len;
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            while ((len = fileStream.read(buffer)) != -1) {
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
        } catch (Exception e) {
            return "";
        }
    }
}
