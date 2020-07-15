package FileEngine.md5;

import java.io.FileInputStream;
import java.io.IOException;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class Md5Util {
    public static String getMD5(String filePath) {
        byte[] buffer = new byte[8192];
        BigInteger bigInteger;
        int len;
        try (FileInputStream fis = new FileInputStream(filePath)) {
            MessageDigest md = MessageDigest.getInstance("MD5");
            while ((len = fis.read(buffer)) != -1) {
                md.update(buffer, 0, len);
            }
            byte[] b = md.digest();
            bigInteger = new BigInteger(1, b);
            return bigInteger.toString(16);
        } catch (NoSuchAlgorithmException | IOException e) {
            return null;
        }
    }
}
