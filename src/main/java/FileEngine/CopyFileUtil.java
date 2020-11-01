package FileEngine;

import java.io.*;

public class CopyFileUtil {
    public static void copyFile(InputStream source, File dest) {
        try (BufferedInputStream bis = new BufferedInputStream(source);
             BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(dest))) {
            byte[] buffer = new byte[8192];
            int count = bis.read(buffer);
            while (count != -1) {
                //使用缓冲流写数据
                bos.write(buffer, 0, count);
                //刷新
                bos.flush();
                count = bis.read(buffer);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void copyFile(File file, File dest) {
        try (FileInputStream inputStream = new FileInputStream(file)) {
            copyFile(inputStream, dest);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
