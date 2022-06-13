package file.engine.utils.file;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;

public class FileUtil {

    /**
     * 清空一个目录，不删除目录本身
     *
     * @param file 目录文件
     */
    public static void deleteDir(File file) {
        if (!file.exists()) {
            return;
        }
        File[] content = file.listFiles();//取得当前目录下所有文件和文件夹
        if (content == null || content.length == 0) {
            return;
        }
        for (File temp : content) {
            //直接删除文件
            if (temp.isDirectory()) {//判断是否是目录
                deleteDir(temp);//递归调用，删除目录里的内容
            }
            //删除空目录
            if (!temp.delete()) {
                System.err.println("Failed to delete " + temp.getAbsolutePath());
            }
        }
    }

    public static String getParentPath(String path) {
        if (path != null) {
            int index = path.lastIndexOf(File.separatorChar);
            return path.substring(0, index);
        }
        return "";
    }

    public static boolean isFile(String text) {
        return Files.isRegularFile(Path.of(text));
    }

    public static String getFileName(String path) {
        if (path != null) {
            int index = path.lastIndexOf(File.separatorChar);
            return path.substring(index + 1);
        }
        return "";
    }

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
