package file.engine.utils.file;

import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;

@Slf4j
public class FileUtil {

    public static boolean isFileExist(String path) {
        try {
            return Files.exists(Path.of(path));
        } catch (InvalidPathException ignored) {
            // ignored
        } catch (Exception e) {
            log.error("error: {}", e.getMessage(), e);
        }
        return false;
    }

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
        if (content == null) {
            return;
        }
        for (File temp : content) {
            //直接删除文件
            if (temp.isDirectory()) {//判断是否是目录
                deleteDir(temp);//递归调用，删除目录里的内容
            }
            //删除空目录
            if (!temp.delete()) {
                log.error("Failed to delete " + temp.getAbsolutePath());
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

    public static boolean isDir(String path) {
        return Files.isDirectory(Path.of(path));
    }

    public static boolean isFile(String path) {
        return Files.isRegularFile(Path.of(path));
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
            bis.transferTo(bos);
        } catch (IOException e) {
            log.error("error: {}", e.getMessage(), e);
        }
    }

    public static void copyFile(File file, File dest) {
        try (FileInputStream inputStream = new FileInputStream(file)) {
            copyFile(inputStream, dest);
        } catch (IOException e) {
            log.error("error: {}", e.getMessage(), e);
        }
    }

    public static void copyDir(String sourcePath, String newPath) {
        File source = new File(sourcePath);
        File dest = new File(newPath);
        String[] filePath = source.list();
        //获取该文件夹下的所有文件以及目录的名字
        if (filePath == null || filePath.length == 0) {
            return;
        }
        if (!dest.exists()) {
            dest.mkdirs();
        }
        for (String temp : filePath) {
            //查看其数组中每一个是文件还是文件夹
            if (isDir(sourcePath + File.separator + temp)) {
                //为文件夹，进行递归
                copyDir(sourcePath + File.separator + temp, newPath + File.separator + temp);
            } else {
                //为文件则进行拷贝
                copyFile(new File(sourcePath + File.separator + temp), new File(newPath + File.separator + temp));
            }
        }
    }
}
