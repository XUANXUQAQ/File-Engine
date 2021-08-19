package file.engine.utils.file;

import java.io.File;

public class FilePathUtil {

    public static String getParentPath(String path) {
        File f = new File(path);
        return f.getParentFile().getAbsolutePath();
    }

    public static boolean isFile(String text) {
        File file = new File(text);
        return file.isFile();
    }

    public static String getFileName(String path) {
        if (path != null) {
            int index = path.lastIndexOf(File.separator);
            return path.substring(index + 1);
        }
        return "";
    }
}
