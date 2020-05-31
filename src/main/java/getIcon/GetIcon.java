package getIcon;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.io.File;

public class GetIcon {

    /**
     * 获取大图标
     *
     * @param path 路径
     * @return Image or null
     */
    public static Icon getBigIcon(String path) {
        File f = new File(path);
        if (f.exists()) {
            try {
                FileSystemView fsv = FileSystemView.getFileSystemView();
                return fsv.getSystemIcon(f);
            } catch (NullPointerException ignored) {

            }
        }
        return null;
    }
}
