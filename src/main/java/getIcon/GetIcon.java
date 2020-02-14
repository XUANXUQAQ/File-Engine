package getIcon;

import javax.swing.*;
import java.io.File;
import java.io.FileNotFoundException;

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
                sun.awt.shell.ShellFolder sf = sun.awt.shell.ShellFolder.getShellFolder(f);
                return new ImageIcon(sf.getIcon(true));
            } catch (FileNotFoundException | NullPointerException ignored) {

            }
        }
        return null;
    }
}
