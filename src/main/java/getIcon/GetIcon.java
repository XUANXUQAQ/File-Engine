package getIcon;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.io.File;

public class GetIcon {

    /**
     * ��ȡ��ͼ��
     *
     * @param path ·��
     * @return Image or null
     */
    public static Icon getBigIcon(String path) {
        File f = new File(path);
        if (f.exists()) {
            try {
                FileSystemView fsv = FileSystemView.getFileSystemView();
                return fsv.getSystemIcon(f);
                /*sun.awt.shell.ShellFolder sf = sun.awt.shell.ShellFolder.getShellFolder(f);
                return new ImageIcon(sf.getIcon(true));*/
            } catch (NullPointerException ignored) {

            }
        }
        return null;
    }
}
