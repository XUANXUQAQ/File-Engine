package getIcon;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.io.File;
import java.util.concurrent.ConcurrentHashMap;

public class GetIcon {
    private static ConcurrentHashMap<String, Icon> iconCache = new ConcurrentHashMap<>(300);
    private static FileSystemView fsv = FileSystemView.getFileSystemView();
    private static boolean isFull = false;
    private static Icon dllIcon = fsv.getSystemIcon(new File("C:\\Windows\\System32\\sysmain.dll"));
    private static Icon folderIcon = fsv.getSystemIcon(new File("C:\\Windows"));

    public static Icon getBigIcon(String path) {
        if (path.toLowerCase().endsWith("dll")) {
            return dllIcon;
        }
        File f = new File(path);
        if (f.exists()) {
            if (f.isDirectory()) {
                return folderIcon;
            } else {
                try {
                    Icon icon = iconCache.get(path);
                    if (icon == null) {
                        icon = fsv.getSystemIcon(f);
                        if (!isFull) {
                            if (iconCache.size() < 300) {
                                iconCache.put(path, icon);
                            } else {
                                isFull = true;
                            }
                        }
                    }
                    return icon;
                } catch (NullPointerException ignored) {

                }
            }
        }
        return null;
    }
}
