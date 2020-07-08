package getIcon;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.File;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class GetIcon {
    private static ConcurrentHashMap<String, ImageIcon> iconCache = new ConcurrentHashMap<>(1000);
    private static FileSystemView fsv = FileSystemView.getFileSystemView();
    private static AtomicInteger cacheNum = new AtomicInteger(0);
    private static ImageIcon dllImageIcon;
    private static ImageIcon folderImageIcon;
    private static ImageIcon txtImageIcon;
    private static volatile boolean isInitialized = false;

    private static ImageIcon changeIcon(ImageIcon icon, int width, int height) {
        try {
            Image image = icon.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT);
            return new ImageIcon(image);
        } catch (NullPointerException e) {
            return null;
        }
    }

    public static void initIconCache(int width, int height) {
        //  changeIcon((ImageIcon) fsv.getSystemIcon(new File("")), width, height);
        dllImageIcon = changeIcon((ImageIcon) fsv.getSystemIcon(new File("C:\\Windows\\System32\\sysmain.dll")), width, height);
        folderImageIcon = changeIcon((ImageIcon) fsv.getSystemIcon(new File("C:\\Windows")), width, height);
        txtImageIcon = changeIcon((ImageIcon) fsv.getSystemIcon(new File("user\\cmds.txt")), width, height);
    }

    public static ImageIcon getBigIcon(String path, int width, int height) {
        if (!isInitialized) {
            initIconCache(width, height);
            isInitialized = true;
        }
        File f = new File(path);
        String lowerCase = path.toLowerCase();
        if (f.exists()) {
            if (lowerCase.endsWith("dll")) {
                return dllImageIcon;
            }
            if (lowerCase.endsWith("txt")) {
                return txtImageIcon;
            }
            if (f.isDirectory()) {
                return folderImageIcon;
            } else {
                ImageIcon imageIcon = iconCache.get(path);
                if (imageIcon == null) {
                    imageIcon = changeIcon((ImageIcon) fsv.getSystemIcon(f), width, height);
                }
                if (imageIcon != null) {
                    if (cacheNum.get() < 1000) {
                        if (!iconCache.containsKey(path)) {
                            iconCache.put(path, imageIcon);
                            cacheNum.incrementAndGet();
                        }
                    }
                }
                return imageIcon;
            }
        }
        return null;
    }
}
