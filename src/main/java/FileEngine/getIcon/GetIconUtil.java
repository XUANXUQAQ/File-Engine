package FileEngine.getIcon;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.File;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author XUANXU
 */
public class GetIconUtil {
    private final ConcurrentHashMap<String, ImageIcon> ICON_CACHE_MAP = new ConcurrentHashMap<>(1000);
    private final FileSystemView FILE_SYSTEM_VIEW = FileSystemView.getFileSystemView();
    private final AtomicInteger CACHE_NUM = new AtomicInteger(0);
    private ImageIcon dllImageIcon;
    private ImageIcon folderImageIcon;
    private ImageIcon txtImageIcon;
    private ImageIcon vbsImageIcon;
    private volatile boolean isInitialized = false;

    private static class GetIconUtilBuilder {
        private static final GetIconUtil INSTANCE = new GetIconUtil();
    }

    public static GetIconUtil getInstance() {
        return GetIconUtilBuilder.INSTANCE;
    }

    private ImageIcon changeIcon(ImageIcon icon, int width, int height) {
        try {
            Image image = icon.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT);
            return new ImageIcon(image);
        } catch (NullPointerException e) {
            return null;
        }
    }

    private void initIconCache(int width, int height) {
        //添加其他常量图标  changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("")), width, height);
        dllImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows\\System32\\sysmain.dll")), width, height);
        folderImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows")), width, height);
        txtImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user\\cmds.txt")), width, height);
        vbsImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user\\shortcutGenerator.vbs")), width, height);
    }

    public ImageIcon getBigIcon(String path, int width, int height) {
        if (!isInitialized) {
            initIconCache(width, height);
            isInitialized = true;
        }
        if (path == null) {
            return null;
        }
        File f = new File(path);
        String fName = f.getName();
        if (f.exists()) {
            if (fName.endsWith(".dll")) {
                return dllImageIcon;
            }
            if (fName.endsWith(".txt")) {
                return txtImageIcon;
            }
            if (fName.endsWith(".vbs")) {
                return vbsImageIcon;
            }
            if (f.isDirectory()) {
                return folderImageIcon;
            } else {
                ImageIcon imageIcon = ICON_CACHE_MAP.get(path);
                if (imageIcon == null) {
                    imageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(f), width, height);
                }
                if (imageIcon != null) {
                    if (CACHE_NUM.get() < 1000) {
                        ICON_CACHE_MAP.put(path, imageIcon);
                        CACHE_NUM.incrementAndGet();
                    }
                }
                return imageIcon;
            }
        }
        return null;
    }
}
