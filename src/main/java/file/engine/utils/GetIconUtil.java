package file.engine.utils;

import file.engine.utils.system.properties.IsDebug;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.File;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author XUANXU
 */
public class GetIconUtil {
    private static final FileSystemView FILE_SYSTEM_VIEW = FileSystemView.getFileSystemView();
    private final ConcurrentHashMap<String, ImageIcon> iconMap = new ConcurrentHashMap<>();

    private static volatile GetIconUtil INSTANCE = null;

    private GetIconUtil() {
        initIconCache();
    }

    public static GetIconUtil getInstance() {
        if (INSTANCE == null) {
            synchronized (GetIconUtil.class) {
                if (INSTANCE == null) {
                    INSTANCE = new GetIconUtil();
                }
            }
        }
        return INSTANCE;
    }

    public ImageIcon changeIcon(ImageIcon icon, int width, int height) {
        if (icon == null) {
            return null;
        }
        Image image = icon.getImage().getScaledInstance(width, height, Image.SCALE_FAST);
        return new ImageIcon(image);
    }

    private void initIconCache() {
        iconMap.put("dllImageIcon", Objects.requireNonNull((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows\\System32\\sysmain.dll"))));
        iconMap.put("folderImageIcon", Objects.requireNonNull((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows"))));
        iconMap.put("txtImageIcon", Objects.requireNonNull((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user\\cmds.txt"))));
        iconMap.put("blankIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/blank.png"))));
        iconMap.put("recycleBin", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/recyclebin.png"))));
        iconMap.put("updateIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/update.png"))));
        iconMap.put("helpIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/help.png"))));
        iconMap.put("completeIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/complete.png"))));
        iconMap.put("loadingIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/loading.gif"))));
    }

    public ImageIcon getCommandIcon(String commandName, int width, int height) {
        if (commandName == null || commandName.isEmpty()) {
            if (IsDebug.isDebug()) {
                System.err.println("Command is empty");
            }
            return null;
        }
        switch (commandName) {
            case "clearbin":
                return changeIcon(iconMap.get("recycleBin"), width, height);
            case "update":
            case "clearUpdate":
                return changeIcon(iconMap.get("updateIcon"), width, height);
            case "help":
                return changeIcon(iconMap.get("helpIcon"), width, height);
            case "version":
                return changeIcon(iconMap.get("blankIcon"), width, height);
            default:
                return null;
        }
    }

    public ImageIcon getBigIcon(String pathOrKey, int width, int height) {
        if (pathOrKey == null || pathOrKey.isEmpty()) {
            return changeIcon(iconMap.get("blankIcon"), width, height);
        }
        if (iconMap.containsKey(pathOrKey)) {
            return changeIcon(iconMap.get(pathOrKey), width, height);
        }
        File f = new File(pathOrKey);
        pathOrKey = pathOrKey.toLowerCase();
        if (f.exists()) {
            //已保存的常量图标
            if (pathOrKey.endsWith(".dll") || pathOrKey.endsWith(".sys")) {
                return changeIcon(iconMap.get("dllImageIcon"), width, height);
            }
            if (pathOrKey.endsWith(".txt")) {
                return changeIcon(iconMap.get("txtImageIcon"), width, height);
            }
            //检测是否为文件夹
            if (f.isDirectory()) {
                return changeIcon(iconMap.get("folderImageIcon"), width, height);
            } else {
                return changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(f), width, height);
            }
        } else {
            return changeIcon(iconMap.get("blankIcon"), width, height);
        }
    }
}
