package file.engine.utils;

import file.engine.utils.system.properties.IsDebug;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.File;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author XUANXU
 */
public class GetIconUtil {
    private static final FileSystemView FILE_SYSTEM_VIEW = FileSystemView.getFileSystemView();
    private final AtomicBoolean isInitialized = new AtomicBoolean(false);
    private final ConcurrentHashMap<String, ImageIcon> iconMap = new ConcurrentHashMap<>();

    private static volatile GetIconUtil INSTANCE = null;

    private GetIconUtil() {
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
        Image image = icon.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT);
        return new ImageIcon(image);
    }

    private void initIconCache(int width, int height) {
        //添加其他常量图标  changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("")), width, height); 获取应用程序时使用
        //或者 changeIcon(new ImageIcon(GetIconUtil.class.getResource("")), width, height); 本地图标时使用
        iconMap.put("dllImageIcon", Objects.requireNonNull(changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows\\System32\\sysmain.dll")), width, height)));
        iconMap.put("folderImageIcon", Objects.requireNonNull(changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows")), width, height)));
        iconMap.put("txtImageIcon", Objects.requireNonNull(changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user\\cmds.txt")), width, height)));
        iconMap.put("blankIcon", Objects.requireNonNull(changeIcon(new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/blank.png"))), width, height)));
        iconMap.put("recycleBin", Objects.requireNonNull(changeIcon(new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/recyclebin.png"))), width, height)));
        iconMap.put("updateIcon", Objects.requireNonNull(changeIcon(new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/update.png"))), width, height)));
        iconMap.put("helpIcon", Objects.requireNonNull(changeIcon(new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/help.png"))), width, height)));
        iconMap.put("completeIcon", Objects.requireNonNull(changeIcon(new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/complete.png"))), width, height)));
        iconMap.put("loadingIcon", Objects.requireNonNull(changeIcon(new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/loading.gif"))), width, height)));
    }

    public ImageIcon getIcon(String key) {
        return iconMap.get(key);
    }

    public ImageIcon getCommandIcon(String commandName, int width, int height) {
        if (!isInitialized.get()) {
            initIconCache(width, height);
            isInitialized.set(true);
        }
        if (commandName == null || commandName.isEmpty()) {
            if (IsDebug.isDebug()) {
                System.err.println("Command is empty");
            }
            return null;
        }
        switch (commandName) {
            case "clearbin":
                return iconMap.get("recycleBin");
            case "update":
            case "clearUpdate":
                return iconMap.get("updateIcon");
            case "help":
                return iconMap.get("helpIcon");
            case "version":
                return iconMap.get("blankIcon");
            default:
                return null;
        }
    }

    public ImageIcon getBigIcon(String path, int width, int height) {
        if (!isInitialized.get()) {
            initIconCache(width, height);
            isInitialized.set(true);
        }
        if (path == null || path.isEmpty()) {
            return iconMap.get("blankIcon");
        }
        File f = new File(path);
        path = path.toLowerCase();
        if (f.exists()) {
            //已保存的常量图标
            if (path.endsWith(".dll") || path.endsWith(".sys")) {
                return iconMap.get("dllImageIcon");
            }
            if (path.endsWith(".txt")) {
                return iconMap.get("txtImageIcon");
            }
            //检测是否为文件夹
            if (f.isDirectory()) {
                return iconMap.get("folderImageIcon");
            } else {
                return changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(f), width, height);
            }
        } else {
            return iconMap.get("blankIcon");
        }
    }
}
