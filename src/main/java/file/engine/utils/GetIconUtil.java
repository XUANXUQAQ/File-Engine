package file.engine.utils;

import file.engine.IsDebug;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.File;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author XUANXU
 */
public class GetIconUtil {
    private static final FileSystemView FILE_SYSTEM_VIEW = FileSystemView.getFileSystemView();
    private volatile ImageIcon dllImageIcon;
    private volatile ImageIcon folderImageIcon;
    private volatile ImageIcon txtImageIcon;
    private volatile ImageIcon helpIcon;
    private volatile ImageIcon updateIcon;
    private volatile ImageIcon blankIcon;
    private volatile ImageIcon recycleBin;
    private final AtomicBoolean isInitialized = new AtomicBoolean(false);

    private static volatile GetIconUtil INSTANCE = null;

    private GetIconUtil() {}

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

    private ImageIcon changeIcon(ImageIcon icon, int width, int height) {
        try {
            Image image = icon.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT);
            return new ImageIcon(image);
        } catch (NullPointerException e) {
            return null;
        }
    }

    private void initIconCache(int width, int height) {
        //添加其他常量图标  changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("")), width, height); 获取应用程序时使用
        //或者 changeIcon(new ImageIcon(GetIconUtil.class.getResource("")), width, height); 本地图标时使用
        dllImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows\\System32\\sysmain.dll")), width, height);
        folderImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows")), width, height);
        txtImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user\\cmds.txt")), width, height);
        blankIcon = changeIcon(new ImageIcon(GetIconUtil.class.getResource("/icons/blank.png")), width, height);
        recycleBin = changeIcon(new ImageIcon(GetIconUtil.class.getResource("/icons/recyclebin.png")), width, height);
        updateIcon = changeIcon(new ImageIcon(GetIconUtil.class.getResource("/icons/update.png")), width, height);
        helpIcon = changeIcon(new ImageIcon(GetIconUtil.class.getResource("/icons/help.png")), width, height);
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
                return recycleBin;
            case "update":
                return updateIcon;
            case "help":
                return helpIcon;
            case "version":
                return blankIcon;
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
            return blankIcon;
        }
        File f = new File(path);
        path = path.toLowerCase();
        if (f.exists()) {
            //已保存的常量图标
            if (path.endsWith(".dll") || path.endsWith(".sys")) {
                return dllImageIcon;
            }
            if (path.endsWith(".txt")) {
                return txtImageIcon;
            }
            //检测是否为文件夹
            if (f.isDirectory()) {
                return folderImageIcon;
            } else {
                return changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(f), width, height);
            }
        } else {
            return blankIcon;
        }
    }
}
