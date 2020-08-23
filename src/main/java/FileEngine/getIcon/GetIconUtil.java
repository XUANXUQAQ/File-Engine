package FileEngine.getIcon;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.File;

/**
 * @author XUANXU
 */
public class GetIconUtil {
    private static final FileSystemView FILE_SYSTEM_VIEW = FileSystemView.getFileSystemView();
    private ImageIcon dllImageIcon;
    private ImageIcon folderImageIcon;
    private ImageIcon txtImageIcon;
    private ImageIcon vbsImageIcon;
    private ImageIcon helpIcon;
    private ImageIcon updateIcon;
    private ImageIcon blankIcon;
    private ImageIcon recycleBin;
    private volatile boolean isInitialized = false;

    private static class GetIconUtilBuilder {
        private static final GetIconUtil INSTANCE = new GetIconUtil();
    }

    private GetIconUtil() {
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
        //添加其他常量图标  changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("")), width, height); 获取应用程序时使用
        //或者 changeIcon(new ImageIcon(GetIconUtil.class.getResource("")), width, height); 本地图标时使用
        dllImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows\\System32\\sysmain.dll")), width, height);
        folderImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("C:\\Windows")), width, height);
        txtImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user\\cmds.txt")), width, height);
        vbsImageIcon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user\\shortcutGenerator.vbs")), width, height);
        blankIcon = changeIcon(new ImageIcon(GetIconUtil.class.getResource("/icons/blank.png")), width, height);
        recycleBin = changeIcon(new ImageIcon(GetIconUtil.class.getResource("/icons/recyclebin.png")), width, height);
        updateIcon = changeIcon(new ImageIcon(GetIconUtil.class.getResource("/icons/update.png")), width, height);
        helpIcon = changeIcon(new ImageIcon(GetIconUtil.class.getResource("/icons/help.png")), width, height);
    }

    public ImageIcon getCommandIcon(String commandName, int width, int height) throws NullPointerException {
        if (!isInitialized) {
            initIconCache(width, height);
            isInitialized = true;
        }
        if (commandName == null || commandName.isEmpty()) {
            throw new NullPointerException("No icon matched with command");
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
                throw new NullPointerException("No icon matched with command");
        }
    }

    public ImageIcon getBigIcon(String path, int width, int height) {
        if (!isInitialized) {
            initIconCache(width, height);
            isInitialized = true;
        }
        if (path == null || path.isEmpty()) {
            return blankIcon;
        }
        File f = new File(path);
        if (f.exists()) {
            //已保存的常量图标
            if (path.endsWith(".dll")) {
                return dllImageIcon;
            }
            if (path.endsWith(".txt")) {
                return txtImageIcon;
            }
            if (path.endsWith(".vbs")) {
                return vbsImageIcon;
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
