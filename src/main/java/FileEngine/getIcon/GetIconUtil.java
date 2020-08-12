package FileEngine.getIcon;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.File;

/**
 * @author XUANXU
 */
public class GetIconUtil {
    private final FileSystemView FILE_SYSTEM_VIEW = FileSystemView.getFileSystemView();
    private static ImageIcon dllImageIcon;
    private static ImageIcon folderImageIcon;
    private static ImageIcon txtImageIcon;
    private static ImageIcon vbsImageIcon;
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
        if (path == null || path.isEmpty()) {
            return null;
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
            return null;
        }
    }
}
