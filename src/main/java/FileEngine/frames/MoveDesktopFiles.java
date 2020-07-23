package FileEngine.frames;

import FileEngine.moveFiles.MoveFilesUtil;
import FileEngine.translate.TranslateUtil;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.io.File;
import java.util.ArrayList;

/**
 * @author XUANXU
 */
public class MoveDesktopFiles implements Runnable {
    @Override
    public void run() {
        boolean desktop1;
        boolean desktop2;
        File fileDesktop = FileSystemView.getFileSystemView().getHomeDirectory();
        File fileBackUp = new File("Files");
        if (!fileBackUp.exists()) {
            fileBackUp.mkdir();
        }
        ArrayList<String> preserveFiles = new ArrayList<>();
        preserveFiles.add(fileDesktop.getAbsolutePath());
        preserveFiles.add("C:\\Users\\Public\\Desktop");
        MoveFilesUtil moveFiles = new MoveFilesUtil(preserveFiles);
        desktop1 = moveFiles.moveFolder(fileDesktop.getAbsolutePath(), fileBackUp.getAbsolutePath());
        desktop2 = moveFiles.moveFolder("C:\\Users\\Public\\Desktop", fileBackUp.getAbsolutePath());
        if (desktop1 || desktop2) {
            JOptionPane.showMessageDialog(null, TranslateUtil.getInstance().getTranslation("Files with the same name are detected, please move them by yourself"));
        }
    }
}
