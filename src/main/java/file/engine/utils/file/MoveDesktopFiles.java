package file.engine.utils.file;

import javax.swing.filechooser.FileSystemView;
import java.io.File;
import java.util.ArrayList;

public class MoveDesktopFiles {
    public static boolean start() {
        boolean desktop1HasConflictFile;
        boolean desktop2HasConflictFile;
        boolean mkdirRet = true;
        File fileDesktop = FileSystemView.getFileSystemView().getHomeDirectory();
        File fileBackUp = new File("Files");
        if (!fileBackUp.exists()) {
            mkdirRet = fileBackUp.mkdir();
        }
        if (mkdirRet) {
            ArrayList<String> preserveFiles = new ArrayList<>();
            preserveFiles.add(fileDesktop.getAbsolutePath());
            preserveFiles.add("C:\\Users\\Public\\Desktop");
            MoveFilesUtil moveFiles = new MoveFilesUtil(preserveFiles);
            desktop1HasConflictFile = moveFiles.moveFolder(fileDesktop.getAbsolutePath(), fileBackUp.getAbsolutePath());
            desktop2HasConflictFile = moveFiles.moveFolder("C:\\Users\\Public\\Desktop", fileBackUp.getAbsolutePath());
            return !desktop1HasConflictFile && !desktop2HasConflictFile;
        } else {
            System.err.println("Error mkdir \"Files\"");
            return false;
        }
    }
}
