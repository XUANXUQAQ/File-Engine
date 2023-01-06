package file.engine.utils.file;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.io.*;
import java.util.ArrayList;

public class MoveDesktopFilesUtil {
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

    private record MoveFilesUtil(ArrayList<String> preserveFiles) {

        /**
         * 清空文件夹
         *
         * @param dir 文件夹路径
         * @return true如果删除完成 否则false
         */
        private boolean deleteDir(File dir) {
            // 如果是文件夹
            if (dir.isDirectory()) {
                // 则读出该文件夹下的的所有文件
                String[] children = dir.list();
                // 递归删除目录中的子目录下
                if (!(children == null || children.length == 0)) {
                    for (String child : children) {
                        boolean isDelete = deleteDir(new File(dir, child));
                        if (!isDelete) {
                            return false;
                        }
                    }
                }
            }
            if (preserveFiles.contains(dir.getAbsolutePath()) || "desktop.ini".equals(dir.getName())) {
                return true;
            } else {
                return dir.delete();
            }
        }

        // 复制某个目录及目录下的所有子目录和文件到新文件夹
        private boolean copyFolder(String oldPath, String newPath) {
            boolean isHasRepeatFiles = false;
            try {
                boolean isDirCreated;
                File newPathFile = new File(newPath);
                if (newPathFile.exists()) {
                    isDirCreated = true;
                } else {
                    isDirCreated = (new File(newPath)).mkdirs();
                }
                if (!isDirCreated) {
                    return false;
                }
                // 读取整个文件夹的内容到file字符串数组，下面设置一个游标i，不停地向下移开始读这个数组
                File filelist = new File(oldPath);
                String[] file = filelist.list();
                File temp;
                assert file != null;
                for (String s : file) {
                    if (!s.endsWith("desktop.ini")) {
                        // 如果oldPath以路径分隔符/或者\结尾，那么则oldPath/文件名就可以了
                        if (oldPath.endsWith(File.separator)) {
                            temp = new File(oldPath + s);
                        } else {
                            temp = new File(oldPath + File.separator + s);
                        }

                        if (temp.isFile()) {
                            if (!isFileExist(newPath + "/" + (temp.getName()))) {
                                BufferedInputStream input = new BufferedInputStream(new FileInputStream(temp));
                                BufferedOutputStream output = new BufferedOutputStream(new FileOutputStream(newPath
                                        + "/" + (temp.getName())));
                                byte[] bufferarray = new byte[1024 * 64];
                                int prereadlength;
                                while ((prereadlength = input.read(bufferarray)) != -1) {
                                    output.write(bufferarray, 0, prereadlength);
                                }
                                output.flush();
                                output.close();
                                input.close();
                            } else {
                                preserveFiles.add(temp.getAbsolutePath());
                                isHasRepeatFiles = true;
                            }
                        }
                        if (temp.isDirectory()) {
                            copyFolder(oldPath + "/" + s, newPath + "/" + s);
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            return isHasRepeatFiles;
        }

        /**
         * 移动一个文件夹中的所有文件到另一个文件夹
         *
         * @param oldPath 需要移动的文件夹
         * @param newPath 移动到的文件夹
         * @return 是否含有冲突文件
         */
        private boolean moveFolder(String oldPath, String newPath) {
            // 先复制文件
            boolean isDirCreated;
            File newPathFile = new File(newPath);
            if (newPathFile.exists()) {
                isDirCreated = true;
            } else {
                isDirCreated = (new File(newPath)).mkdirs();
            }
            if (!isDirCreated) {
                JOptionPane.showMessageDialog(null, "Create " + newPath + " dir failed", "ERROR", JOptionPane.ERROR_MESSAGE);
                return false;
            }
            boolean isHasRepeated = copyFolder(oldPath, newPath);
            deleteDir(new File(oldPath));
            return isHasRepeated;
        }

        private boolean isFileExist(String path) {
            return new File(path).exists();
        }
    }
}
