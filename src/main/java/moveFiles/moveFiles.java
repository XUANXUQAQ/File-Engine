package moveFiles;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;


public class moveFiles {
    private String origin;

    private boolean deleteDir(File dir) {
        // 如果是文件夹
        if (dir.isDirectory()) {
            // 则读出该文件夹下的的所有文件
            String[] children = dir.list();
            // 递归删除目录中的子目录下
            assert children != null;
            for (String child : children) {
                boolean isDelete = deleteDir(new File(dir, child));
                if (!isDelete) {
                    return false;
                }
            }
        }
        if (dir.getAbsolutePath().equals(origin) || dir.getName().equals("desktop.ini")) {
            return true;
        } else {
            return dir.delete();
        }
    }

    // 复制某个目录及目录下的所有子目录和文件到新文件夹
    private void copyFolder(String oldPath, String newPath) {
        try {
            (new File(newPath)).mkdirs();
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
                        FileInputStream input = new FileInputStream(temp);
                        FileOutputStream output = new FileOutputStream(newPath
                                + "/" + (temp.getName()));
                        byte[] bufferarray = new byte[1024 * 64];
                        int prereadlength;
                        while ((prereadlength = input.read(bufferarray)) != -1) {
                            output.write(bufferarray, 0, prereadlength);
                        }
                        output.flush();
                        output.close();
                        input.close();
                    }
                    if (temp.isDirectory()) {
                        copyFolder(oldPath + "/" + s, newPath + "/" + s);
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("复制整个文件夹内容操作出错");
        }
    }

    public void moveFolder(String oldPath, String newPath) {
        // 先复制文件
        copyFolder(oldPath, newPath);
        origin = oldPath;
        deleteDir(new File(oldPath));
    }
}
