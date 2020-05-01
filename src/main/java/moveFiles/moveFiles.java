package moveFiles;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;


public class moveFiles {
    private ArrayList<String> preserveFiles;

    public moveFiles(ArrayList<String> _preserveFiles) {
        this.preserveFiles = _preserveFiles;
    }

    private boolean deleteDir(File dir) {
        // ������ļ���
        if (dir.isDirectory()) {
            // ��������ļ����µĵ������ļ�
            String[] children = dir.list();
            // �ݹ�ɾ��Ŀ¼�е���Ŀ¼��
            assert children != null;
            for (String child : children) {
                boolean isDelete = deleteDir(new File(dir, child));
                if (!isDelete) {
                    return false;
                }
            }
        }
        if (preserveFiles.contains(dir.getAbsolutePath()) || dir.getName().equals("desktop.ini")) {
            return true;
        } else {
            return dir.delete();
        }
    }

    // ����ĳ��Ŀ¼��Ŀ¼�µ�������Ŀ¼���ļ������ļ���
    private boolean copyFolder(String oldPath, String newPath) {
        boolean isHasRepeatFiles = false;
        try {
            (new File(newPath)).mkdirs();
            // ��ȡ�����ļ��е����ݵ�file�ַ������飬��������һ���α�i����ͣ�������ƿ�ʼ���������
            File filelist = new File(oldPath);
            String[] file = filelist.list();
            File temp;
            assert file != null;
            for (String s : file) {
                if (!s.endsWith("desktop.ini")) {
                    // ���oldPath��·���ָ���/����\��β����ô��oldPath/�ļ����Ϳ�����
                    if (oldPath.endsWith(File.separator)) {
                        temp = new File(oldPath + s);
                    } else {
                        temp = new File(oldPath + File.separator + s);
                    }

                    if (temp.isFile()) {
                        if (!isFileExist(newPath + "/" + (temp.getName()))) {
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
            System.out.println("���������ļ������ݲ�������");
        }
        return isHasRepeatFiles;
    }

    public boolean moveFolder(String oldPath, String newPath) {
        // �ȸ����ļ�
        boolean isHasRepeated = copyFolder(oldPath, newPath);
        deleteDir(new File(oldPath));
        return isHasRepeated;
    }

    private boolean isFileExist(String path) {
        return new File(path).exists();
    }
}
