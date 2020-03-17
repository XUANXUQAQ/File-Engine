package moveFiles;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;


public class moveFiles {
    private String origin;

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
        if (dir.getAbsolutePath().equals(origin) || dir.getName().equals("desktop.ini")) {
            return true;
        } else {
            return dir.delete();
        }
    }

    // ����ĳ��Ŀ¼��Ŀ¼�µ�������Ŀ¼���ļ������ļ���
    private void copyFolder(String oldPath, String newPath) {
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
            System.out.println("���������ļ������ݲ�������");
        }
    }

    public void moveFolder(String oldPath, String newPath) {
        // �ȸ����ļ�
        copyFolder(oldPath, newPath);
        origin = oldPath;
        deleteDir(new File(oldPath));
    }
}
