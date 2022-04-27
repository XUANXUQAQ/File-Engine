package file.engine.utils;

import file.engine.services.TranslateService;
import file.engine.utils.file.FileUtil;

import javax.swing.*;
import java.awt.*;
import java.io.*;

public class OpenFileUtil {

    /**
     * 以普通权限运行文件，失败则打开文件位置
     *
     * @param path 文件路径
     */
    @SuppressWarnings("IndexOfReplaceableByContains")
    public static void openWithoutAdmin(String path) {
        File file = new File(path);
        String pathLower = path.toLowerCase();
        Desktop desktop;
        if (file.exists()) {
            try {
                if (pathLower.endsWith(".url")) {
                    if (Desktop.isDesktopSupported()) {
                        desktop = Desktop.getDesktop();
                        desktop.open(new File(path));
                    }
                } else if (pathLower.endsWith(".lnk")) {
                    Runtime.getRuntime().exec("explorer.exe \"" + path + "\"");
                } else {
                    String command;
                    if (file.isFile()) {
                        command = "start " + path.substring(0, 2) + "\"" + path.substring(2) + "\"";
                        String tmpDir = new File("").getAbsolutePath().indexOf(" ") != -1 ?
                                System.getProperty("java.io.tmpdir") : new File("tmp").getAbsolutePath();
                        String vbsFilePath = generateBatAndVbsFile(command, tmpDir, FileUtil.getParentPath(path));
                        Runtime.getRuntime().exec("explorer.exe " + vbsFilePath.substring(0, 2) + "\"" + vbsFilePath.substring(2) + "\"");
                    } else {
                        Runtime.getRuntime().exec("explorer.exe \"" + path + "\"");
                    }
                }
            } catch (Exception e) {
                //打开上级文件夹
                e.printStackTrace();
                try {
                    openFolderByExplorer(path);
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
        } else {
            JOptionPane.showMessageDialog(null, TranslateService.getInstance().getTranslation("File not exist"));
        }
    }

    public static void openFolderByExplorer(String dir) throws IOException {
        Runtime.getRuntime().exec("explorer.exe /select, \"" + dir + "\"");
    }

    /**
     * 以管理员方式运行文件，失败则打开文件位置
     *
     * @param path 文件路径
     */
    public static void openWithAdmin(String path) {
        TranslateService translateService = TranslateService.getInstance();
        File file = new File(path);
        if (file.isDirectory()) {
            try {
                openFolderByExplorer(path);
            } catch (IOException e) {
                e.printStackTrace();
                JOptionPane.showMessageDialog(null, translateService.getTranslation("Execute failed"));
            }
            return;
        }
        if (file.exists()) {
            try {
                String command = file.getAbsolutePath();
                String start = "cmd.exe /c start " + command.substring(0, 2);
                String end = "\"" + command.substring(2) + "\"";
                Runtime.getRuntime().exec(start + end, null, file.getParentFile());
            } catch (IOException e) {
                //打开上级文件夹
                try {
                    OpenFileUtil.openFolderByExplorer(file.getAbsolutePath());
                } catch (IOException e1) {
                    JOptionPane.showMessageDialog(null, translateService.getTranslation("Execute failed"));
                    e.printStackTrace();
                }
            }
        } else {
            JOptionPane.showMessageDialog(null, translateService.getTranslation("File not exist"));
        }
    }

    /**
     * 在windows的temp目录(或者该软件的tmp目录，如果路径中没有空格)中生成bat以及用于隐藏bat的vbs脚本
     *
     * @param command    要运行的cmd命令
     * @param filePath   文件位置（必须传入文件夹）
     * @param workingDir 应用打开后的工作目录
     * @return vbs的路径
     */
    private static String generateBatAndVbsFile(String command, String filePath, String workingDir) {
        char disk = workingDir.charAt(0);
        String start = workingDir.substring(0, 2);
        String end = workingDir.substring(2);
        File batFilePath = new File(filePath, "openBat_File_Engine.bat");
        File vbsFilePath = new File(filePath, "openVbs_File_Engine.vbs");
        try (BufferedWriter batW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(batFilePath), System.getProperty("sun.jnu.encoding")));
             BufferedWriter vbsW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(vbsFilePath), System.getProperty("sun.jnu.encoding")))) {
            //生成bat
            batW.write(disk + ":");
            batW.newLine();
            batW.write("cd " + start + "\"" + end + "\"");
            batW.newLine();
            batW.write(command);
            //生成vbs
            vbsW.write("set ws=createobject(\"wscript.shell\")");
            vbsW.newLine();
            vbsW.write("ws.run \"" + batFilePath + "\", 0");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return vbsFilePath.getAbsolutePath();
    }
}
