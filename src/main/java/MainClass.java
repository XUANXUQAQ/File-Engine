import DllInterface.FileMonitor;
import com.alibaba.fastjson.JSONObject;
import frames.SearchBar;
import frames.SettingsFrame;
import frames.TaskBar;
import hotkeyListener.CheckHotKey;
import mdlaf.MaterialLookAndFeel;
import search.Search;

import javax.swing.*;
import java.io.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;


public class MainClass {

    private static void copyFile(InputStream source, File dest) {
        try (OutputStream os = new FileOutputStream(dest); BufferedInputStream bis = new BufferedInputStream(source); BufferedOutputStream bos = new BufferedOutputStream(os)) {
            byte[] buffer = new byte[8192];
            int count = bis.read(buffer);
            while (count != -1) {
                //使用缓冲流写数据
                bos.write(buffer, 0, count);
                //刷新
                bos.flush();
                count = bis.read(buffer);
            }
        } catch (IOException ignored) {

        }
    }

    private static void deleteDir(File file) {
        if (!file.exists()) {//判断是否待删除目录是否存在
            return;
        }

        String[] content = file.list();//取得当前目录下所有文件和文件夹
        if (content != null) {
            for (String name : content) {
                File temp = new File(file.getAbsolutePath(), name);
                if (temp.isDirectory()) {//判断是否是目录
                    deleteDir(temp.getAbsolutePath());//递归调用，删除目录里的内容
                    temp.delete();//删除空目录
                } else {
                    if (!temp.delete()) {//直接删除文件
                        System.out.println("Failed to delete " + name);
                    }
                }
            }
        }
    }

    private static void deleteDir(String path) {
        File file = new File(path);
        deleteDir(file);
    }

    private static void deleteFile(String path) {
        File file = new File(path);
        file.delete();
    }


    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(new MaterialLookAndFeel(new MaterialDesign.materialLookAndFeel()));
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
        String osArch = System.getProperty("os.arch");
        if (osArch.contains("64")) {
            SettingsFrame.name = "File-Engine-x64.exe";
        } else {
            SettingsFrame.name = "File-Engine-x86.exe";
        }

        File database = new File("data.db");
        boolean isManualUpdate = false;
        if (!database.exists()) {
            System.out.println("无data文件，正在搜索并重建");
            //初始化数据库
            initDatabase();
            isManualUpdate = true;
        }

        if (!initSettingsJson()) {
            System.err.println("initialize failed");
            System.exit(-1);
        }

        SettingsFrame.readAllSettings();

        startOrIgnoreUpdateAndExit(isUpdateSignExist());

        //清空tmp
        deleteDir(new File("tmp"));

        if (!initFoldersAndFiles()) {
            System.err.println("initialize failed");
            System.exit(-1);
        }

        SettingsFrame.getInstance();

        SearchBar searchBar = SearchBar.getInstance();
        Search search = Search.getInstance();
        TaskBar taskBar = TaskBar.getInstance();
        taskBar.showTaskBar();

        if (isManualUpdate) {
            search.setManualUpdate(true);
        }

        if (!isLatest()) {
            taskBar.showMessage(SettingsFrame.getTranslation("Info"), SettingsFrame.getTranslation("New version can be updated"));
        }


        try {
            while (SettingsFrame.isNotMainExit()) {
                // 主循环开始
                Thread.sleep(100);
            }
            CheckHotKey.getInstance().stopListen();
            FileMonitor.INSTANCE.stop_monitor();
            Thread.sleep(8000);
            deleteFile(SettingsFrame.getTmp().getAbsolutePath() + File.separator + "fileAdded.txt");
            deleteFile(SettingsFrame.getTmp().getAbsolutePath() + File.separator + "fileRemoved.txt");
            System.exit(0);
        } catch (InterruptedException ignored) {

        }
    }

    private static void initDatabase() {
        Connection conn = null;
        Statement stmt;
        String sql = "CREATE TABLE list";
        try {
            Class.forName("org.sqlite.JDBC");
            conn = DriverManager.getConnection("jdbc:sqlite:data.db");
            System.out.println("open database successfully");
            stmt = conn.createStatement();
            stmt.execute("BEGIN;");
            for (int i = 0; i < 26; i++) {
                String command = sql + i + " " + "(PATH text unique)" + ";";
                stmt.executeUpdate(command);
            }
            stmt.execute("COMMIT");
        } catch (Exception e) {
            System.err.println("initialize database error");
        } finally {
            try {
                if (conn != null) {
                    conn.close();
                }
            } catch (SQLException ignored) {

            }
        }
    }

    private static void releaseAllDependence(boolean is64Bit) {
        if (is64Bit) {
            copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86-64/fileMonitor.dll");
            copyOrIgnoreFile("user/getAscII.dll", "/win32-x86-64/getAscII.dll");
            copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86-64/hotkeyListener.dll");
            copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86-64/isLocalDisk.dll");
            copyOrIgnoreFile("user/fileSearcher.exe", "/win32-x86-64/fileSearcher.exe");
            copyOrIgnoreFile("user/restart.exe", "/win32-x86-64/restart.exe");
        } else {
            copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86/fileMonitor.dll");
            copyOrIgnoreFile("user/getAscII.dll", "/win32-x86/getAscII.dll");
            copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86/hotkeyListener.dll");
            copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86/isLocalDisk.dll");
            copyOrIgnoreFile("user/fileSearcher.exe", "/win32-x86/fileSearcher.exe");
            copyOrIgnoreFile("user/restart.exe", "/win32-x86/restart.exe");
        }
        copyOrIgnoreFile("user/shortcutGenerator.vbs", "/shortcutGenerator.vbs");
        copyOrIgnoreFile("user/sqlite3.dll", "/sqlite3.dll");
    }

    private static void copyOrIgnoreFile(String path, String rootPath) {
        File target;
        target = new File(path);
        if (!target.exists()) {
            InputStream resource = MainClass.class.getResourceAsStream(rootPath);
            copyFile(resource, target);
            try {
                resource.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static void startOrIgnoreUpdateAndExit(boolean isUpdate) {
        //复制updater.exe
        if (isUpdate) {
            if (SettingsFrame.name.contains("x64")) {
                copyOrIgnoreFile("updater.exe", "/win32-x86-64/updater.exe");
            } else {
                copyOrIgnoreFile("updater.exe", "/win32-x86/updater.exe");
            }
            File updaterExe = new File("updater.exe");
            String absPath = updaterExe.getAbsolutePath();
            String path = absPath.substring(0, 2) + "\"" + absPath.substring(2) + "\"";
            String command = "cmd.exe /c " + path + " " + "\"" + SettingsFrame.name + "\"";
            try {
                Runtime.getRuntime().exec(command);
                System.exit(0);
            } catch (Exception ignored) {

            }
        } else {
            deleteFile("updater.exe");
        }
    }

    private static boolean initFoldersAndFiles() {
        boolean isFailed;
        //user
        isFailed = createFileOrFolder("user", false);
        //tmp
        File tmp = new File("tmp");
        String tempPath = tmp.getAbsolutePath();
        isFailed = isFailed && createFileOrFolder(tmp, false);
        isFailed = isFailed && createFileOrFolder(tempPath + File.separator + "fileAdded.txt", true);
        isFailed = isFailed && createFileOrFolder(tempPath + File.separator + "fileRemoved.txt", true);
        //cache.dat
        isFailed = isFailed && createFileOrFolder("user/cache.dat", true);
        //cmd.txt
        isFailed = isFailed && createFileOrFolder("user/cmds.txt", true);
        releaseAllDependence(SettingsFrame.name.contains("x64"));
        return isFailed;
    }

    private static boolean initSettingsJson() {
        return createFileOrFolder("user/settings.json", true);
    }

    private static boolean createFileOrFolder(File file, boolean isFile) {
        boolean result;
        try {
            if (!file.exists()) {
                if (isFile) {
                    result = file.createNewFile();
                } else {
                    result = file.mkdirs();
                }
            } else {
                result = true;
            }
        } catch (IOException e) {
            result = false;
        }
        return result;
    }

    private static boolean createFileOrFolder(String path, boolean isFile) {
        File file = new File(path);
        return createFileOrFolder(file, isFile);
    }

    private static boolean isLatest() {
        //检测是否为最新版本
        try {
            JSONObject info = SettingsFrame.getInfo();
            String latestVersion = info.getString("version");
            if (Double.parseDouble(latestVersion) > Double.parseDouble(SettingsFrame.version)) {
                return false;
            }
        } catch (IOException ignored) {

        }
        return true;
    }

    private static boolean isUpdateSignExist() {
        File user = new File("user");
        File[] userFiles = user.listFiles();
        boolean isUpdate = false;
        if (userFiles != null) {
            for (File each : userFiles) {
                if (each.getName().equals("update")) {
                    isUpdate = true;
                    each.delete();
                    break;
                }
            }
        }
        return isUpdate;
    }
}
