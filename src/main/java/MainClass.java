import DllInterface.FileMonitor;
import br.com.margel.weblaf.WebLookAndFeel;
import com.alibaba.fastjson.JSONObject;
import frames.SettingsFrame;
import frames.TaskBar;
import hotkeyListener.CheckHotKey;
import search.Search;

import javax.swing.*;
import java.io.*;
import java.math.BigInteger;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Objects;


public class MainClass {
    //32bit
    private static final String fileMonitor86Md5 = "1005aa7fa75ae86d314afcfc5df0af6b";
    private static final String fileSearcher86Md5 = "c79e7ed8c1a915bc4b79c9405d8a5828";
    private static final String getAscII86Md5 = "e370e53ce6c18758a5468fe11ccca652";
    private static final String hotkeyListener86Md5 = "15bd4db12a4939969c27c03ac9e57ddd";
    private static final String isLocalDisk86Md5 = "9b1c4c4fc44b52bff4f226b39c1ac46f";
    private static final String updater86Md5 = "b11a1307c497f00e570b238224173ba2";
    private static final String fileSearcherUSN86Md5 = "48153cfabd03e2ab907f8d361bce9130";
    private static final String isNTFS86Md5 = "2aff387756192c704c0c876f2ad12fa2";
    private static final String sqlite386Md5 = "82b03cdb95fb0ef88b876d141b478a6d";
    //64bit
    private static final String fileMonitor64Md5 = "db64b40ed1ccec6a7f2af1b40c1d22ab";
    private static final String fileSearcher64Md5 = "beaf00c5652bcf74a6ecbb2ded170bb9";
    private static final String getAscII64Md5 = "eff607d2dd4a7e4c878948fe8f24b3ea";
    private static final String hotkeyListener64Md5 = "41388e31d6fc22fb430f636d402cf608";
    private static final String isLocalDisk64Md5 = "64f64bc828f477aa9ce6f5f8fd6010f3";
    private static final String updater64Md5 = "bf8482e14b1457395f2ef1ec200f95c0";
    private static final String fileSearcherUSN64Md5 = "c800f1dab50df73794df2a94a1c847a0";
    private static final String isNTFS64Md5 = "b5f7ea2923a42873883a3bcda2bafd2";
    private static final String sqlite364Md5 = "658c71b8b93ba4eb5b4936f46a112449";

    private static final String shortcutGeneratorMd5 = "fa4e26f99f3dcd58d827828c411ea5d7";

    private static void initializeDllInterface() {
        try {
            Class.forName("DllInterface.FileMonitor");
            Class.forName("DllInterface.IsLocalDisk");
            Class.forName("DllInterface.HotkeyListener");
            Class.forName("DllInterface.GetAscII");
            Class.forName("DllInterface.isNTFS");
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

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

    private static String getFileName() {
        StackTraceElement[] stacktrace = Thread.currentThread().getStackTrace();
        StackTraceElement e = stacktrace[stacktrace.length - 1];
        return e.getFileName();
    }

    private static boolean isTableExist(ArrayList<String> tableNames) {
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:data.db"); Statement stmt = conn.createStatement()) {
            for (String tableName : tableNames) {
                String sql = "SELECT * FROM " + tableName + ";";
                stmt.execute(sql);
            }
            return true;
        } catch (SQLException e) {
            return false;
        }
    }

    private static boolean isDatabaseDamaged() {
        ArrayList<String> list = new ArrayList<>();
        for (int i = 0; i <= 40; i++) {
            list.add("list" + i);
        }
        return !isTableExist(list);
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

    private static void deleteUpdater() {
        File file = new File("updater.exe");
        file.delete();
    }

    private static String getMD5(String filePath) {
        byte[] buffer = new byte[8192];
        BigInteger bigInteger;
        int len;
        try (FileInputStream fis = new FileInputStream(filePath)) {
            MessageDigest md = MessageDigest.getInstance("MD5");
            while ((len = fis.read(buffer)) != -1) {
                md.update(buffer, 0, len);
            }
            byte[] b = md.digest();
            bigInteger = new BigInteger(1, b);
            return bigInteger.toString(16);
        } catch (NoSuchAlgorithmException | IOException e) {
            return null;
        }
    }


    public static void main(String[] args) {
        try {
            Class.forName("org.sqlite.JDBC");
            UIManager.setLookAndFeel(new WebLookAndFeel());
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        SettingsFrame.set64Bit(System.getProperty("os.arch").contains("64"));

        SettingsFrame.name = getFileName();

        File database = new File("data.db");
        boolean isManualUpdate = false;
        if (!database.exists() || isDatabaseDamaged()) {
            System.out.println("无data文件，正在搜索并重建");
            //初始化数据库
            Search.initDatabase();
            isManualUpdate = true;
        }

        if (!initSettingsJson()) {
            System.err.println("initialize failed");
            System.exit(-1);
        }

        startOrIgnoreUpdateAndExit(isUpdateSignExist());

        //清空tmp
        deleteDir(new File("tmp"));

        if (!initFoldersAndFiles()) {
            System.err.println("initialize failed");
            System.exit(-1);
        }

        initializeDllInterface();

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
            System.exit(0);
        } catch (InterruptedException ignored) {

        }
    }

    private static void releaseAllDependence(boolean is64Bit) {
        if (is64Bit) {
            copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86-64/fileMonitor.dll", fileMonitor64Md5);
            copyOrIgnoreFile("user/getAscII.dll", "/win32-x86-64/getAscII.dll", getAscII64Md5);
            copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86-64/hotkeyListener.dll", hotkeyListener64Md5);
            copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86-64/isLocalDisk.dll", isLocalDisk64Md5);
            copyOrIgnoreFile("user/fileSearcher.exe", "/win32-x86-64/fileSearcher.exe", fileSearcher64Md5);
            copyOrIgnoreFile("user/fileSearcherUSN.exe", "/win32-x86-64/fileSearcherUSN.exe", fileSearcherUSN64Md5);
            copyOrIgnoreFile("user/isNTFS.dll", "/win32-x86-64/isNTFS.dll", isNTFS64Md5);
            copyOrIgnoreFile("user/sqlite3.dll", "/win32-x86-64/sqlite3.dll", sqlite364Md5);
        } else {
            copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86/fileMonitor.dll", fileMonitor86Md5);
            copyOrIgnoreFile("user/getAscII.dll", "/win32-x86/getAscII.dll", getAscII86Md5);
            copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86/hotkeyListener.dll", hotkeyListener86Md5);
            copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86/isLocalDisk.dll", isLocalDisk86Md5);
            copyOrIgnoreFile("user/fileSearcher.exe", "/win32-x86/fileSearcher.exe", fileSearcher86Md5);
            copyOrIgnoreFile("user/fileSearcherUSN.exe", "/win32-x86/fileSearcherUSN.exe", fileSearcherUSN86Md5);
            copyOrIgnoreFile("user/isNTFS.dll", "/win32-x86/isNTFS.dll", isNTFS86Md5);
            copyOrIgnoreFile("user/sqlite3.dll", "/win32-x86/sqlite3.dll", sqlite386Md5);
        }
        copyOrIgnoreFile("user/shortcutGenerator.vbs", "/shortcutGenerator.vbs", shortcutGeneratorMd5);
    }

    private static void copyOrIgnoreFile(String path, String rootPath, String md5) {
        File target = new File(path);
        String fileMd5 = getMD5(target.getAbsolutePath());
        if (!target.exists() || !Objects.equals(fileMd5, md5)) {
            if (SettingsFrame.isDebug()) {
                System.out.println("正在重新释放文件：" + path);
            }
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
            if (SettingsFrame.is64Bit()) {
                copyOrIgnoreFile("updater.exe", "/win32-x86-64/updater.exe", updater64Md5);
            } else {
                copyOrIgnoreFile("updater.exe", "/win32-x86/updater.exe", updater86Md5);
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
            deleteUpdater();
        }
    }

    private static boolean initFoldersAndFiles() {
        boolean isFailed;
        //user
        isFailed = createFileOrFolder("user", false, false);
        //plugins
        isFailed = isFailed && createFileOrFolder("plugins", false, false);
        //tmp
        File tmp = new File("tmp");
        String tempPath = tmp.getAbsolutePath();
        isFailed = isFailed && createFileOrFolder(tmp, false, false);
        isFailed = isFailed && createFileOrFolder(tempPath + File.separator + "fileAdded.txt", true, true);
        isFailed = isFailed && createFileOrFolder(tempPath + File.separator + "fileRemoved.txt", true, true);
        //cache.dat
        isFailed = isFailed && createFileOrFolder("user/cache.dat", true, false);
        //cmd.txt
        isFailed = isFailed && createFileOrFolder("user/cmds.txt", true, false);
        releaseAllDependence(SettingsFrame.is64Bit());
        return isFailed;
    }

    private static boolean initSettingsJson() {
        return createFileOrFolder("user/settings.json", true, false);
    }

    private static boolean createFileOrFolder(File file, boolean isFile, boolean isDeleteOnExit) {
        boolean result;
        try {
            if (!file.exists()) {
                if (isFile) {
                    result = file.createNewFile();
                } else {
                    result = file.mkdirs();
                }
                if (isDeleteOnExit) {
                    file.deleteOnExit();
                }
            } else {
                result = true;
            }
        } catch (IOException e) {
            result = false;
        }
        return result;
    }

    private static boolean createFileOrFolder(String path, boolean isFile, boolean isDeleteOnExit) {
        File file = new File(path);
        return createFileOrFolder(file, isFile, isDeleteOnExit);
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
