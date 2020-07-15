package FileEngine;

import FileEngine.dllInterface.FileMonitor;
import FileEngine.frames.SearchBar;
import FileEngine.frames.SettingsFrame;
import FileEngine.frames.TaskBar;
import FileEngine.hotkeyListener.CheckHotKey;
import FileEngine.pluginSystem.PluginUtil;
import FileEngine.sqliteConfig.SQLiteUtil;
import FileEngine.search.SearchUtil;
import FileEngine.md5.Md5Util;
import br.com.margel.weblaf.WebLookAndFeel;
import com.alibaba.fastjson.JSONObject;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Objects;


/**
 * @author XUANXU
 */
public class MainClass {
    //32bit
    private static final String FILE_MONITOR_86_MD_5 = "1005aa7fa75ae86d314afcfc5df0af6b";
    private static final String FILE_SEARCHER_86_MD_5 = "8ff53f2f2a0d2ec0d7a89d3f4f1bb491";
    private static final String GET_ASC_II_86_MD_5 = "e370e53ce6c18758a5468fe11ccca652";
    private static final String HOTKEY_LISTENER_86_MD_5 = "15bd4db12a4939969c27c03ac9e57ddd";
    private static final String IS_LOCAL_DISK_86_MD_5 = "9b1c4c4fc44b52bff4f226b39c1ac46f";
    private static final String FILE_SEARCHER_USN_86_MD_5 = "48153cfabd03e2ab907f8d361bce9130";
    private static final String IS_NTFS_86_MD_5 = "2aff387756192c704c0c876f2ad12fa2";
    private static final String SQLITE3_86_MD_5 = "82b03cdb95fb0ef88b876d141b478a6d";
    private static final String UPDATER_BAT_86_MD_5 = "005b57b56c3402b2038f90abc6e141e7";
    //64bit
    private static final String FILE_MONITOR_64_MD_5 = "db64b40ed1ccec6a7f2af1b40c1d22ab";
    private static final String FILE_SEARCHER_64_MD_5 = "41645c85b1b14823cd1fcf9cce069477";
    private static final String GET_ASC_II_64_MD_5 = "eff607d2dd4a7e4c878948fe8f24b3ea";
    private static final String HOTKEY_LISTENER_64_MD_5 = "41388e31d6fc22fb430f636d402cf608";
    private static final String IS_LOCAL_DISK_64_MD_5 = "64f64bc828f477aa9ce6f5f8fd6010f3";
    private static final String FILE_SEARCHER_USN_64_MD_5 = "c800f1dab50df73794df2a94a1c847a0";
    private static final String IS_NTFS_64_MD_5 = "b5f7ea2923a42873883a3bcda2bafd2";
    private static final String SQLITE3_64_MD_5 = "658c71b8b93ba4eb5b4936f46a112449";
    private static final String UPDATER_BAT_64_MD_5 = "357d7cc1cf023cb6c90f73926c6f2f55";

    private static final String SHORTCUT_GENERATOR_MD_5 = "fa4e26f99f3dcd58d827828c411ea5d7";

    private static void initializeDllInterface() {
        try {
            Class.forName("FileEngine.dllInterface.FileMonitor");
            Class.forName("FileEngine.dllInterface.IsLocalDisk");
            Class.forName("FileEngine.dllInterface.HotkeyListener");
            Class.forName("FileEngine.dllInterface.GetAscII");
            Class.forName("FileEngine.dllInterface.isNTFS");
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static void updatePlugins() {
        File sign = new File("user/updatePlugin");
        File tmpPlugins = new File("tmp/pluginsUpdate");
        if (sign.exists()) {
            sign.delete();
            File[] files = tmpPlugins.listFiles();
            if (files == null || files.length == 0) {
                return;
            }
            for (File eachPlugin : files) {
                String pluginName = eachPlugin.getName();
                File targetPlugin = new File("plugins" + File.separator + pluginName);
                try {
                    copyFile(new FileInputStream(eachPlugin), targetPlugin);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }
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

    private static boolean isTableExist(ArrayList<String> tableNames) {
        try (Statement stmt = SQLiteUtil.getStatement()) {
            for (String tableName : tableNames) {
                String sql = "SELECT * FROM " + tableName + ";";
                stmt.execute(sql);
            }
            return true;
        } catch (Exception e) {
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
        boolean ret = false;
        int count = 0;
        File updater = new File("updater.bat");
        try {
            while (!ret) {
                ret = updater.delete();
                Thread.sleep(1000);
                count++;
                if (count > 3) {
                    break;
                }
            }
        } catch (InterruptedException ignored) {
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
        //SettingsFrame.set64Bit(false);

        try {
            SQLiteUtil.initConnection("jdbc:sqlite:data.db");
        } catch (SQLException e) {
            System.err.println("initialize database connection failed");
            e.printStackTrace();
            System.exit(-1);
        }

        boolean isManualUpdate = false;
        if (isDatabaseDamaged()) {
            System.out.println("无data文件，正在搜索并重建");
            //初始化数据库
            try {
                SQLiteUtil.createAllTables();
            } catch (Exception e) {
                System.err.println("initialize database connection failed");
                e.printStackTrace();
                System.exit(-1);
            }
            isManualUpdate = true;
        }

        if (!initSettingsJson()) {
            System.err.println("initialize settings failed");
            System.exit(-1);
        }

        startOrIgnoreUpdateAndExit(isUpdateSignExist());
        updatePlugins();

        //清空tmp
        deleteDir(new File("tmp"));

        if (!initFoldersAndFiles()) {
            System.err.println("initialize dependencies failed");
            System.exit(-1);
        }


        initializeDllInterface();

        try {
            PluginUtil.loadAllPlugins("plugins");
        } catch (Exception e) {
            e.printStackTrace();
        }

        TaskBar taskBar = TaskBar.getInstance();
        taskBar.showTaskBar();
        SearchUtil search = SearchUtil.getInstance();

        if (isManualUpdate) {
            search.setManualUpdate(true);
        }

        if (!isLatest()) {
            taskBar.showMessage(SettingsFrame.getTranslation("Info"), SettingsFrame.getTranslation("New version can be updated"));
        }

        if (PluginUtil.isPluginTooOld()) {
            String oldPlugins = PluginUtil.getAllOldPluginsName();
            taskBar.showMessage(SettingsFrame.getTranslation("Warning"), oldPlugins + "\n" + SettingsFrame.getTranslation("Plugin Api is too old"));
        }

        if (PluginUtil.isPluginRepeat()) {
            String repeatPlugins = PluginUtil.getRepeatPlugins();
            taskBar.showMessage(SettingsFrame.getTranslation("Warning"), repeatPlugins + "\n" + SettingsFrame.getTranslation("Duplicate plugin, please delete it in plugins folder"));
        }

        try {
            while (SettingsFrame.isNotMainExit()) {
                // 主循环开始
                Thread.sleep(100);
            }
            PluginUtil.unloadAllPlugins();
            SearchBar.getInstance().releaseAllSqlCache();
            CheckHotKey.getInstance().stopListen();
            FileMonitor.INSTANCE.stop_monitor();
            Thread.sleep(8000);
            System.exit(0);
        } catch (InterruptedException ignored) {

        }
    }

    private static void releaseAllDependence(boolean is64Bit) {
        if (is64Bit) {
            copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86-64/fileMonitor.dll", FILE_MONITOR_64_MD_5);
            copyOrIgnoreFile("user/getAscII.dll", "/win32-x86-64/getAscII.dll", GET_ASC_II_64_MD_5);
            copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86-64/hotkeyListener.dll", HOTKEY_LISTENER_64_MD_5);
            copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86-64/isLocalDisk.dll", IS_LOCAL_DISK_64_MD_5);
            copyOrIgnoreFile("user/fileSearcher.exe", "/win32-x86-64/fileSearcher.exe", FILE_SEARCHER_64_MD_5);
            copyOrIgnoreFile("user/fileSearcherUSN.exe", "/win32-x86-64/fileSearcherUSN.exe", FILE_SEARCHER_USN_64_MD_5);
            copyOrIgnoreFile("user/isNTFS.dll", "/win32-x86-64/isNTFS.dll", IS_NTFS_64_MD_5);
            copyOrIgnoreFile("user/sqlite3.dll", "/win32-x86-64/sqlite3.dll", SQLITE3_64_MD_5);
        } else {
            copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86/fileMonitor.dll", FILE_MONITOR_86_MD_5);
            copyOrIgnoreFile("user/getAscII.dll", "/win32-x86/getAscII.dll", GET_ASC_II_86_MD_5);
            copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86/hotkeyListener.dll", HOTKEY_LISTENER_86_MD_5);
            copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86/isLocalDisk.dll", IS_LOCAL_DISK_86_MD_5);
            copyOrIgnoreFile("user/fileSearcher.exe", "/win32-x86/fileSearcher.exe", FILE_SEARCHER_86_MD_5);
            copyOrIgnoreFile("user/fileSearcherUSN.exe", "/win32-x86/fileSearcherUSN.exe", FILE_SEARCHER_USN_86_MD_5);
            copyOrIgnoreFile("user/isNTFS.dll", "/win32-x86/isNTFS.dll", IS_NTFS_86_MD_5);
            copyOrIgnoreFile("user/sqlite3.dll", "/win32-x86/sqlite3.dll", SQLITE3_86_MD_5);
        }
        copyOrIgnoreFile("user/shortcutGenerator.vbs", "/shortcutGenerator.vbs", SHORTCUT_GENERATOR_MD_5);
    }

    private static void copyOrIgnoreFile(String path, String rootPath, String md5) {
        File target = new File(path);
        String fileMd5 = Md5Util.getMD5(target.getAbsolutePath());
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
            File update = new File("user/update");
            update.delete();
            File updaterBat = new File("updater.bat");
            if (SettingsFrame.is64Bit()) {
                copyOrIgnoreFile("updater.bat", "/win32-x86-64/updater.bat", UPDATER_BAT_64_MD_5);
            } else {
                copyOrIgnoreFile("updater.bat", "/win32-x86/updater.bat", UPDATER_BAT_86_MD_5);
            }
            Desktop desktop;
            if (Desktop.isDesktopSupported()) {
                desktop = Desktop.getDesktop();
                try {
                    desktop.open(updaterBat);
                    Thread.sleep(100);
                    System.exit(0);
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
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
        File user = new File("user/update");
        return user.exists();
    }
}
