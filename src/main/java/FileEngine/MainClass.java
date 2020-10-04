package FileEngine;

import FileEngine.SQLiteConfig.SQLiteUtil;
import FileEngine.checkHotkey.CheckHotKeyUtil;
import FileEngine.configs.AllConfigs;
import FileEngine.daemon.DaemonUtil;
import FileEngine.dllInterface.FileMonitor;
import FileEngine.frames.PluginMarket;
import FileEngine.frames.SearchBar;
import FileEngine.frames.SettingsFrame;
import FileEngine.frames.TaskBar;
import FileEngine.md5.Md5Util;
import FileEngine.pluginSystem.PluginUtil;
import FileEngine.search.SearchUtil;
import FileEngine.threadPool.CachedThreadPool;
import FileEngine.translate.TranslateUtil;
import com.alibaba.fastjson.JSONObject;
import com.formdev.flatlaf.FlatDarculaLaf;

import java.awt.*;
import java.io.*;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;


public class MainClass {
    private static final String FILE_MONITOR_64_MD_5 = "7258ca0167d10d6cd6963fd0c35dff84";
    private static final String FILE_SEARCHER_64_MD_5 = "c1c9de6ab35ed5091ab6c55babd54404";
    private static final String GET_ASC_II_64_MD_5 = "eff607d2dd4a7e4c878948fe8f24b3ea";
    private static final String HOTKEY_LISTENER_64_MD_5 = "41388e31d6fc22fb430f636d402cf608";
    private static final String IS_LOCAL_DISK_64_MD_5 = "f8a71d3496d8cc188713d521e6dfa2b2";
    private static final String FILE_SEARCHER_USN_64_MD_5 = "f9bb252301900a7868163419a376a8f6";
    private static final String SQLITE3_64_MD_5 = "658c71b8b93ba4eb5b4936f46a112449";
    private static final String UPDATER_BAT_64_MD_5 = "357d7cc1cf023cb6c90f73926c6f2f55";
    private static final String GET_HANDLE_64_MD_5 = "e95850346153eb73c4f62a71685ec7fb";
    private static final String DAEMON_PROCESS_64_MD_5 = "797826e9b5c4ca9ccc3fb90520e5d805";

    private static final String SHORTCUT_GENERATOR_MD_5 = "fa4e26f99f3dcd58d827828c411ea5d7";

    private static void initializeDllInterface() throws ClassNotFoundException {
        Class.forName("FileEngine.dllInterface.FileMonitor");
        Class.forName("FileEngine.dllInterface.IsLocalDisk");
        Class.forName("FileEngine.dllInterface.HotkeyListener");
        Class.forName("FileEngine.dllInterface.GetAscII");
        Class.forName("FileEngine.dllInterface.GetHandle");
    }

    private static void updatePlugins() throws FileNotFoundException {
        File sign = new File("user/updatePlugin");
        File tmpPlugins = new File("tmp/pluginsUpdate");
        if (sign.exists()) {
            if (AllConfigs.isDebug()) {
                System.out.println("正在更新插件");
            }
            sign.delete();
            File[] files = tmpPlugins.listFiles();
            if (files == null || files.length == 0) {
                return;
            }
            for (File eachPlugin : files) {
                String pluginName = eachPlugin.getName();
                File targetPlugin = new File("plugins" + File.separator + pluginName);
                copyFile(new FileInputStream(eachPlugin), targetPlugin);
            }
        }
    }

    private static void copyFile(InputStream source, File dest) {
        try (BufferedInputStream bis = new BufferedInputStream(source);
             BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(dest))) {
            byte[] buffer = new byte[8192];
            int count = bis.read(buffer);
            while (count != -1) {
                //使用缓冲流写数据
                bos.write(buffer, 0, count);
                //刷新
                bos.flush();
                count = bis.read(buffer);
            }
        } catch (IOException e) {
            e.printStackTrace();
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
        if (!file.exists()) {
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

    private static void deleteUpdater() throws InterruptedException {
        boolean ret = false;
        int count = 0;
        File updater = new File("updater.bat");
        if (updater.exists()) {
            while (!ret) {
                ret = updater.delete();
                Thread.sleep(1000);
                count++;
                if (count > 3) {
                    break;
                }
            }
        }
    }

    public static void main(String[] args) {
        try {
            Class.forName("org.sqlite.JDBC");
            FlatDarculaLaf.install();

            if (!System.getProperty("os.arch").contains("64")) {
                System.err.println("NOT 64 BIT");
                return;
            }

            SQLiteUtil.initConnection("jdbc:sqlite:data.db");

            boolean isManualUpdate = false;
            if (isDatabaseDamaged()) {
                System.out.println("无data文件，正在搜索并重建");
                //初始化数据库
                SQLiteUtil.createAllTables();
                isManualUpdate = true;
            }

            try (Statement stmt = SQLiteUtil.getStatement()) {
                stmt.executeUpdate("CREATE TABLE IF NOT EXISTS cache(PATH text unique);");
            }

            startOrIgnoreUpdateAndExit(isUpdateSignExist());
            updatePlugins();

            //清空tmp
            deleteDir(new File("tmp"));

            if (!initFoldersAndFiles()) {
                System.err.println("initialize dependencies failed");
                return;
            }

            initializeDllInterface();

            AllConfigs.readAllSettings();

            PluginUtil.getInstance().loadAllPlugins("plugins");

            TaskBar taskBar = TaskBar.getInstance();
            taskBar.showTaskBar();
            SearchUtil search = SearchUtil.getInstance();

            if (isManualUpdate) {
                search.setStatus(SearchUtil.MANUAL_UPDATE);
            }

            TranslateUtil translateUtil = TranslateUtil.getInstance();
            if (!isLatest()) {
                taskBar.showMessage(translateUtil.getTranslation("Info"), translateUtil.getTranslation("New version can be updated"));
            }

            if (PluginUtil.getInstance().isPluginTooOld()) {
                String oldPlugins = PluginUtil.getInstance().getAllOldPluginsName();
                taskBar.showMessage(translateUtil.getTranslation("Warning"), oldPlugins + "\n" + translateUtil.getTranslation("Plugin Api is too old"));
            }

            if (PluginUtil.getInstance().isPluginRepeat()) {
                String repeatPlugins = PluginUtil.getInstance().getRepeatPlugins();
                taskBar.showMessage(translateUtil.getTranslation("Warning"), repeatPlugins + "\n" + translateUtil.getTranslation("Duplicate plugin, please delete it in plugins folder"));
            }

            if (PluginUtil.getInstance().isPluginLoadError()) {
                String errorPlugins = PluginUtil.getInstance().getLoadingErrorPlugins();
                taskBar.showMessage(translateUtil.getTranslation("Warning"), errorPlugins + "\n" + translateUtil.getTranslation("Loading plugins error"));
            }

            PluginUtil.getInstance().releaseAllResources();

            if (!AllConfigs.isDebug()) {
                DaemonUtil.startDaemon(new File("").getAbsolutePath());
            }

            StringBuilder stringBuilder = new StringBuilder();
            AtomicBoolean isFinished = new AtomicBoolean(false);
            CachedThreadPool.getInstance().executeTask(() -> PluginUtil.getInstance().isAllPluginLatest(stringBuilder, isFinished));

            while (AllConfigs.isNotMainExit()) {
                // 主循环开始
                if (isFinished.get()) {
                    isFinished.set(false);
                    String notLatestPlugins = stringBuilder.toString();
                    if (!notLatestPlugins.isEmpty()) {
                        TaskBar.getInstance().showMessage(TranslateUtil.getInstance().getTranslation("Info"),
                                TranslateUtil.getInstance().getTranslation(notLatestPlugins + "\n" +
                                        TranslateUtil.getInstance().getTranslation("New versions of these plugins can be updated")));
                    }
                }
                TimeUnit.MILLISECONDS.sleep(50);
            }
            //确保关闭所有资源
            TimeUnit.SECONDS.sleep(8);
            System.exit(0);
        } catch (Exception e) {
            closeAndExit();
        }
    }

    private static void closeAndExit() {
        AllConfigs.setMainExit(true);
        SettingsFrame.getInstance().hideFrame();
        PluginMarket.getInstance().hideWindow();
        SearchBar.getInstance().closeSearchBar();
        PluginUtil.getInstance().unloadAllPlugins();
        CheckHotKeyUtil.getInstance().stopListen();
        FileMonitor.INSTANCE.stop_monitor();
        SQLiteUtil.closeConnection();
        DaemonUtil.stopDaemon();
    }

    private static void releaseAllDependence() {
        copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86-64/fileMonitor.dll", FILE_MONITOR_64_MD_5);
        copyOrIgnoreFile("user/getAscII.dll", "/win32-x86-64/getAscII.dll", GET_ASC_II_64_MD_5);
        copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86-64/hotkeyListener.dll", HOTKEY_LISTENER_64_MD_5);
        copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86-64/isLocalDisk.dll", IS_LOCAL_DISK_64_MD_5);
        copyOrIgnoreFile("user/fileSearcher.exe", "/win32-x86-64/fileSearcher.exe", FILE_SEARCHER_64_MD_5);
        copyOrIgnoreFile("user/fileSearcherUSN.exe", "/win32-x86-64/fileSearcherUSN.exe", FILE_SEARCHER_USN_64_MD_5);
        copyOrIgnoreFile("user/sqlite3.dll", "/win32-x86-64/sqlite3.dll", SQLITE3_64_MD_5);
        copyOrIgnoreFile("user/shortcutGenerator.vbs", "/shortcutGenerator.vbs", SHORTCUT_GENERATOR_MD_5);
        copyOrIgnoreFile("user/getHandle.dll", "/win32-x86-64/getHandle.dll", GET_HANDLE_64_MD_5);
        copyOrIgnoreFile("user/daemonProcess.exe", "/win32-x86-64/daemonProcess.exe", DAEMON_PROCESS_64_MD_5);
    }

    private static void copyOrIgnoreFile(String path, String rootPath, String md5) {
        File target = new File(path);
        String fileMd5 = Md5Util.getMD5(target.getAbsolutePath());
        if (!target.exists() || !md5.equals(fileMd5)) {
            if (AllConfigs.isDebug()) {
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

    private static void startOrIgnoreUpdateAndExit(boolean isUpdate) throws InterruptedException, IOException {
        if (isUpdate) {
            File update = new File("user/update");
            update.delete();
            File updaterBat = new File("updater.bat");
            copyOrIgnoreFile("updater.bat", "/updater.bat", UPDATER_BAT_64_MD_5);
            Desktop desktop;
            if (Desktop.isDesktopSupported()) {
                desktop = Desktop.getDesktop();
                desktop.open(updaterBat);
                Thread.sleep(500);
                System.exit(0);
            }
        } else {
            deleteUpdater();
        }
    }

    private static boolean initFoldersAndFiles() {
        boolean isFailed;
        //settings.json
        //isFailed = createFileOrFolder("user/settings.json", true, false);
        //user
        isFailed = createFileOrFolder("user", false, false);
        //plugins
        isFailed &= createFileOrFolder("plugins", false, false);
        //tmp
        File tmp = new File("tmp");
        String tempPath = tmp.getAbsolutePath();
        isFailed &= createFileOrFolder(tmp, false, false);
        isFailed &= createFileOrFolder(tempPath + File.separator + "fileAdded.txt", true, true);
        isFailed &= createFileOrFolder(tempPath + File.separator + "fileRemoved.txt", true, true);
        //cmd.txt
        isFailed &= createFileOrFolder("user/cmds.txt", true, false);
        releaseAllDependence();
        return isFailed;
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
            JSONObject info = SettingsFrame.getUpdateInfo();
            if (info != null) {
                String latestVersion = info.getString("version");
                if (Double.parseDouble(latestVersion) > Double.parseDouble(AllConfigs.version)) {
                    return false;
                }
            }
        } catch (IOException | InterruptedException ignored) {
        }
        return true;
    }

    private static boolean isUpdateSignExist() {
        File user = new File("user/update");
        return user.exists();
    }
}
