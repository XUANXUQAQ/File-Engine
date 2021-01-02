package FileEngine;

import FileEngine.eventHandler.impl.stop.RestartEvent;
import FileEngine.utils.classScan.ClassScannerUtil;
import FileEngine.configs.AllConfigs;
import FileEngine.configs.Enums;
import FileEngine.utils.database.DatabaseUtil;
import FileEngine.utils.database.SQLiteUtil;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventUtil;
import FileEngine.eventHandler.impl.ReadConfigsAndBootSystemEvent;
import FileEngine.eventHandler.impl.SetDefaultSwingLaf;
import FileEngine.eventHandler.impl.configs.SetConfigsEvent;
import FileEngine.eventHandler.impl.database.UpdateDatabaseEvent;
import FileEngine.eventHandler.impl.plugin.ReleasePluginResourcesEvent;
import FileEngine.eventHandler.impl.stop.CloseEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarMessageEvent;
import FileEngine.frames.SettingsFrame;
import FileEngine.utils.Md5Util;
import FileEngine.utils.moveFiles.CopyFileUtil;
import FileEngine.utils.pluginSystem.PluginUtil;
import FileEngine.utils.CachedThreadPoolUtil;
import FileEngine.utils.TranslateUtil;
import com.alibaba.fastjson.JSONObject;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;


public class MainClass {
    private static final String FILE_MONITOR_64_MD_5 = "5a8c123397c8e89614d4f9b91c2fa8f9";
    private static final String FILE_SEARCHER_64_MD_5 = "fa6a144d3f7bf6363abd143ce777f417";
    private static final String GET_ASC_II_64_MD_5 = "eff607d2dd4a7e4c878948fe8f24b3ea";
    private static final String HOTKEY_LISTENER_64_MD_5 = "dca474d8385fd9bbd6a3ea3e7375bba0";
    private static final String IS_LOCAL_DISK_64_MD_5 = "f8a71d3496d8cc188713d521e6dfa2b2";
    private static final String FILE_SEARCHER_USN_64_MD_5 = "4bda825986f333e8efbb11e5d3a77158";
    private static final String SQLITE3_64_MD_5 = "658c71b8b93ba4eb5b4936f46a112449";
    private static final String UPDATER_BAT_64_MD_5 = "357d7cc1cf023cb6c90f73926c6f2f55";
    private static final String GET_HANDLE_64_MD_5 = "df48132c425c20ba65821bb73657a3fc";
    private static final String DAEMON_PROCESS_64_MD_5 = "9a0529b3f8ee3961e0f9ed5f562cd0e1";
    private static final String SHORTCUT_GEN_MD_5="d2d3215c2a0741370851f2d4ed738b54";

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
            if (IsDebug.isDebug()) {
                System.out.println("正在更新插件");
            }
            boolean isUpdatePluginSignDeleted = sign.delete();
            if (!isUpdatePluginSignDeleted) {
                System.err.println("删除插件更新标志失败");
            }
            File[] files = tmpPlugins.listFiles();
            if (files == null || files.length == 0) {
                return;
            }
            for (File eachPlugin : files) {
                String pluginName = eachPlugin.getName();
                File targetPlugin = new File("plugins" + File.separator + pluginName);
                CopyFileUtil.copyFile(new FileInputStream(eachPlugin), targetPlugin);
            }
        }
    }

    private static boolean isTableExist(ArrayList<String> tableNames) {
        try (Statement stmt = SQLiteUtil.getStatement()) {
            for (String tableName : tableNames) {
                String sql= String.format("SELECT ASCII, PATH FROM %s;", tableName);
                stmt.executeQuery(sql);
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private static boolean isAtDiskC() {
        return new File("").getAbsolutePath().startsWith("C:");
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
                    //删除空目录
                    if (temp.delete()) {
                        System.err.println("Failed to delete " + name);
                    }
                } else {
                    if (!temp.delete()) {//直接删除文件
                        System.err.println("Failed to delete " + name);
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

    private static void createCacheTable() throws SQLException {
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement("CREATE TABLE IF NOT EXISTS cache(PATH text unique);")) {
            pStmt.executeUpdate();
        }
    }

    private static void checkVersion() {
        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (!isLatest()) {
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Info"), translateUtil.getTranslation("New version can be updated")));
        }
    }

    private static void checkOldPlugin() {
        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (PluginUtil.getInstance().isPluginTooOld()) {
            String oldPlugins = PluginUtil.getInstance().getAllOldPluginsName();
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Warning"), oldPlugins + "\n" + translateUtil.getTranslation("Plugin Api is too old")));
        }
    }

    private static void checkRepeatPlugin() {
        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (PluginUtil.getInstance().isPluginRepeat()) {
            String repeatPlugins = PluginUtil.getInstance().getRepeatPlugins();
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Warning"), repeatPlugins + "\n" + translateUtil.getTranslation("Duplicate plugin, please delete it in plugins folder")));
        }
    }

    private static void checkErrorPlugin() {
        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (PluginUtil.getInstance().isPluginLoadError()) {
            String errorPlugins = PluginUtil.getInstance().getLoadingErrorPlugins();
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Warning"), errorPlugins + "\n" + translateUtil.getTranslation("Loading plugins error")));
        }
    }

    private static void initDatabase() throws SQLException {
        SQLiteUtil.initConnection("jdbc:sqlite:data.db");
        createCacheTable();
    }

    private static void checkPluginInfo() {
        checkVersion();
        checkOldPlugin();
        checkRepeatPlugin();
        checkErrorPlugin();
    }

    private static void checkRunningDirAtDiskC() {
        EventUtil eventUtil = EventUtil.getInstance();
        if (isAtDiskC()) {
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    TranslateUtil.getInstance().getTranslation("Warning"),
                    TranslateUtil.getInstance().getTranslation("Putting the software on the C drive may cause index failure issue")));
        }
    }

    public static void main(String[] args) {
        try {
            Class.forName("org.sqlite.JDBC");

            if (!System.getProperty("os.arch").contains("64")) {
                JOptionPane.showMessageDialog(null, "Not 64 Bit", "ERROR", JOptionPane.ERROR_MESSAGE);
                return;
            }

            initDatabase();
            startOrIgnoreUpdateAndExit(isUpdateSignExist());
            updatePlugins();

            //清空tmp
            deleteDir(new File("tmp"));

            if (!initFoldersAndFiles()) {
                System.err.println("initialize dependencies failed");
                return;
            }

            initializeDllInterface();

            ClassScannerUtil.executeStaticMethodByName("registerEventHandler");

            EventUtil eventUtil = EventUtil.getInstance();

            if (sendStartSignalFailed()) {
                System.err.println("初始化失败");
                System.exit(0);
            }

            eventUtil.putEvent(new SetDefaultSwingLaf());
            eventUtil.putEvent(new SetConfigsEvent());

            checkPluginInfo();

            eventUtil.putEvent(new ReleasePluginResourcesEvent());

            if (isDatabaseDamaged()) {
                eventUtil.putEvent(new ShowTaskBarMessageEvent(
                        TranslateUtil.getInstance().getTranslation("Info"),
                        TranslateUtil.getInstance().getTranslation("Updating file index")));
                eventUtil.putEvent(new UpdateDatabaseEvent());
            } else {
                checkIndex();
            }

            checkRunningDirAtDiskC();

            mainLoop();

            //确保关闭所有资源
            TimeUnit.SECONDS.sleep(5);
            System.exit(0);
        } catch (Exception e) {
            e.printStackTrace();
            closeAndExit();
        }
    }

    private static void mainLoop() throws InterruptedException {
        Date startTime = new Date();
        Date endTime;
        int checkTimeCount = 0;
        long timeDiff;
        long div = 24 * 60 * 60 * 1000;
        int restartCount = 0;

        StringBuilder notLatestPluginsBuilder = new StringBuilder();
        AtomicBoolean isFinished = new AtomicBoolean(false);
        CachedThreadPoolUtil.getInstance().executeTask(() -> PluginUtil.getInstance().isAllPluginLatest(notLatestPluginsBuilder, isFinished));

        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        while (eventUtil.isNotMainExit()) {
            // 主循环开始
            if (isFinished.get()) {
                isFinished.set(false);
                String notLatestPlugins = notLatestPluginsBuilder.toString();
                if (!notLatestPlugins.isEmpty()) {
                    eventUtil.putEvent(new ShowTaskBarMessageEvent(
                            translateUtil.getTranslation("Info"),
                            notLatestPlugins + "\n" +
                            translateUtil.getTranslation("New versions of these plugins can be updated")));
                }
            }
            //检查已工作时间
            checkTimeCount++;
            if (checkTimeCount > 2000) {
                //100s检查一次时间
                checkTimeCount = 0;
                endTime = new Date();
                timeDiff = endTime.getTime() - startTime.getTime();
                long diffDays = timeDiff / div;
                if (diffDays > 2) {
                    restartCount++;
                    startTime = endTime;
                    //启动时间已经超过2天,更新索引
                    eventUtil.putEvent(new ShowTaskBarMessageEvent(
                            translateUtil.getTranslation("Info"),
                            translateUtil.getTranslation("Updating file index")));
                    eventUtil.putEvent(new UpdateDatabaseEvent());
                }
                if (restartCount > 3) {
                    restartCount = 0;
                    //超过6天未重启
                    eventUtil.putEvent(new RestartEvent());
                }
            }
            TimeUnit.MILLISECONDS.sleep(50);
        }
    }

    private static void checkIndex() {
        int startTimes = 0;
        File startTimeCount = new File("user/startTimeCount.dat");
        boolean isFileCreated;
        if (!startTimeCount.exists()) {
            try {
                isFileCreated = startTimeCount.createNewFile();
            } catch (IOException e) {
                isFileCreated = false;
                e.printStackTrace();
            }
        } else {
            isFileCreated = true;
        }
        if (isFileCreated) {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(startTimeCount), StandardCharsets.UTF_8))) {
                //读取启动次数
                String times = reader.readLine();
                if (!(times == null || times.isEmpty())) {
                    startTimes = Integer.parseInt(times);
                    //使用次数大于3次，优化数据库
                    if (startTimes >= 3) {
                        startTimes = 0;
                        if (DatabaseUtil.getInstance().getStatus() == Enums.DatabaseStatus.NORMAL) {
                            EventUtil.getInstance().putEvent(new ShowTaskBarMessageEvent(
                                    TranslateUtil.getInstance().getTranslation("Info"),
                                    TranslateUtil.getInstance().getTranslation("Updating file index")));
                            EventUtil.getInstance().putEvent(new UpdateDatabaseEvent());
                        }
                    }
                }
                //自增后写入
                startTimes++;
                try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(startTimeCount), StandardCharsets.UTF_8))) {
                    writer.write(String.valueOf(startTimes));
                }
            } catch (Exception throwables) {
                throwables.printStackTrace();
            }
        }
    }

    private static boolean sendStartSignalFailed() {
        EventUtil eventUtil = EventUtil.getInstance();

        Event event = new ReadConfigsAndBootSystemEvent();
        eventUtil.putEvent(event);
        return eventUtil.waitForEvent(event);
    }

    private static void closeAndExit() {
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.putEvent(new CloseEvent());
    }

    private static void releaseAllDependence() {
        copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86-64/fileMonitor.dll", FILE_MONITOR_64_MD_5);
        copyOrIgnoreFile("user/getAscII.dll", "/win32-x86-64/getAscII.dll", GET_ASC_II_64_MD_5);
        copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86-64/hotkeyListener.dll", HOTKEY_LISTENER_64_MD_5);
        copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86-64/isLocalDisk.dll", IS_LOCAL_DISK_64_MD_5);
        copyOrIgnoreFile("user/fileSearcher.exe", "/win32-x86-64/fileSearcher.exe", FILE_SEARCHER_64_MD_5);
        copyOrIgnoreFile("user/fileSearcherUSN.exe", "/win32-x86-64/fileSearcherUSN.exe", FILE_SEARCHER_USN_64_MD_5);
        copyOrIgnoreFile("user/sqlite3.dll", "/win32-x86-64/sqlite3.dll", SQLITE3_64_MD_5);
        copyOrIgnoreFile("user/getHandle.dll", "/win32-x86-64/getHandle.dll", GET_HANDLE_64_MD_5);
        copyOrIgnoreFile("user/daemonProcess.exe", "/win32-x86-64/daemonProcess.exe", DAEMON_PROCESS_64_MD_5);
        copyOrIgnoreFile("user/shortcutGenerator.vbs", "/shortcutGenerator.vbs", SHORTCUT_GEN_MD_5);
    }

    private static void copyOrIgnoreFile(String path, String rootPath, String md5) {
        File target = new File(path);
        String fileMd5 = Md5Util.getMD5(target.getAbsolutePath());
        if (!target.exists() || !md5.equals(fileMd5)) {
            if (IsDebug.isDebug()) {
                System.out.println("正在重新释放文件：" + path);
            }
            InputStream resource = MainClass.class.getResourceAsStream(rootPath);
            CopyFileUtil.copyFile(resource, target);
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
            if (update.delete()) {
                System.err.println("删除更新标志失败");
            }
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
            e.printStackTrace();
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
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return true;
    }

    private static boolean isUpdateSignExist() {
        File user = new File("user/update");
        return user.exists();
    }
}
