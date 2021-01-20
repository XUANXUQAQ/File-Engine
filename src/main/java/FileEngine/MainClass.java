package FileEngine;

import FileEngine.annotation.EventRegister;
import FileEngine.configs.AllConfigs;
import FileEngine.configs.Enums;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.impl.ReadConfigsAndBootSystemEvent;
import FileEngine.eventHandler.impl.database.UpdateDatabaseEvent;
import FileEngine.eventHandler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import FileEngine.eventHandler.impl.stop.RestartEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarMessageEvent;
import FileEngine.utils.CachedThreadPoolUtil;
import FileEngine.utils.EventUtil;
import FileEngine.utils.Md5Util;
import FileEngine.utils.TranslateUtil;
import FileEngine.utils.classScan.ClassScannerUtil;
import FileEngine.utils.database.DatabaseUtil;
import FileEngine.utils.database.SQLiteUtil;
import FileEngine.utils.moveFiles.CopyFileUtil;
import FileEngine.utils.pluginSystem.PluginUtil;
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


public class MainClass {
    private static final String FILE_MONITOR_64_MD_5 = "5a8c123397c8e89614d4f9b91c2fa8f9";
    private static final String GET_ASC_II_64_MD_5 = "62a56c26e1afa7c4fa3f441aadb9d515";
    private static final String HOTKEY_LISTENER_64_MD_5 = "a212cc427a89a614402e59897c82e50d";
    private static final String IS_LOCAL_DISK_64_MD_5 = "f8a71d3496d8cc188713d521e6dfa2b2";
    private static final String FILE_SEARCHER_USN_64_MD_5 = "44e1d54765b7a7849c273dd132cbe2d1";
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
        File[] content = file.listFiles();//取得当前目录下所有文件和文件夹
        if (content == null || content.length == 0) {
            return;
        }
        for (File temp : content) {
            //直接删除文件
            if (temp.isDirectory()) {//判断是否是目录
                deleteDir(temp);//递归调用，删除目录里的内容
            }
            //删除空目录
            if (!temp.delete()) {
                System.err.println("Failed to delete " + temp.getAbsolutePath());
            }
        }
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

    private static void createPriorityTable() throws SQLException {
        try (Statement statement = SQLiteUtil.getStatement()) {
            int row = statement.executeUpdate("CREATE TABLE IF NOT EXISTS priority(SUFFIX text unique, PRIORITY INT)");
            if (row == 0) {
                statement.executeUpdate("INSERT OR IGNORE INTO priority VALUES(\"defaultPriority\", 0)");
                statement.executeUpdate("INSERT OR IGNORE INTO priority VALUES(\"exe\", 1)");
                statement.executeUpdate("INSERT OR IGNORE INTO priority VALUES(\"lnk\", 2)");
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
                    translateUtil.getTranslation("Info"),
                    translateUtil.getTranslation("New version can be updated"),
                    new ShowSettingsFrameEvent("tabAbout")));
        }
    }

    private static void checkOldApiPlugin() {
        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (PluginUtil.getInstance().isPluginTooOld()) {
            String oldPlugins = PluginUtil.getInstance().getAllOldPluginsName();
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Warning"),
                    oldPlugins + "\n" + translateUtil.getTranslation("Plugin Api is too old"),
                    new ShowSettingsFrameEvent("tabPlugin")));
        }
    }

    private static void checkRepeatPlugin() {
        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (PluginUtil.getInstance().isPluginRepeat()) {
            String repeatPlugins = PluginUtil.getInstance().getRepeatPlugins();
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Warning"),
                    repeatPlugins + "\n" + translateUtil.getTranslation("Duplicate plugin, please delete it in plugins folder"),
                    new ShowSettingsFrameEvent("tabPlugin")));
        }
    }

    private static void checkErrorPlugin() {
        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (PluginUtil.getInstance().isPluginLoadError()) {
            String errorPlugins = PluginUtil.getInstance().getLoadingErrorPlugins();
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Warning"),
                    errorPlugins + "\n" + translateUtil.getTranslation("Loading plugins error"),
                    new ShowSettingsFrameEvent("tabPlugin")));
        }
    }

    private static void initDatabase() throws SQLException {
        SQLiteUtil.initConnection("jdbc:sqlite:data.db");
        createCacheTable();
        createPriorityTable();
    }

    private static void checkPluginInfo() {
        checkOldApiPlugin();
        checkRepeatPlugin();
        checkErrorPlugin();
        checkPluginVersion();
    }

    private static void checkPluginVersion() {
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        cachedThreadPoolUtil.executeTask(() -> {
            StringBuilder notLatestPluginsBuilder = new StringBuilder();
            PluginUtil pluginUtil = PluginUtil.getInstance();
            pluginUtil.checkAllPluginsVersion(notLatestPluginsBuilder);
            EventUtil eventUtil = EventUtil.getInstance();
            TranslateUtil translateUtil = TranslateUtil.getInstance();
            String notLatestPlugins = notLatestPluginsBuilder.toString();
            if (!notLatestPlugins.isEmpty()) {
                eventUtil.putEvent(
                        new ShowTaskBarMessageEvent(translateUtil.getTranslation("Info"),
                                notLatestPlugins + "\n" +
                                        translateUtil.getTranslation("New versions of these plugins can be updated"),
                                new ShowSettingsFrameEvent("tabPlugin")));
            }
        });
    }

    private static void checkRunningDirAtDiskC() {
        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (isAtDiskC()) {
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Warning"),
                    translateUtil.getTranslation("Putting the software on the C drive may cause index failure issue")));
        }
    }

    /**
     *  -Dfile.encoding=UTF-8
     *  -Dsun.java2d.noddraw=true
     *  -Djna.library.path=user
     *  -Dswing.aatext=true
     *  -Djna.debug_load=false
     *  -DFile_Engine_Debug=true
     */
    private static void setSystemProperties() {
        //todo Debug在发布时设置为false
        System.setProperty("File_Engine_Debug", "true");
        System.setProperty("file.encoding", "UTF-8");
        System.setProperty("sun.java2d.noddraw", "true");
        System.setProperty("jna.library.path", "user");
        System.setProperty("swing.aatext", "true");
        System.setProperty("jna.debug_load", "false");
    }

    public static void main(String[] args) {
        try {
            setSystemProperties();

            if (!System.getProperty("os.arch").contains("64")) {
                JOptionPane.showMessageDialog(null, "Not 64 Bit", "ERROR", JOptionPane.ERROR_MESSAGE);
                return;
            }

            startOrIgnoreUpdateAndExit(isUpdateSignExist());

            Class.forName("org.sqlite.JDBC");

            initDatabase();

            updatePlugins();

            //清空tmp
            deleteDir(new File("tmp"));

            initFoldersAndFiles();

            initializeDllInterface();

            ClassScannerUtil.executeMethodByAnnotation(EventRegister.class, null);

            sendStartSignal();

            checkRunningDirAtDiskC();

            checkVersion();

            checkPluginInfo();

            mainLoop();

            CachedThreadPoolUtil.getInstance().shutdown();

            System.exit(0);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    private static void mainLoop() throws InterruptedException {
        Date startTime = new Date();
        Date endTime;
        int checkTimeCount = 0;
        long timeDiff;
        long div = 24 * 60 * 60 * 1000;
        int restartCount = 0;

        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();

        boolean isDatabaseDamaged = isDatabaseDamaged();
        boolean isCheckIndex = false;
        if (!IsDebug.isDebug()) {
            isCheckIndex = checkIndex();
        }

        if (isDatabaseDamaged || isCheckIndex) {
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Info"),
                    translateUtil.getTranslation("Updating file index")));
            eventUtil.putEvent(new UpdateDatabaseEvent());
        }


        while (eventUtil.isNotMainExit()) {
            // 主循环开始
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
                if (restartCount > 2) {
                    restartCount = 0;
                    //超过限定时间未重启
                    eventUtil.putEvent(new RestartEvent());
                }
            }
            TimeUnit.MILLISECONDS.sleep(50);
        }
    }

    private static boolean checkIndex() {
        int startTimes = 0;
        File startTimeCount = new File("user/startTimeCount.dat");
        boolean isFileCreated;
        boolean ret = false;
        if (startTimeCount.exists()) {
            isFileCreated = true;
        } else {
            try {
                isFileCreated = startTimeCount.createNewFile();
            } catch (IOException e) {
                isFileCreated = false;
                e.printStackTrace();
            }
        }
        if (isFileCreated) {
            try (BufferedReader reader =
                         new BufferedReader(new InputStreamReader(new FileInputStream(startTimeCount), StandardCharsets.UTF_8))) {
                //读取启动次数
                String times = reader.readLine();
                if (!(times == null || times.isEmpty())) {
                    startTimes = Integer.parseInt(times);
                    //使用次数大于3次，优化数据库
                    if (startTimes >= 3) {
                        startTimes = 0;
                        if (DatabaseUtil.getInstance().getStatus() == Enums.DatabaseStatus.NORMAL) {
                            ret = true;
                        }
                    }
                }
                //自增后写入
                startTimes++;
                try (BufferedWriter writer = new BufferedWriter(
                        new OutputStreamWriter(new FileOutputStream(startTimeCount), StandardCharsets.UTF_8))) {
                    writer.write(String.valueOf(startTimes));
                }
            } catch (Exception throwables) {
                throwables.printStackTrace();
            }
        }
        return ret;
    }

    private static void sendStartSignal() {
        EventUtil eventUtil = EventUtil.getInstance();

        Event event = new ReadConfigsAndBootSystemEvent();
        eventUtil.putEvent(event);
        if (eventUtil.waitForEvent(event)) {
            throw new RuntimeException("初始化失败");
        }
    }

    private static void releaseAllDependence() throws IOException {
        copyOrIgnoreFile("user/fileMonitor.dll", "/win32-native/fileMonitor.dll", FILE_MONITOR_64_MD_5);
        copyOrIgnoreFile("user/getAscII.dll", "/win32-native/getAscII.dll", GET_ASC_II_64_MD_5);
        copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-native/hotkeyListener.dll", HOTKEY_LISTENER_64_MD_5);
        copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-native/isLocalDisk.dll", IS_LOCAL_DISK_64_MD_5);
        copyOrIgnoreFile("user/fileSearcherUSN.exe", "/win32-native/fileSearcherUSN.exe", FILE_SEARCHER_USN_64_MD_5);
        copyOrIgnoreFile("user/sqlite3.dll", "/win32-native/sqlite3.dll", SQLITE3_64_MD_5);
        copyOrIgnoreFile("user/getHandle.dll", "/win32-native/getHandle.dll", GET_HANDLE_64_MD_5);
        copyOrIgnoreFile("user/daemonProcess.exe", "/win32-native/daemonProcess.exe", DAEMON_PROCESS_64_MD_5);
        copyOrIgnoreFile("user/shortcutGenerator.vbs", "/shortcutGenerator.vbs", SHORTCUT_GEN_MD_5);
    }

    private static void copyOrIgnoreFile(String path, String rootPath, String md5) throws IOException {
        File target = new File(path);
        String fileMd5 = Md5Util.getMD5(target.getAbsolutePath());
        if (!target.exists() || !md5.equals(fileMd5)) {
            if (IsDebug.isDebug()) {
                System.out.println("正在重新释放文件：" + path);
            }
            try (InputStream resource = MainClass.class.getResourceAsStream(rootPath)) {
                CopyFileUtil.copyFile(resource, target);
            }
        }
    }

    private static void startOrIgnoreUpdateAndExit(boolean isUpdate) throws InterruptedException, IOException {
        if (isUpdate) {
            File closeSignal = new File("tmp/closeDaemon");
            if (closeSignal.createNewFile()) {
                System.err.println("添加退出守护进程标志失败");
            }
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

    private static void initFoldersAndFiles() throws IOException {
        boolean isSucceeded;
        //user
        isSucceeded = createFileOrFolder("user", false, false);
        //plugins
        isSucceeded &= createFileOrFolder("plugins", false, false);
        //tmp
        File tmp = new File("tmp");
        String tempPath = tmp.getAbsolutePath();
        isSucceeded &= createFileOrFolder(tmp, false, false);
        isSucceeded &= createFileOrFolder(tempPath + File.separator + "fileAdded.txt", true, true);
        isSucceeded &= createFileOrFolder(tempPath + File.separator + "fileRemoved.txt", true, true);
        //cmd.txt
        isSucceeded &= createFileOrFolder("user/cmds.txt", true, false);
        releaseAllDependence();
        if (!isSucceeded) {
            throw new RuntimeException("初始化依赖项失败");
        }
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
            JSONObject info = AllConfigs.getInstance().getUpdateInfo();
            if (info != null) {
                String latestVersion = info.getString("version");
                if (Double.parseDouble(latestVersion) > Double.parseDouble(AllConfigs.version)) {
                    return false;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return true;
    }

    private static boolean isUpdateSignExist() {
        File user = new File("user/update");
        return user.exists();
    }
}
