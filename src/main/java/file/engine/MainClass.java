package file.engine;

import com.github.promeg.pinyinhelper.Pinyin;
import com.github.promeg.tinypinyin.lexicons.java.cncity.CnCityDict;
import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.dllInterface.GetHandle;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BootSystemEvent;
import file.engine.event.handler.impl.ReadConfigsEvent;
import file.engine.event.handler.impl.configs.CheckConfigsEvent;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.database.InitializeDatabaseEvent;
import file.engine.event.handler.impl.database.UpdateDatabaseEvent;
import file.engine.event.handler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.services.DatabaseService;
import file.engine.services.TranslateService;
import file.engine.utils.*;
import file.engine.utils.clazz.scan.ClassScannerUtil;
import file.engine.utils.connection.SQLiteUtil;
import file.engine.utils.file.FileUtil;
import file.engine.utils.system.properties.IsDebug;
import file.engine.utils.system.properties.IsPreview;
import lombok.SneakyThrows;

import javax.swing.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Date;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

import static file.engine.utils.StartupUtil.hasStartup;


public class MainClass {
    private static final int UPDATE_DATABASE_THRESHOLD = 3;
    private static final String FILE_MONITOR_MD5 = "b6b09efb3f1471cbac1b01df92ce983a";
    private static final String GET_ASC_II_MD5 = "dea00d07d351fece770cd0bb2ad9af10";
    private static final String HOTKEY_LISTENER_MD5 = "6d71f646529b69cff9d50fcce8d4b6e4";
    private static final String IS_LOCAL_DISK_MD5 = "59793a286030bfafe8b8f0fa84b498ba";
    private static final String FILE_SEARCHER_USN_MD5 = "6f2bd9509a0d1b0953ca006d27743b4a";
    private static final String SQLITE3_MD5 = "eb75b1a3ec5dbf58cf6d6cb307961ab5";
    private static final String GET_HANDLE_MD5 = "27a1a67f1ab5d275fbc5848b5455372d";
    private static final String SHORTCUT_GEN_MD5 = "fa4e26f99f3dcd58d827828c411ea5d7";
    private static final String RESULT_PIPE_MD5 = "35e8f5de0917a9a557d81efbab0988cc";
    private static final String GET_DPI_MD5 = "2d835577b3505af292966411b50e93b4";
    private static final String GET_START_MENU_MD5 = "3c83f83fe7273a44d9e3c27510c2e342";
    private static final String SQLITE_JDBC_MD5 = "580fd050832e37d14bc04e8d5d13b7b1";

    /**
     * 加载本地释放的dll
     *
     * @throws ClassNotFoundException 加载失败
     */
    private static void initializeDllInterface() throws ClassNotFoundException {
        Class.forName("file.engine.dllInterface.FileMonitor");
        Class.forName("file.engine.dllInterface.IsLocalDisk");
        Class.forName("file.engine.dllInterface.HotkeyListener");
        Class.forName("file.engine.dllInterface.GetAscII");
        Class.forName("file.engine.dllInterface.GetHandle");
        Class.forName("file.engine.dllInterface.ResultPipe");
    }

    /**
     * 更新启动器
     */
    private static void updateLauncher() {
        File sign = new File("user/updateLauncher");
        if (!sign.exists()) {
            return;
        }
        if (!sign.delete()) {
            System.err.println("删除插件更新标志失败");
        }
        File launcherFile = new File("tmp/" + Constants.LAUNCH_WRAPPER_NAME);
        if (!launcherFile.exists()) {
            return;
        }
        ProcessUtil.stopDaemon();
        try {
            ProcessUtil.waitForProcess(Constants.LAUNCH_WRAPPER_NAME, 10);
            File originLauncherFile = new File("..", Constants.LAUNCH_WRAPPER_NAME);
            FileUtil.copyFile(launcherFile, originLauncherFile);
            OpenFileUtil.openWithAdmin(originLauncherFile.getAbsolutePath());
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * 如果有更新标志，更新插件
     *
     * @throws FileNotFoundException 找不到文件更新失败
     */
    private static void updatePlugins() throws FileNotFoundException {
        File sign = new File("user/updatePlugin");
        if (!sign.exists()) {
            return;
        }
        if (!sign.delete()) {
            System.err.println("删除插件更新标志失败");
        }
        File tmpPlugins = new File("tmp/pluginsUpdate");
        if (IsDebug.isDebug()) {
            System.out.println("正在更新插件");
        }
        File[] files = tmpPlugins.listFiles();
        if (files == null || files.length == 0) {
            return;
        }
        for (File eachPlugin : files) {
            String pluginName = eachPlugin.getName();
            File targetPlugin = new File("plugins" + File.separator + pluginName);
            FileUtil.copyFile(new FileInputStream(eachPlugin), targetPlugin);
        }
    }


    /**
     * 检查当前版本
     */
    private static void checkVersion() {
        if (AllConfigs.getInstance().isCheckUpdateStartup()) {
            EventManagement eventManagement = EventManagement.getInstance();
            TranslateService translateService = TranslateService.getInstance();
            if (!isLatest()) {
                eventManagement.putEvent(new ShowTaskBarMessageEvent(
                        translateService.getTranslation("Info"),
                        translateService.getTranslation("New version can be updated"),
                        new ShowSettingsFrameEvent("tabAbout")));
            }
        }
    }

    /**
     * -Dfile.encoding=UTF-8
     * -Dsun.java2d.noddraw=true
     * -Djna.library.path=user
     * -Dswing.aatext=true
     * -Djna.debug_load=false
     * -DFile_Engine_Debug=true  todo Debug设置为true
     * -DFile_Engine_Preview=true
     */
    private static void setSystemProperties() {
        System.setProperty("File_Engine_Preview", "false");
        System.setProperty("file.encoding", "UTF-8");
        System.setProperty("sun.java2d.noddraw", "true");
        System.setProperty("swing.aatext", "true");
        System.setProperty("org.sqlite.lib.path", Path.of("user/").toAbsolutePath().toString());
        System.setProperty("org.sqlite.lib.name", "sqliteJDBC.dll");
    }

    private static void readAllConfigs() {
        EventManagement eventManagement = EventManagement.getInstance();
        // 发送读取所有配置事件，初始化配置
        ReadConfigsEvent readConfigsEvent = new ReadConfigsEvent();
        eventManagement.putEvent(readConfigsEvent);
        if (eventManagement.waitForEvent(readConfigsEvent)) {
            JOptionPane.showMessageDialog(null, "Read configs failed", "ERROR", JOptionPane.ERROR_MESSAGE);
            throw new RuntimeException("Read Configs Failed");
        }
    }

    private static void initDatabase() {
        EventManagement eventManagement = EventManagement.getInstance();
        InitializeDatabaseEvent initializeDatabaseEvent = new InitializeDatabaseEvent();
        eventManagement.putEvent(initializeDatabaseEvent);
        if (eventManagement.waitForEvent(initializeDatabaseEvent)) {
            JOptionPane.showMessageDialog(null, "Initialize database failed", "ERROR", JOptionPane.ERROR_MESSAGE);
            throw new RuntimeException("Initialize database failed");
        }
    }

    private static void setAllConfigs() {
        EventManagement eventManagement = EventManagement.getInstance();
        SetConfigsEvent setConfigsEvent = new SetConfigsEvent();
        eventManagement.putEvent(setConfigsEvent);
        if (eventManagement.waitForEvent(setConfigsEvent)) {
            JOptionPane.showMessageDialog(null, "Set configs failed", "ERROR", JOptionPane.ERROR_MESSAGE);
            throw new RuntimeException("Set configs failed");
        }
    }

    private static void initEventManagement() {
        // 初始化事件注册中心，注册所有事件
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.readClassList();
        {
            eventManagement.registerAllHandler();
            eventManagement.registerAllListener();
            if (IsDebug.isDebug()) {
                ClassScannerUtil.printClassesWithAnnotation();
            }
        }
        eventManagement.releaseClassesList();
    }

    private static void checkConfigs() {
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(new CheckConfigsEvent(), event -> {
            Optional<String> optional = event.getReturnValue();
            optional.ifPresent((errorInfo) -> {
                if (!errorInfo.isEmpty()) {
                    EventManagement.getInstance().putEvent(new ShowTaskBarMessageEvent(TranslateService.getInstance().getTranslation("Warning"),
                            TranslateService.getInstance().getTranslation(errorInfo), new ShowSettingsFrameEvent("tabSearchSettings")));
                }
            });
        }, null);
    }

    public static void main(String[] args) {
        try {
            setSystemProperties();

            if (!System.getProperty("os.arch").contains("64")) {
                JOptionPane.showMessageDialog(null, "Not 64 Bit", "ERROR", JOptionPane.ERROR_MESSAGE);
                throw new RuntimeException("Not 64 Bit");
            }
            updateLauncher();
            updatePlugins();

            //清空tmp
            FileUtil.deleteDir(new File("tmp"));
            initFoldersAndFiles();
            Class.forName("org.sqlite.JDBC");
            initializeDllInterface();
            initEventManagement();
            readAllConfigs();
            initDatabase();
            initPinyin();

            // 初始化全部完成，发出启动系统事件
            if (sendBootSystemSignal()) {
                JOptionPane.showMessageDialog(null, "Boot system failed", "ERROR", JOptionPane.ERROR_MESSAGE);
                throw new RuntimeException("Boot System Failed");
            }

            checkConfigs();
            setAllConfigs();
            checkVersion();

            mainLoop();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    private static void initPinyin() {
        Pinyin.init(Pinyin.newConfig().with(CnCityDict.getInstance()));
    }

    /**
     * 主循环
     */
    @SneakyThrows
    private static void mainLoop() {
        Date startTime = new Date();
        Date endTime;
        long timeDiff;
        final long div = 24 * 60 * 60 * 1000;
        boolean isNeedUpdate = SQLiteUtil.isDatabaseDamaged();
        boolean isDatabaseOutDated = false;

        if (!IsDebug.isDebug()) {
            isNeedUpdate |= checkStartTimes();
        }

        EventManagement eventManagement = EventManagement.getInstance();
        TranslateService translateService = TranslateService.getInstance();

        if (hasStartup() == 1) {
            eventManagement.putEvent(
                    new ShowTaskBarMessageEvent(translateService.getTranslation("Warning"),
                            translateService.getTranslation("The startup path is invalid")));
        }
        AllConfigs allConfigs = AllConfigs.getInstance();
        while (eventManagement.notMainExit()) {
            // 主循环开始
            //检查已工作时间
            endTime = new Date();
            timeDiff = endTime.getTime() - startTime.getTime();
            long diffDays = timeDiff / div;
            if (diffDays > 2) {
                startTime = endTime;
                //启动时间已经超过2天,更新索引
                isDatabaseOutDated = true;
            }
            // 更新标志isNeedUpdate为true，则更新
            // 数据库损坏或者重启次数超过3次，需要重建索引
            if ((isDatabaseOutDated && !GetHandle.INSTANCE.isForegroundFullscreen()) || isNeedUpdate) {
                isDatabaseOutDated = false;
                isNeedUpdate = false;
                String availableDisks = allConfigs.getAvailableDisks();
                String[] disks = RegexUtil.comma.split(availableDisks);
                //检查文件大小
                long totalSize = 0;
                for (String eachDisk : disks) {
                    String name = eachDisk.charAt(0) + ".db";
                    try {
                        Path diskDatabaseFile = Path.of("data/" + name);
                        long length = Files.size(diskDatabaseFile);
                        totalSize += length;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                eventManagement.putEvent(new ShowTaskBarMessageEvent(
                        translateService.getTranslation("Info"),
                        translateService.getTranslation("Updating file index")));
                // 文件大小超过600M则删除之前的索引，否则只进行添加
                eventManagement.putEvent(new UpdateDatabaseEvent(totalSize > 6L * 1024 * 1024 * 100),
                        event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(
                                translateService.getTranslation("Info"),
                                translateService.getTranslation("Search Done"))),
                        event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(
                                translateService.getTranslation("Warning"),
                                translateService.getTranslation("Search Failed"))));
            }
            TimeUnit.MILLISECONDS.sleep(50);
        }
    }

    /**
     * 检查启动次数，若已超过三次则发出重新更新索引信号
     *
     * @return true如果启动超过三次
     */
    private static boolean checkStartTimes() {
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
                    try {
                        startTimes = Integer.parseInt(times);
                        //使用次数大于3次，优化数据库
                        if (startTimes >= UPDATE_DATABASE_THRESHOLD) {
                            startTimes = 0;
                            if (DatabaseService.getInstance().getStatus() == Constants.Enums.DatabaseStatus.NORMAL) {
                                ret = true;
                            }
                        }
                    } catch (NumberFormatException e) {
                        ret = true;
                    }
                }
                //自增后写入
                startTimes++;
                try (BufferedWriter writer = new BufferedWriter(
                        new OutputStreamWriter(new FileOutputStream(startTimeCount), StandardCharsets.UTF_8))) {
                    writer.write(String.valueOf(startTimes));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return ret;
    }

    /**
     * 初始化全部完成，发出启动系统事件
     */
    private static boolean sendBootSystemSignal() {
        EventManagement eventManagement = EventManagement.getInstance();

        Event event = new BootSystemEvent();
        eventManagement.putEvent(event);
        return eventManagement.waitForEvent(event);
    }

    private static void releaseAllDependence() throws IOException {
        copyOrIgnoreFile("user/fileMonitor.dll", "/win32-native/fileMonitor.dll", FILE_MONITOR_MD5);
        copyOrIgnoreFile("user/getAscII.dll", "/win32-native/getAscII.dll", GET_ASC_II_MD5);
        copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-native/hotkeyListener.dll", HOTKEY_LISTENER_MD5);
        copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-native/isLocalDisk.dll", IS_LOCAL_DISK_MD5);
        copyOrIgnoreFile("user/fileSearcherUSN.exe", "/win32-native/fileSearcherUSN.exe", FILE_SEARCHER_USN_MD5);
        copyOrIgnoreFile("user/sqlite3.dll", "/win32-native/sqlite3.dll", SQLITE3_MD5);
        copyOrIgnoreFile("user/getHandle.dll", "/win32-native/getHandle.dll", GET_HANDLE_MD5);
        copyOrIgnoreFile("user/shortcutGenerator.vbs", "/shortcutGenerator.vbs", SHORTCUT_GEN_MD5);
        copyOrIgnoreFile("user/resultPipe.dll", "/win32-native/resultPipe.dll", RESULT_PIPE_MD5);
        copyOrIgnoreFile("user/getDpi.exe", "/win32-native/getDpi.exe", GET_DPI_MD5);
        copyOrIgnoreFile("user/getStartMenu.dll", "/win32-native/getStartMenu.dll", GET_START_MENU_MD5);
        copyOrIgnoreFile("user/sqliteJDBC.dll", "/win32-native/sqliteJDBC.dll", SQLITE_JDBC_MD5);
    }

    private static void copyOrIgnoreFile(String path, String rootPath, String md5) throws IOException {
        File target = new File(path);
        String fileMd5 = Md5Util.getMD5(target.getAbsolutePath());
        if (!target.exists() || !md5.equals(fileMd5)) {
            if (IsDebug.isDebug()) {
                System.out.println("正在重新释放文件：" + path);
            }
            try (InputStream resource = MainClass.class.getResourceAsStream(rootPath)) {
                FileUtil.copyFile(resource, target);
            }
        }
    }

    /**
     * 释放所有文件
     *
     * @throws IOException 释放失败
     */
    private static void initFoldersAndFiles() throws IOException {
        boolean isSucceeded;
        //user
        isSucceeded = createFileOrFolder("user", false, false);
        //plugins
        isSucceeded &= createFileOrFolder("plugins", false, false);
        //tmp
        File tmp = new File("tmp");
        isSucceeded &= createFileOrFolder(tmp, false, false);
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

    private static boolean createFileOrFolder(String path, boolean isFile, @SuppressWarnings("SameParameterValue") boolean isDeleteOnExit) {
        File file = new File(path);
        return createFileOrFolder(file, isFile, isDeleteOnExit);
    }

    private static boolean isLatest() {
        //检测是否为最新版本
        try {
            Map<String, Object> info = AllConfigs.getInstance().getUpdateInfo();
            if (info != null) {
                String latestVersion = (String) info.get("version");
                if (Double.parseDouble(latestVersion) > Double.parseDouble(Constants.version) || IsPreview.isPreview()) {
                    return false;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return true;
    }
}
