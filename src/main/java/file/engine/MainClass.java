package file.engine;

import com.github.promeg.pinyinhelper.Pinyin;
import com.github.promeg.tinypinyin.lexicons.java.cncity.CnCityDict;
import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.dllInterface.CudaAccelerator;
import file.engine.dllInterface.GetHandle;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BootSystemEvent;
import file.engine.event.handler.impl.ReadConfigsEvent;
import file.engine.event.handler.impl.configs.CheckConfigsEvent;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.daemon.StartDaemonEvent;
import file.engine.event.handler.impl.daemon.StopDaemonEvent;
import file.engine.event.handler.impl.database.CheckDatabaseEmptyEvent;
import file.engine.event.handler.impl.database.InitializeDatabaseEvent;
import file.engine.event.handler.impl.database.UpdateDatabaseEvent;
import file.engine.event.handler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.services.DatabaseService;
import file.engine.services.TranslateService;
import file.engine.utils.Md5Util;
import file.engine.utils.RegexUtil;
import file.engine.utils.clazz.scan.ClassScannerUtil;
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


public class MainClass {
    private static final int UPDATE_DATABASE_THRESHOLD = 3;
    private static final String FILE_MONITOR_MD5 = "024e0f16487806e69bbbbe2bf4bd3290";
    private static final String GET_ASC_II_MD5 = "647c48fd0b56cc9a276775be6288ed97";
    private static final String HOTKEY_LISTENER_MD5 = "6d71f646529b69cff9d50fcce8d4b6e4";
    private static final String IS_LOCAL_DISK_MD5 = "f9794b94942a1ecb0acd078dfb789024";
    private static final String FILE_SEARCHER_USN_MD5 = "5a662e3ecc98b43b6c79fd1522dfa927";
    private static final String SQLITE3_MD5 = "dade4d608e258014e311867c764acf77";
    private static final String GET_HANDLE_MD5 = "41418ee9061e5940d061aeeb0db91316";
    private static final String SHORTCUT_GEN_MD5 = "fa4e26f99f3dcd58d827828c411ea5d7";
    private static final String RESULT_PIPE_MD5 = "e91b7783a6add81d8123e2698c52efb6";
    private static final String GET_DPI_MD5 = "31c3080b7a46a85e3245ed9781a0fcb3";
    private static final String GET_KNOWN_FOLDER_MD5 = "ebba6f89849fd2583934c7d48ec1e224";
    private static final String SQLITE_JDBC_MD5 = "607931b6a0655945daf1db51312b4473";
    private static final String EMPTY_RECYCLE_BIN_MD5 = "431225a47e74fe343b42e4bba741b80b";
    private static final String CUDA_ACCELERATOR_MD5 = "2b86d60f0c8e43071eb5870d46cf0bd8";
    private static final String CUDA_RUNTIME_MD5 = "d7cfc69c62e8eb977d827f46bab408da";

    public static void main(String[] args) {
        try {
            setSystemProperties();

            if (!System.getProperty("os.arch").contains("64")) {
                JOptionPane.showMessageDialog(null, "Not 64 Bit", "ERROR", JOptionPane.ERROR_MESSAGE);
                throw new RuntimeException("Not 64 Bit");
            }
            updatePlugins();

            initFoldersAndFiles();
            Class.forName("org.sqlite.JDBC");
            initializeDllInterface();
            if (CudaAccelerator.INSTANCE.isCudaAvailableOnSystem()) {
                CudaAccelerator.INSTANCE.initialize();
            }
            initEventManagement();
            updateLauncher();
            //清空tmp
            FileUtil.deleteDir(new File("tmp"));
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
        Class.forName("file.engine.dllInterface.EmptyRecycleBin");
        Class.forName("file.engine.dllInterface.CudaAccelerator");
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
            System.err.println("删除启动器更新标志失败");
        }
        File launcherFile = new File("tmp/" + Constants.LAUNCH_WRAPPER_NAME);
        if (!launcherFile.exists()) {
            System.err.println(Constants.LAUNCH_WRAPPER_NAME + "不存在于tmp中");
            return;
        }
        EventManagement eventManagement = EventManagement.getInstance();
        StopDaemonEvent stopDaemonEvent = new StopDaemonEvent();
        eventManagement.putEvent(stopDaemonEvent);
        if (eventManagement.waitForEvent(stopDaemonEvent)) {
            System.err.println("更新启动器失败");
        } else {
            File originLauncherFile = new File("..", Constants.LAUNCH_WRAPPER_NAME);
            FileUtil.copyFile(launcherFile, originLauncherFile);
            eventManagement.putEvent(new StartDaemonEvent());
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
                    eventManagement.putEvent(new ShowTaskBarMessageEvent(TranslateService.getInstance().getTranslation("Warning"),
                            errorInfo, new ShowSettingsFrameEvent("tabSearchSettings")));
                }
            });
        }, null);
    }

    private static void initPinyin() {
        Pinyin.init(Pinyin.newConfig().with(CnCityDict.getInstance()));
    }

    /**
     * 主循环
     * 检查启动时间并更新索引
     */
    @SneakyThrows
    private static void mainLoop() {
        EventManagement eventManagement = EventManagement.getInstance();
        TranslateService translateService = TranslateService.getInstance();
        AllConfigs allConfigs = AllConfigs.getInstance();

        boolean isNeedUpdate = false;
        boolean isDatabaseOutDated = false;

        if (!IsDebug.isDebug()) {
            isNeedUpdate = isStartOverThreshold();
        }
        CheckDatabaseEmptyEvent checkDatabaseEmptyEvent = new CheckDatabaseEmptyEvent();
        eventManagement.putEvent(checkDatabaseEmptyEvent);
        if (eventManagement.waitForEvent(checkDatabaseEmptyEvent)) {
            throw new RuntimeException("check database empty failed");
        }
        Optional<Object> returnValue = checkDatabaseEmptyEvent.getReturnValue();
        // 不使用lambda表达式，否则需要转换成原子或者进行包装
        if (returnValue.isPresent()) {
            isNeedUpdate |= (boolean) returnValue.get();
        }
        Date startTime = new Date();
        Date endTime;
        final long div = 24 * 60 * 60 * 1000;
        while (eventManagement.notMainExit()) {
            // 主循环开始
            //检查已工作时间
            endTime = new Date();
            final long timeDiff = endTime.getTime() - startTime.getTime();
            final long diffDays = timeDiff / div;
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
    private static boolean isStartOverThreshold() {
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
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(startTimeCount), StandardCharsets.UTF_8))) {
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
                try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(startTimeCount), StandardCharsets.UTF_8))) {
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
        copyOrIgnoreFile("user/getWindowsKnownFolder.dll", "/win32-native/getWindowsKnownFolder.dll", GET_KNOWN_FOLDER_MD5);
        copyOrIgnoreFile("user/sqliteJDBC.dll", "/win32-native/sqliteJDBC.dll", SQLITE_JDBC_MD5);
        copyOrIgnoreFile("user/emptyRecycleBin.dll", "/win32-native/emptyRecycleBin.dll", EMPTY_RECYCLE_BIN_MD5);
        copyOrIgnoreFile("user/cudaAccelerator.dll", "/win32-native/cudaAccelerator.dll", CUDA_ACCELERATOR_MD5);
        copyOrIgnoreFile("cudart64_110.dll", "/win32-native/cudart64_110.dll", CUDA_RUNTIME_MD5);
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
