package file.engine;

import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BootSystemEvent;
import file.engine.event.handler.impl.configs.CheckConfigsEvent;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.daemon.StartDaemonEvent;
import file.engine.event.handler.impl.daemon.StopDaemonEvent;
import file.engine.event.handler.impl.database.StartCoreEvent;
import file.engine.event.handler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.services.TranslateService;
import file.engine.utils.CompressUtil;
import file.engine.utils.Md5Util;
import file.engine.utils.clazz.scan.ClassScannerUtil;
import file.engine.utils.file.FileUtil;
import file.engine.utils.system.properties.IsDebug;
import file.engine.utils.system.properties.IsPreview;
import lombok.extern.slf4j.Slf4j;

import javax.swing.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

@Slf4j
public class MainClass {

    public static void main(String[] args) {
        try {
            setSystemProperties();

            if (!System.getProperty("os.arch").contains("64")) {
                JOptionPane.showMessageDialog(null, "Not 64 Bit", "ERROR", JOptionPane.ERROR_MESSAGE);
                throw new RuntimeException("Not 64 Bit");
            }
            updatePlugins();

            initFoldersAndFiles();
            releaseAllDependence();
            initEventManagement();
            updateLauncher();
            //清空tmp
            FileUtil.deleteDir(new File("tmp"));
            //兼容以前版本，将data文件夹移动到core
            File data = new File("data");
            if (data.exists()) {
                FileUtil.copyDir("data", "core\\data");
                FileUtil.deleteDir(data);
                Files.delete(Path.of("data"));
            }
            sendStartCoreEvent();
            setAllConfigs();
            checkConfigs();
            // 初始化全部完成，发出启动系统事件
            if (sendBootSystemSignal()) {
                JOptionPane.showMessageDialog(null, "Boot system failed", "ERROR", JOptionPane.ERROR_MESSAGE);
                throw new RuntimeException("Boot System Failed");
            }
            checkVersion();
        } catch (Exception e) {
            log.error("error: {}", e.getMessage(), e);
            System.exit(-1);
        }
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
            log.info("正在更新插件");
        }
        File[] files = tmpPlugins.listFiles();
        if (files == null) {
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
        if (AllConfigs.getInstance().getConfigEntity().isCheckUpdateStartup()) {
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
     * -DFile_Engine_Debug=true  todo Debug设置为true
     * -DFile_Engine_Preview=true
     */
    private static void setSystemProperties() {
        System.setProperty("file.encoding", "UTF-8");
    }

    private static void setAllConfigs() {
        EventManagement eventManagement = EventManagement.getInstance();
        SetConfigsEvent setConfigsEvent = new SetConfigsEvent(null);
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
        eventManagement.registerAllHandler();
        eventManagement.registerAllListener();
        if (IsDebug.isDebug()) {
            ClassScannerUtil.saveToClassListFile();
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

    private static void sendStartCoreEvent() {
        EventManagement eventManagement = EventManagement.getInstance();
        StartCoreEvent startCoreEvent = new StartCoreEvent();
        eventManagement.putEvent(startCoreEvent);
        eventManagement.waitForEvent(startCoreEvent);
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
        checkMd5AndReplace("user/hotkeyListener.dll", "/win32-native/hotkeyListener.dll");
        checkMd5AndReplace("user/isLocalDisk.dll", "/win32-native/isLocalDisk.dll");
        checkMd5AndReplace("user/getHandle.dll", "/win32-native/getHandle.dll");
        checkMd5AndReplace("user/shortcutGenerator.vbs", "/shortcutGenerator.vbs");
        checkMd5AndReplace("user/getWindowsKnownFolder.dll", "/win32-native/getWindowsKnownFolder.dll");
        checkMd5AndReplace("user/emptyRecycleBin.dll", "/win32-native/emptyRecycleBin.dll");
        checkMd5AndReplace("user/systemThemeInfo.dll", "/win32-native/systemThemeInfo.dll");
        checkMd5AndReplace("user/getDpi.exe", "/win32-native/getDpi.exe");
        String corePath = "tmp/File-Engine-Core.7z";
        checkMd5AndReplace(corePath, "/win32-native/File-Engine-Core.7z");
        CompressUtil.decompress7Z(new File(corePath), Constants.FILE_ENGINE_CORE_DIR);
    }

    private static void checkMd5AndReplace(String path, String rootPath) throws IOException {
        try (InputStream insideJar = Objects.requireNonNull(MainClass.class.getResourceAsStream(rootPath))) {
            File target = new File(path);
            String fileMd5 = Md5Util.getMD5(target.getAbsolutePath());
            String md5InsideJar = Md5Util.getMD5(insideJar);
            if (!target.exists() || !md5InsideJar.equals(fileMd5)) {
                if (IsDebug.isDebug()) {
                    log.info("正在重新释放文件：" + path);
                }
                FileUtil.copyFile(MainClass.class.getResourceAsStream(rootPath), target);
            }
        }
    }

    /**
     * 释放所有文件
     */
    private static void initFoldersAndFiles() {
        boolean isSucceeded;
        //user
        isSucceeded = createFileOrFolder("user", false, false);
        //plugins
        isSucceeded &= createFileOrFolder("plugins", false, false);
        //tmp
        File tmp = new File("tmp");
        isSucceeded &= createFileOrFolder(tmp, false, false);
        //core
        isSucceeded &= createFileOrFolder("core", false, false);
        //cmd.txt
        isSucceeded &= createFileOrFolder("user/cmds.txt", true, false);
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
            log.error("error: {}", e.getMessage(), e);
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
            if (info != null && !info.isEmpty()) {
                String latestVersion = (String) info.get("version");
                if (Double.parseDouble(latestVersion) > Double.parseDouble(Constants.version) || IsPreview.isPreview()) {
                    return false;
                }
            }
        } catch (IOException e) {
            log.error("error: {}", e.getMessage(), e);
        }
        return true;
    }
}
