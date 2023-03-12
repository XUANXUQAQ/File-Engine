package file.engine.configs;

import com.formdev.flatlaf.FlatDarculaLaf;
import com.formdev.flatlaf.FlatDarkLaf;
import com.formdev.flatlaf.FlatIntelliJLaf;
import com.formdev.flatlaf.FlatLightLaf;
import com.formdev.flatlaf.intellijthemes.*;
import com.formdev.flatlaf.intellijthemes.materialthemeuilite.FlatMaterialDarkerIJTheme;
import com.formdev.flatlaf.intellijthemes.materialthemeuilite.FlatMaterialLighterIJTheme;
import com.google.gson.Gson;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.dllInterface.GetWindowsKnownFolder;
import file.engine.dllInterface.IsLocalDisk;
import file.engine.dllInterface.gpu.GPUAccelerator;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BootSystemEvent;
import file.engine.event.handler.impl.configs.*;
import file.engine.event.handler.impl.download.StartDownloadEvent;
import file.engine.event.handler.impl.frame.settingsFrame.GetExcludeComponentEvent;
import file.engine.event.handler.impl.monitor.disk.StartMonitorDiskEvent;
import file.engine.event.handler.impl.plugin.LoadAllPluginsEvent;
import file.engine.event.handler.impl.stop.CloseEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.event.handler.impl.taskbar.ShowTrayIconEvent;
import file.engine.services.TranslateService;
import file.engine.services.download.DownloadManager;
import file.engine.utils.RegexUtil;
import file.engine.utils.gson.GsonUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.Getter;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static file.engine.configs.Constants.DEFAULT_SWING_THEME;
import static file.engine.configs.Constants.Enums;
import static file.engine.utils.StartupUtil.hasStartup;

/**
 * 保存软件运行时的所有配置信息
 */
public class AllConfigs {
    @Getter
    private volatile ConfigEntity configEntity;
    private final LinkedHashMap<String, AddressUrl> updateAddressMap = new LinkedHashMap<>();
    private final LinkedHashSet<String> cmdSet = new LinkedHashSet<>();
    private static boolean isFirstRunApp = false;

    private static volatile AllConfigs instance = null;

    private AllConfigs() {
    }

    public static AllConfigs getInstance() {
        if (instance == null) {
            synchronized (AllConfigs.class) {
                if (instance == null) {
                    instance = new AllConfigs();
                }
            }
        }
        return instance;
    }

    /**
     * 将swingTheme字符串映射到枚举类
     *
     * @param swingTheme swingTheme名称
     * @return swingTheme枚举类实例
     */
    private Constants.Enums.SwingThemes swingThemesMapper(String swingTheme) {
        if (null == swingTheme || swingTheme.isEmpty()) {
            return swingThemesMapper(configEntity.getSwingTheme());
        }
        for (Constants.Enums.SwingThemes each : Constants.Enums.SwingThemes.values()) {
            if (each.toString().equals(swingTheme)) {
                return each;
            }
        }
        return Constants.Enums.SwingThemes.MaterialLighter;
    }

    /**
     * 检测是不是第一次运行
     *
     * @return boolean
     */
    public static boolean isFirstRun() {
        return isFirstRunApp;
    }

    /**
     * 获取cmdSet的一个复制
     *
     * @return cmdSet clone
     */
    public LinkedHashSet<String> getCmdSet() {
        return new LinkedHashSet<>(cmdSet);
    }

    /**
     * 获取边框类型
     *
     * @return 边框类型
     * @see Constants.Enums.BorderType
     */
    public Constants.Enums.BorderType getBorderType() {
        String borderType = configEntity.getBorderType();
        for (Constants.Enums.BorderType each : Constants.Enums.BorderType.values()) {
            if (each.toString().equals(borderType)) {
                return each;
            }
        }
        return Constants.Enums.BorderType.AROUND;
    }

    /**
     * 获取网络代理信息
     *
     * @return ProxyInfo
     */
    public ProxyInfo getProxy() {
        return new ProxyInfo(
                configEntity.getProxyAddress(),
                configEntity.getProxyPort(),
                configEntity.getProxyUserName(),
                configEntity.getProxyPassword(),
                configEntity.getProxyType());
    }

    public Set<String> getUnAvailableDiskSet() {
        String disks = configEntity.getDisks();
        String[] splitDisks = RegexUtil.comma.split(disks);
        Set<String> set = ConcurrentHashMap.newKeySet();
        for (String root : splitDisks) {
            if (!isDiskAvailable(root)) {
                set.add(root);
            }
        }
        return set;
    }

    public String getAvailableDisks() {
        String disks = configEntity.getDisks();
        String[] splitDisks = RegexUtil.comma.split(disks);
        StringBuilder stringBuilder = new StringBuilder();
        for (String root : splitDisks) {
            if (isDiskAvailable(root)) {
                stringBuilder.append(root).append(",");
            }
        }
        return stringBuilder.toString();
    }

    private boolean isDiskAvailable(String root) {
        return Files.exists(Path.of(root)) && IsLocalDisk.INSTANCE.isDiskNTFS(root);
    }

    /**
     * 初始化cmdSet
     */
    private void initCmdSetSettings() {
        //获取所有自定义命令
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("user/cmds.txt"), StandardCharsets.UTF_8))) {
            String each;
            while ((each = br.readLine()) != null) {
                cmdSet.add(each);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 更新服务器地址
     */
    private void initUpdateAddress() {
        //todo 添加更新服务器地址
        updateAddressMap.put("jsdelivr CDN",
                new AddressUrl(
                        "https://cdn.jsdelivr.net/gh/XUANXUQAQ/File-Engine-Version/version.json",
                        "https://cdn.jsdelivr.net/gh/XUANXUQAQ/File-Engine-Version/plugins.json"
                ));
        updateAddressMap.put("GitHub",
                new AddressUrl(
                        "https://raw.githubusercontent.com/XUANXUQAQ/File-Engine-Version/master/version.json",
                        "https://raw.githubusercontent.com/XUANXUQAQ/File-Engine-Version/master/plugins.json"
                ));
        updateAddressMap.put("GitHack",
                new AddressUrl(
                        "https://raw.githack.com/XUANXUQAQ/File-Engine-Version/master/version.json",
                        "https://raw.githack.com/XUANXUQAQ/File-Engine-Version/master/plugins.json"
                ));
    }

    /**
     * 根据用户选择的更新服务器获取地址
     *
     * @return addressUrl
     */
    public AddressUrl getUpdateUrlFromMap() {
        return getUpdateUrlFromMap(configEntity.getUpdateAddress());
    }

    /**
     * 获取所有更新服务器
     *
     * @return Set
     */
    public Set<String> getAllUpdateAddress() {
        return updateAddressMap.keySet();
    }

    public AddressUrl getUpdateUrlFromMap(String updateAddress) {
        return updateAddressMap.get(updateAddress);
    }

    /**
     * 获取所有可用的本地磁盘
     *
     * @return String，用逗号隔开
     */
    private String getLocalDisks() {
        File[] files = File.listRoots();
        if (files == null || files.length == 0) {
            return "";
        }
        String diskName;
        StringBuilder stringBuilder = new StringBuilder();
        for (File each : files) {
            diskName = each.getAbsolutePath();
            if (IsLocalDisk.INSTANCE.isDiskNTFS(diskName) && IsLocalDisk.INSTANCE.isLocalDisk(diskName)) {
                stringBuilder.append(each.getAbsolutePath()).append(",");
            }
        }
        return stringBuilder.toString();
    }

    /**
     * 获取用户配置的磁盘信息
     *
     * @param settingsInJson 用户配置json
     */
    private void readDisks(Map<String, Object> settingsInJson) {
        String disks = getFromJson(settingsInJson, "disks", getLocalDisks());
        String[] stringDisk = RegexUtil.comma.split(disks);
        StringBuilder stringBuilder = new StringBuilder();
        for (String each : stringDisk) {
            stringBuilder.append(each).append(",");
        }
        configEntity.setDisks(stringBuilder.toString());
    }

    private void readIsEnableGpuAccelerate(Map<String, Object> settingsInJson) {
        boolean isEnableGpuAccelerate = getFromJson(settingsInJson, "isEnableGpuAccelerate", true);
        if (isEnableGpuAccelerate) {
            configEntity.setEnableGpuAccelerate(GPUAccelerator.INSTANCE.isGPUAvailableOnSystem());
        } else {
            configEntity.setEnableGpuAccelerate(false);
        }
    }

    private void readGpuDevice(Map<String, Object> settingsInJson) {
        String deviceNumber = getFromJson(settingsInJson, "gpuDevice", "");
        Map<String, String> devices = GPUAccelerator.INSTANCE.getDevices();
        if (!deviceNumber.isEmpty() && devices.containsValue(deviceNumber)) {
            configEntity.setGpuDevice(deviceNumber);
        } else {
            configEntity.setGpuDevice("");
        }
    }

    private void readIsAttachExplorer(Map<String, Object> settingsInJson) {
        configEntity.setAttachExplorer(getFromJson(settingsInJson, "isAttachExplorer", true));
    }

    private void readResponseCtrl(Map<String, Object> settingsInJson) {
        configEntity.setDoubleClickCtrlOpen(getFromJson(settingsInJson, "doubleClickCtrlOpen", true));
    }

    private void readRoundRadius(Map<String, Object> settingsInJson) {
        configEntity.setRoundRadius(Double.parseDouble(getFromJson(settingsInJson, "roundRadius", 20.0).toString()));
    }

    private void readUpdateAddress(Map<String, Object> settingsInJson) {
        configEntity.setUpdateAddress(getFromJson(settingsInJson, "updateAddress", "jsdelivr CDN"));
    }

    private void readCacheNumLimit(Map<String, Object> settingsInJson) {
        configEntity.setCacheNumLimit(getFromJson(settingsInJson, "cacheNumLimit", 1000));
    }

    private void readHotKey(Map<String, Object> settingsInJson) {
        configEntity.setHotkey(getFromJson(settingsInJson, "hotkey", "Ctrl + Alt + K"));
    }

    private void readPriorityFolder(Map<String, Object> settingsInJson) {
        configEntity.setPriorityFolder(getFromJson(settingsInJson, "priorityFolder", ""));
    }

    private void readIgnorePath(Map<String, Object> settingsInJson) {
        String defaultIgnore = "C:\\Windows," + GetWindowsKnownFolder.INSTANCE.getKnownFolder("{AE50C081-EBD2-438A-8655-8A092E34987A}") + ",";
        configEntity.setIgnorePath(getFromJson(settingsInJson, "ignorePath", defaultIgnore));
    }

    private void readUpdateTimeLimit(Map<String, Object> settingsInJson) {
        configEntity.setUpdateTimeLimit(getFromJson(settingsInJson, "updateTimeLimit", 5));
    }

    private void readIsDefaultAdmin(Map<String, Object> settingsInJson) {
        configEntity.setDefaultAdmin(getFromJson(settingsInJson, "isDefaultAdmin", false));
    }

    private void readIsLoseFocusClose(Map<String, Object> settingsInJson) {
        configEntity.setLoseFocusClose(getFromJson(settingsInJson, "isLoseFocusClose", true));
    }

    private void readOpenLastFolderKeyCode(Map<String, Object> settingsInJson) {
        configEntity.setOpenLastFolderKeyCode(getFromJson(settingsInJson, "openLastFolderKeyCode", 17));
    }

    private void readRunAsAdminKeyCode(Map<String, Object> settingsInJson) {
        configEntity.setRunAsAdminKeyCode(getFromJson(settingsInJson, "runAsAdminKeyCode", 16));
    }

    private void readCopyPathKeyCode(Map<String, Object> settingsInJson) {
        configEntity.setCopyPathKeyCode(getFromJson(settingsInJson, "copyPathKeyCode", 18));
    }

    private void readTransparency(Map<String, Object> settingsInJson) {
        configEntity.setTransparency(Float.parseFloat(getFromJson(settingsInJson, "transparency", 0.8f).toString()));
    }

    private void readSearchBarColor(Map<String, Object> settingsInJson) {
        var defaultColor = Constants.DefaultColors.getDefaultSearchBarColor();
        var dark = Constants.DefaultColors.getDark();
        var light = Constants.DefaultColors.getLight();
        int searchBarColor = getFromJson(settingsInJson, "searchBarColor", defaultColor.DEFAULT_SEARCHBAR_COLOR);
        if (searchBarColor == dark.DEFAULT_SEARCHBAR_COLOR || searchBarColor == light.DEFAULT_SEARCHBAR_COLOR) {
            configEntity.setSearchBarColor(defaultColor.DEFAULT_SEARCHBAR_COLOR);
        } else {
            configEntity.setSearchBarColor(searchBarColor);
        }
    }

    private void readDefaultBackground(Map<String, Object> settingsInJson) {
        var defaultColor = Constants.DefaultColors.getDefaultSearchBarColor();
        var dark = Constants.DefaultColors.getDark();
        var light = Constants.DefaultColors.getLight();
        int windowBackgroundColor = getFromJson(settingsInJson, "defaultBackground", defaultColor.DEFAULT_WINDOW_BACKGROUND_COLOR);
        if (windowBackgroundColor == dark.DEFAULT_WINDOW_BACKGROUND_COLOR || windowBackgroundColor == light.DEFAULT_WINDOW_BACKGROUND_COLOR) {
            configEntity.setDefaultBackgroundColor(defaultColor.DEFAULT_WINDOW_BACKGROUND_COLOR);
        } else {
            configEntity.setDefaultBackgroundColor(windowBackgroundColor);
        }
    }

    private void readBorderType(Map<String, Object> settingsInJson) {
        configEntity.setBorderType(getFromJson(settingsInJson, "borderType", Enums.BorderType.AROUND.toString()));
    }

    private void readBorderColor(Map<String, Object> settingsInJson) {
        var defaultColor = Constants.DefaultColors.getDefaultSearchBarColor();
        var dark = Constants.DefaultColors.getDark();
        var light = Constants.DefaultColors.getLight();
        int borderColor = getFromJson(settingsInJson, "borderColor", defaultColor.DEFAULT_BORDER_COLOR);
        if (borderColor == dark.DEFAULT_BORDER_COLOR || borderColor == light.DEFAULT_BORDER_COLOR) {
            configEntity.setBorderColor(defaultColor.DEFAULT_BORDER_COLOR);
        } else {
            configEntity.setBorderColor(borderColor);
        }
    }

    private void readFontColorWithCoverage(Map<String, Object> settingsInJson) {
        var defaultColor = Constants.DefaultColors.getDefaultSearchBarColor();
        var dark = Constants.DefaultColors.getDark();
        var light = Constants.DefaultColors.getLight();
        int fontColorWithCoverage = getFromJson(settingsInJson, "fontColorWithCoverage", defaultColor.DEFAULT_FONT_COLOR_WITH_COVERAGE);
        if (fontColorWithCoverage == light.DEFAULT_FONT_COLOR_WITH_COVERAGE || fontColorWithCoverage == dark.DEFAULT_FONT_COLOR_WITH_COVERAGE) {
            configEntity.setFontColorWithCoverage(defaultColor.DEFAULT_FONT_COLOR_WITH_COVERAGE);
        } else {
            configEntity.setFontColorWithCoverage(fontColorWithCoverage);
        }
    }

    private void readLabelColor(Map<String, Object> settingsInJson) {
        var defaultColor = Constants.DefaultColors.getDefaultSearchBarColor();
        var dark = Constants.DefaultColors.getDark();
        var light = Constants.DefaultColors.getLight();
        int labelColor = getFromJson(settingsInJson, "labelColor", defaultColor.DEFAULT_LABEL_COLOR);
        if (labelColor == dark.DEFAULT_LABEL_COLOR || labelColor == light.DEFAULT_LABEL_COLOR) {
            configEntity.setLabelColor(defaultColor.DEFAULT_LABEL_COLOR);
        } else {
            configEntity.setLabelColor(labelColor);
        }
    }

    private void readFontColor(Map<String, Object> settingsInJson) {
        var defaultColor = Constants.DefaultColors.getDefaultSearchBarColor();
        var dark = Constants.DefaultColors.getDark();
        var light = Constants.DefaultColors.getLight();
        int fontColor = getFromJson(settingsInJson, "fontColor", defaultColor.DEFAULT_FONT_COLOR);
        if (fontColor == dark.DEFAULT_FONT_COLOR || fontColor == light.DEFAULT_FONT_COLOR) {
            configEntity.setFontColor(defaultColor.DEFAULT_FONT_COLOR);
        } else {
            configEntity.setFontColor(fontColor);
        }
    }

    private void readSearchBarFontColor(Map<String, Object> settingsInJson) {
        var defaultColor = Constants.DefaultColors.getDefaultSearchBarColor();
        var dark = Constants.DefaultColors.getDark();
        var light = Constants.DefaultColors.getLight();
        int searchBarFontColor = getFromJson(settingsInJson, "searchBarFontColor", defaultColor.DEFAULT_SEARCHBAR_FONT_COLOR);
        if (searchBarFontColor == dark.DEFAULT_SEARCHBAR_FONT_COLOR || searchBarFontColor == light.DEFAULT_SEARCHBAR_FONT_COLOR) {
            configEntity.setSearchBarFontColor(defaultColor.DEFAULT_SEARCHBAR_FONT_COLOR);
        } else {
            configEntity.setSearchBarFontColor(searchBarFontColor);
        }
    }

    private void readBorderThickness(Map<String, Object> settingsInJson) {
        Object borderThickness = getFromJson(settingsInJson, "borderThickness", 1);
        configEntity.setBorderThickness(Float.parseFloat(String.valueOf(borderThickness)));
    }

    private void readLanguage(Map<String, Object> settingsInJson) {
        TranslateService translateService = TranslateService.getInstance();
        String language = getFromJson(settingsInJson, "language", TranslateService.getDefaultLang());
        configEntity.setLanguage(language);
        translateService.setLanguage(language);
    }

    @SuppressWarnings("unchecked")
    private void readAdvancedConfigs(Map<String, Object> settingsInJson) {
        Map<String, Object> advancedConfigs = (Map<String, Object>) settingsInJson.getOrDefault("advancedConfigs", new HashMap<String, Object>());
        long searchWarmupTimeoutInMills = Long.parseLong(getFromJson(advancedConfigs, "searchWarmupTimeoutInMills", (long) 10 * 60 * 1000).toString());
        long waitForInputAndPrepareSearchTimeoutInMills = Long.parseLong(getFromJson(advancedConfigs, "waitForInputAndPrepareSearchTimeoutInMills", (long) 150).toString());
        long waitForInputAndStartSearchTimeoutInMills = Long.parseLong(getFromJson(advancedConfigs, "waitForInputAndStartSearchTimeoutInMills", (long) 350).toString());
        long waitForSearchTasksTimeoutInMills = Long.parseLong(getFromJson(advancedConfigs, "waitForSearchTasksTimeoutInMills", (long) 5 * 60 * 1000).toString());
        configEntity.setAdvancedConfigEntity(new AdvancedConfigEntity(searchWarmupTimeoutInMills,
                waitForInputAndPrepareSearchTimeoutInMills,
                waitForInputAndStartSearchTimeoutInMills,
                waitForSearchTasksTimeoutInMills));
    }

    private void readSearchThreadNumber(Map<String, Object> settingsInJson) {
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int searchThreadNumber = getFromJson(settingsInJson, "searchThreadNumber", availableProcessors);
        if (searchThreadNumber > availableProcessors || searchThreadNumber < 1) {
            searchThreadNumber = availableProcessors;
        }
        configEntity.setSearchThreadNumber(searchThreadNumber);
    }

    private void readProxy(Map<String, Object> settingsInJson) {
        configEntity.setProxyAddress(getFromJson(settingsInJson, "proxyAddress", ""));
        configEntity.setProxyPort(getFromJson(settingsInJson, "proxyPort", 0));
        configEntity.setProxyUserName(getFromJson(settingsInJson, "proxyUserName", ""));
        configEntity.setProxyPassword(getFromJson(settingsInJson, "proxyPassword", ""));
        configEntity.setProxyType(getFromJson(settingsInJson, "proxyType", Enums.ProxyType.PROXY_DIRECT));
    }

    private void readCheckUpdateStartup(Map<String, Object> settings) {
        configEntity.setCheckUpdateStartup(getFromJson(settings, "isCheckUpdateStartup", true));
    }

    private void readSwingTheme(Map<String, Object> settingsInJson) {
        configEntity.setSwingTheme(getFromJson(settingsInJson, "swingTheme", DEFAULT_SWING_THEME));
    }

    private void readShowTipOnCreatingLnk(Map<String, Object> settingsInJson) {
        configEntity.setShowTipCreatingLnk(getFromJson(settingsInJson, "isShowTipOnCreatingLnk", true));
    }

    private String readConfigsJson0() {
        File settings = new File("user/settings.json");
        if (settings.exists()) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(settings), StandardCharsets.UTF_8))) {
                String line;
                StringBuilder result = new StringBuilder();
                while (null != (line = br.readLine())) {
                    result.append(line);
                }
                return result.toString();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            isFirstRunApp = true;
        }
        return "";
    }

    /**
     * 打开配置文件，解析为json
     *
     * @return JSON
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> getSettingsJSON() {
        String jsonConfig = readConfigsJson0();
        if (jsonConfig.isEmpty()) {
            return new HashMap<>();
        } else {
            return GsonUtil.getInstance().getGson().fromJson(jsonConfig, Map.class);
        }
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> getConfigMap() {
        Gson gson = GsonUtil.getInstance().getGson();
        String jsonString = gson.toJson(configEntity);
        return gson.fromJson(jsonString, Map.class);
    }

    /**
     * 尝试从json中读取，若失败则返回默认值
     *
     * @param json       json数据
     * @param key        key
     * @param defaultObj 默认值
     * @return 读取值或默认值
     */
    @SuppressWarnings("unchecked")
    private <T> T getFromJson(Map<String, Object> json, String key, Object defaultObj) {
        if (json == null) {
            return (T) defaultObj;
        }
        Object tmp = json.get(key);
        if (tmp == null) {
            if (IsDebug.isDebug()) {
                System.err.println("配置文件读取到null值   key : " + key);
            }
            return (T) defaultObj;
        }
        return (T) tmp;
    }

    /**
     * 检查配置并发出警告
     *
     * @param configEntity 配置
     * @return 错误信息
     */
    private static String checkPriorityFolder(ConfigEntity configEntity) {
        String priorityFolder = configEntity.getPriorityFolder();
        if (!priorityFolder.isEmpty() && !Files.exists(Path.of(priorityFolder))) {
            return "Priority folder does not exist";
        }
        return "";
    }

    /**
     * 读取所有配置
     */
    private void readAllSettings() {
        configEntity = new ConfigEntity();
        Map<String, Object> settingsInJson = getSettingsJSON();
        readProxy(settingsInJson);
        readLabelColor(settingsInJson);
        readRoundRadius(settingsInJson);
        readLanguage(settingsInJson);
        readBorderColor(settingsInJson);
        readBorderType(settingsInJson);
        readSearchBarColor(settingsInJson);
        readSearchBarFontColor(settingsInJson);
        readFontColor(settingsInJson);
        readFontColorWithCoverage(settingsInJson);
        readDefaultBackground(settingsInJson);
        readTransparency(settingsInJson);
        readCopyPathKeyCode(settingsInJson);
        readRunAsAdminKeyCode(settingsInJson);
        readOpenLastFolderKeyCode(settingsInJson);
        readIsLoseFocusClose(settingsInJson);
        readIsDefaultAdmin(settingsInJson);
        readUpdateTimeLimit(settingsInJson);
        readIgnorePath(settingsInJson);
        readPriorityFolder(settingsInJson);
        readCacheNumLimit(settingsInJson);
        readUpdateAddress(settingsInJson);
        readHotKey(settingsInJson);
        readResponseCtrl(settingsInJson);
        readIsAttachExplorer(settingsInJson);
        readShowTipOnCreatingLnk(settingsInJson);
        readSwingTheme(settingsInJson);
        readDisks(settingsInJson);
        readCheckUpdateStartup(settingsInJson);
        readBorderThickness(settingsInJson);
        readIsEnableGpuAccelerate(settingsInJson);
        readGpuDevice(settingsInJson);
        readSearchThreadNumber(settingsInJson);
        readAdvancedConfigs(settingsInJson);
        initUpdateAddress();
        initCmdSetSettings();
    }

    /**
     * 设置swing的主题
     *
     * @param theme theme
     */
    private static void setSwingLaf(Constants.Enums.SwingThemes theme) {
        SwingUtilities.invokeLater(() -> {
            switch (theme) {
                case CoreFlatIntelliJLaf -> FlatIntelliJLaf.setup();
                case CoreFlatLightLaf -> FlatLightLaf.setup();
                case CoreFlatDarkLaf -> FlatDarkLaf.setup();
                case Arc -> FlatArcIJTheme.setup();
                case ArcDark -> FlatArcDarkIJTheme.setup();
                case DarkFlat -> FlatDarkFlatIJTheme.setup();
                case Carbon -> FlatCarbonIJTheme.setup();
                case CyanLight -> FlatCyanLightIJTheme.setup();
                case DarkPurple -> FlatDarkPurpleIJTheme.setup();
                case LightFlat -> FlatLightFlatIJTheme.setup();
                case Monocai -> FlatMonocaiIJTheme.setup();
                case OneDark -> FlatOneDarkIJTheme.setup();
                case Gray -> FlatGrayIJTheme.setup();
                case MaterialDesignDark -> FlatMaterialDesignDarkIJTheme.setup();
                case MaterialLighter -> FlatMaterialLighterIJTheme.setup();
                case MaterialDarker -> FlatMaterialDarkerIJTheme.setup();
                case ArcDarkOrange -> FlatArcDarkOrangeIJTheme.setup();
                case Dracula -> FlatDraculaIJTheme.setup();
                case Nord -> FlatNordIJTheme.setup();
                case SolarizedDark -> FlatSolarizedDarkIJTheme.setup();
                case SolarizedLight -> FlatSolarizedLightIJTheme.setup();
                case Vuesion -> FlatVuesionIJTheme.setup();
                case XcodeDark -> FlatXcodeDarkIJTheme.setup();
                case Spacegray -> FlatSpacegrayIJTheme.setup();
                case SystemDefault -> {
                    try {
                        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
                    } catch (ClassNotFoundException | InstantiationException | IllegalAccessException |
                             UnsupportedLookAndFeelException e) {
                        e.printStackTrace();
                    }
                }
                default -> FlatDarculaLaf.setup();
            }
            ArrayList<Component> components = new ArrayList<>(Arrays.asList(Frame.getFrames()));
            var eventManagement = EventManagement.getInstance();
            var event = new GetExcludeComponentEvent();
            eventManagement.putEvent(event);
            if (!eventManagement.waitForEvent(event)) {
                Optional<Collection<? extends Component>> returnValue = event.getReturnValue();
                returnValue.ifPresent(components::addAll);
                // 更新组件主题
                for (Component frame : components) {
                    SwingUtilities.updateComponentTreeUI(frame);
                }
            }
        });
    }

    /**
     * 讲配置保存到文件
     */
    private void saveAllSettings() {
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/settings.json"), StandardCharsets.UTF_8))) {
            String format = GsonUtil.getInstance().getGson().toJson(configEntity);
            buffW.write(format);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 检查configEntity中有没有null值
     *
     * @param config configEntity
     * @return boolean
     */
    private boolean noNullValue(ConfigEntity config) {
        try {
            for (Field field : config.getClass().getDeclaredFields()) {
                field.setAccessible(true);
                Object o = field.get(config);
                if (o == null) {
                    return false;
                }
            }
        } catch (IllegalAccessException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    private void correctInvalidConfigs(ConfigEntity config) {
        if (config.isEnableGpuAccelerate()) {
            config.setEnableGpuAccelerate(GPUAccelerator.INSTANCE.isGPUAvailableOnSystem());
        }
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int searchThreadNumber = config.getSearchThreadNumber();
        if (searchThreadNumber > availableProcessors || searchThreadNumber < 1) {
            config.setSearchThreadNumber(availableProcessors);
        }
    }

    /**
     * 检查是否安装在C盘
     *
     * @return Boolean
     */
    private static boolean isAtDiskC() {
        return System.getProperty("user.dir").startsWith("C:");
    }

    /**
     * 检查软件是否运行在C盘
     */
    private static void checkRunningDirAtDiskC() {
        EventManagement eventManagement = EventManagement.getInstance();
        TranslateService translateService = TranslateService.getInstance();
        if (isAtDiskC()) {
            eventManagement.putEvent(new ShowTaskBarMessageEvent(
                    translateService.getTranslation("Warning"),
                    translateService.getTranslation("Putting the software on the C drive may cause index failure issue")));
        }
    }

    /**
     * 获取更新地址
     *
     * @return url
     */
    private String getUpdateUrl() {
        return getUpdateUrlFromMap().fileEngineVersionUrl;
    }

    /**
     * 从服务器获取最新信息
     *
     * @return JSON
     * @throws IOException 获取失败
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> getUpdateInfo() throws IOException {
        String url = getUpdateUrl();
        DownloadManager downloadManager = new DownloadManager(
                url,
                "version.json",
                new File("tmp").getAbsolutePath()
        );
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(new StartDownloadEvent(downloadManager));
        if (!downloadManager.waitFor(5000)) {
            return Collections.emptyMap();
        }
        String eachLine;
        StringBuilder strBuilder = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("tmp/version.json"), StandardCharsets.UTF_8))) {
            while ((eachLine = br.readLine()) != null) {
                strBuilder.append(eachLine);
            }
        }
        return GsonUtil.getInstance().getGson().fromJson(strBuilder.toString(), Map.class);
    }

    @EventRegister(registerClass = AddCmdEvent.class)
    private static void addCmdEvent(Event event) {
        AllConfigs allConfigs = getInstance();
        AddCmdEvent event1 = (AddCmdEvent) event;
        allConfigs.cmdSet.add(event1.cmd);
        //保存自定义命令
        StringBuilder strb = new StringBuilder();
        for (String each : allConfigs.getCmdSet()) {
            strb.append(each);
            strb.append("\n");
        }
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/cmds.txt"), StandardCharsets.UTF_8))) {
            bw.write(strb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @EventRegister(registerClass = DeleteCmdEvent.class)
    private static void deleteCmdEvent(Event event) {
        AllConfigs allConfigs = AllConfigs.getInstance();
        DeleteCmdEvent deleteCmdEvent = (DeleteCmdEvent) event;
        allConfigs.cmdSet.remove(deleteCmdEvent.cmd);
        //保存自定义命令
        StringBuilder strb = new StringBuilder();
        for (String each : allConfigs.getCmdSet()) {
            strb.append(each);
            strb.append("\n");
        }
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/cmds.txt"), StandardCharsets.UTF_8))) {
            bw.write(strb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @EventRegister(registerClass = CheckConfigsEvent.class)
    private static void checkConfigsEvent(Event event) {
        StringBuilder stringBuilder = new StringBuilder();
        TranslateService translateService = TranslateService.INSTANCE;
        stringBuilder.append(translateService.getTranslation(checkPriorityFolder(getInstance().configEntity)));
        if (!stringBuilder.toString().isEmpty()) {
            stringBuilder.append("\n");
        }
        if (hasStartup() == 1) {
            stringBuilder.append(translateService.getTranslation("The startup path is invalid"));
        }
        event.setReturnValue(stringBuilder.toString());
    }

    @EventRegister(registerClass = BootSystemEvent.class)
    private static void bootSystemEvent(Event event) {
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(new StartMonitorDiskEvent());
        eventManagement.putEvent(new ShowTrayIconEvent());
        eventManagement.putEvent(new LoadAllPluginsEvent("plugins"));
        eventManagement.putEvent(new SetSwingLaf());
        if (isFirstRunApp) {
            checkRunningDirAtDiskC();
        }
    }

    @EventRegister(registerClass = SetConfigsEvent.class)
    private static void setAllConfigsEvent(Event event) {
        SetConfigsEvent setConfigsEvent = (SetConfigsEvent) event;
        AllConfigs allConfigs = getInstance();
        if (setConfigsEvent.getConfigs() == null) {
            // MainClass初始化
            allConfigs.readAllSettings();
            setConfigsEvent.setConfigs(allConfigs.configEntity);
        } else {
            // 添加高级设置参数
            Map<String, Object> configsJson = allConfigs.getSettingsJSON();
            allConfigs.readAdvancedConfigs(configsJson);
            ConfigEntity tempConfigEntity = setConfigsEvent.getConfigs();
            tempConfigEntity.setAdvancedConfigEntity(allConfigs.configEntity.getAdvancedConfigEntity());

            // 更新设置
            if (allConfigs.noNullValue(tempConfigEntity)) {
                allConfigs.correctInvalidConfigs(tempConfigEntity);
                allConfigs.configEntity = tempConfigEntity;
                allConfigs.saveAllSettings();
            } else {
                throw new NullPointerException("configEntity中有Null值");
            }
        }
    }

    @EventRegister(registerClass = SetSwingLaf.class)
    private static void setSwingLafEvent(Event event) {
        AllConfigs instance = getInstance();
        String theme = ((SetSwingLaf) event).theme;
        setSwingLaf(instance.swingThemesMapper(theme));
    }

    @EventListener(listenClass = BootSystemEvent.class)
    private static void shutdownListener(Event event) {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> EventManagement.getInstance().putEvent(new CloseEvent())));
    }

    public record AddressUrl(String fileEngineVersionUrl, String pluginListUrl) {
    }
}
