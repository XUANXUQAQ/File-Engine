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
import file.engine.dllInterface.IsLocalDisk;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BootSystemEvent;
import file.engine.event.handler.impl.ReadConfigsEvent;
import file.engine.event.handler.impl.SetSwingLaf;
import file.engine.event.handler.impl.configs.*;
import file.engine.event.handler.impl.download.StartDownloadEvent;
import file.engine.event.handler.impl.frame.searchBar.*;
import file.engine.event.handler.impl.frame.settingsFrame.GetExcludeComponentEvent;
import file.engine.event.handler.impl.hotkey.RegisterHotKeyEvent;
import file.engine.event.handler.impl.hotkey.ResponseCtrlEvent;
import file.engine.event.handler.impl.monitor.disk.StartMonitorDiskEvent;
import file.engine.event.handler.impl.plugin.ConfigsChangedEvent;
import file.engine.event.handler.impl.plugin.LoadAllPluginsEvent;
import file.engine.event.handler.impl.stop.CloseEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.event.handler.impl.taskbar.ShowTrayIconEvent;
import file.engine.services.TranslateService;
import file.engine.services.download.DownloadManager;
import file.engine.utils.RegexUtil;
import file.engine.utils.gson.GsonUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.Data;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static file.engine.configs.Constants.DEFAULT_SWING_THEME;
import static file.engine.configs.Constants.Enums;

/**
 * 保存软件运行时的所有配置信息
 */
public class AllConfigs {
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
        if ("current".equals(swingTheme)) {
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
     * 是否在将文件拖出时提示已创建快捷方式
     *
     * @return boolean
     */
    public boolean isShowTipOnCreatingLnk() {
        return configEntity.isShowTipCreatingLnk();
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
     * 获取网络代理端口
     *
     * @return proxy port
     */
    public int getProxyPort() {
        return configEntity.getProxyPort();
    }

    /**
     * 获取网络代理用户名
     *
     * @return proxy username
     */
    public String getProxyUserName() {
        return configEntity.getProxyUserName();
    }

    /**
     * 获取网络代理密码
     *
     * @return proxy password
     */
    public String getProxyPassword() {
        return configEntity.getProxyPassword();
    }

    /**
     * 获取网络代理的类型
     *
     * @return proxyType int 返回值的类型由Enums.ProxyType中定义
     * @see Constants.Enums.ProxyType
     */
    public int getProxyType() {
        return configEntity.getProxyType();
    }

    /**
     * 获取网络代理地址
     *
     * @return proxy address
     */
    public String getProxyAddress() {
        return configEntity.getProxyAddress();
    }

    /**
     * 获取搜索框默认字体颜色RGB值
     *
     * @return rgb hex
     */
    public int getSearchBarFontColor() {
        return configEntity.getSearchBarFontColor();
    }

    /**
     * 获取搜索框显示颜色
     *
     * @return rgb hex
     */
    public int getSearchBarColor() {
        return configEntity.getSearchBarColor();
    }

    /**
     * 获取热键
     *
     * @return hotkey 每个案件以 + 分开
     */
    public String getHotkey() {
        return configEntity.getHotkey();
    }

    /**
     * 获取最大cache数量
     *
     * @return cache max size
     */
    public int getCacheNumLimit() {
        return configEntity.getCacheNumLimit();
    }

    /**
     * 获取检测一次系统文件更改的时间
     *
     * @return int 单位 秒
     */
    public int getUpdateTimeLimit() {
        return configEntity.getUpdateTimeLimit();
    }

    /**
     * 获取忽略的文件夹
     *
     * @return ignored path 由 ,（逗号）分开
     */
    public String getIgnorePath() {
        return configEntity.getIgnorePath();
    }

    /**
     * 获取优先搜索文件夹
     *
     * @return priority dir
     */
    public String getPriorityFolder() {
        return configEntity.getPriorityFolder();
    }

    /**
     * 是否在打开文件时默认以管理员身份运行，绕过UAC（危险）
     *
     * @return boolean
     */
    public boolean isDefaultAdmin() {
        return configEntity.isDefaultAdmin();
    }

    /**
     * 是否在窗口失去焦点后自动关闭
     *
     * @return boolean
     */
    public boolean isLoseFocusClose() {
        return configEntity.isLoseFocusClose();
    }

    /**
     * 获取swing皮肤包名称，可由swingThemesMapper转换为Enums.SwingThemes
     *
     * @return swing name
     * @see Constants.Enums.SwingThemes
     * @see #swingThemesMapper(String)
     */
    public String getSwingTheme() {
        return configEntity.getSwingTheme();
    }

    public double getRoundRadius() {
        return configEntity.getRoundRadius();
    }

    /**
     * 获取打开上级文件夹的键盘快捷键code
     *
     * @return keycode
     */
    public int getOpenLastFolderKeyCode() {
        return configEntity.getOpenLastFolderKeyCode();
    }

    /**
     * 获取以管理员身份运行程序快捷键code
     *
     * @return keycode
     */
    public int getRunAsAdminKeyCode() {
        return configEntity.getRunAsAdminKeyCode();
    }

    /**
     * 获取复制文件路径code
     *
     * @return keycode
     */
    public int getCopyPathKeyCode() {
        return configEntity.getCopyPathKeyCode();
    }

    /**
     * 获取不透明度
     *
     * @return opacity
     */
    public float getOpacity() {
        return configEntity.getTransparency();
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
     * 获取搜索下拉框的默认颜色
     *
     * @return rgb hex
     */
    public int getLabelColor() {
        return configEntity.getLabelColor();
    }

    /**
     * 获取更新地址
     *
     * @return url
     */
    public String getUpdateAddress() {
        return configEntity.getUpdateAddress();
    }

    /**
     * 获取下拉框默认背景颜色
     *
     * @return rgb hex
     */
    public int getDefaultBackgroundColor() {
        return configEntity.getDefaultBackgroundColor();
    }

    /**
     * 获取下拉框被选中的背景颜色
     *
     * @return rgb hex
     */
    public int getLabelFontColorWithCoverage() {
        return configEntity.getFontColorWithCoverage();
    }

    /**
     * 获取下拉框被选中的字体颜色
     *
     * @return rgb hex
     */
    public int getLabelFontColor() {
        return configEntity.getFontColor();
    }

    /**
     * 获取边框颜色
     *
     * @return rgb hex
     */
    public int getBorderColor() {
        return configEntity.getBorderColor();
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
     * 是否贴靠在explorer窗口
     *
     * @return true or false
     */
    public boolean isAttachExplorer() {
        return configEntity.isAttachExplorer();
    }

    /**
     * 获取边框厚度
     *
     * @return 厚度
     */
    public int getBorderThickness() {
        return configEntity.getBorderThickness();
    }

    public boolean isCheckUpdateStartup() {
        return configEntity.isCheckUpdateStartup();
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

    public String getAvailableDisks() {
        String disks = AllConfigs.getInstance().getDisks();
        String[] splitDisks = RegexUtil.comma.split(disks);
        StringBuilder stringBuilder = new StringBuilder();
        for (String root : splitDisks) {
            if (Files.exists(Path.of(root)) && IsLocalDisk.INSTANCE.isDiskNTFS(root)) {
                stringBuilder.append(root).append(",");
            }
        }
        return stringBuilder.toString();
    }

    public String getDisks() {
        return configEntity.getDisks();
    }

    /**
     * 是否响应双击Ctrl键
     *
     * @return boolean
     */
    public boolean isResponseCtrl() {
        return configEntity.isDoubleClickCtrlOpen();
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
        } catch (IOException ignored) {
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
        updateAddressMap.put("Gitee",
                new AddressUrl(
                        "https://gitee.com/XUANXUQAQ/file-engine-version/raw/master/version.json",
                        "https://gitee.com/XUANXUQAQ/file-engine-version/raw/master/plugins.json"
                ));
    }

    /**
     * 根据用户选择的更新服务器获取地址
     *
     * @return addressUrl
     */
    public AddressUrl getUpdateUrlFromMap() {
        return getUpdateUrlFromMap(getUpdateAddress());
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
        configEntity.setIgnorePath(getFromJson(settingsInJson, "ignorePath", "C:\\Windows,"));
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
        configEntity.setSearchBarColor(getFromJson(settingsInJson, "searchBarColor", Enums.DefaultColors.DEFAULT_SEARCHBAR_COLOR));
    }

    private void readDefaultBackground(Map<String, Object> settingsInJson) {
        configEntity.setDefaultBackgroundColor(getFromJson(settingsInJson, "defaultBackground", Enums.DefaultColors.DEFAULT_WINDOW_BACKGROUND_COLOR));
    }

    private void readBorderType(Map<String, Object> settingsInJson) {
        configEntity.setBorderType(getFromJson(settingsInJson, "borderType", Enums.BorderType.AROUND.toString()));
    }

    private void readBorderColor(Map<String, Object> settingsInJson) {
        configEntity.setBorderColor(getFromJson(settingsInJson, "borderColor", Enums.DefaultColors.DEFAULT_BORDER_COLOR));
    }

    private void readFontColorWithCoverage(Map<String, Object> settingsInJson) {
        configEntity.setFontColorWithCoverage(getFromJson(settingsInJson, "fontColorWithCoverage", Enums.DefaultColors.DEFAULT_FONT_COLOR_WITH_COVERAGE));
    }

    private void readLabelColor(Map<String, Object> settingsInJson) {
        configEntity.setLabelColor(getFromJson(settingsInJson, "labelColor", Enums.DefaultColors.DEFAULT_LABEL_COLOR));
    }

    private void readFontColor(Map<String, Object> settingsInJson) {
        configEntity.setFontColor(getFromJson(settingsInJson, "fontColor", Enums.DefaultColors.DEFAULT_FONT_COLOR));
    }

    private void readSearchBarFontColor(Map<String, Object> settingsInJson) {
        configEntity.setSearchBarFontColor(getFromJson(settingsInJson, "searchBarFontColor", Enums.DefaultColors.DEFAULT_SEARCHBAR_FONT_COLOR));
    }

    private void readBorderThickness(Map<String, Object> settingsInJson) {
        configEntity.setBorderThickness(getFromJson(settingsInJson, "borderThickness", 1));
    }

    private void readLanguage(Map<String, Object> settingsInJson) {
        TranslateService translateService = TranslateService.getInstance();
        String language = getFromJson(settingsInJson, "language", TranslateService.getDefaultLang());
        configEntity.setLanguage(language);
        translateService.setLanguage(language);
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

    private String readConfigsJson() {
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
        String jsonConfig = readConfigsJson();
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
    private static String checkSettings(ConfigEntity configEntity) {
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
        initUpdateAddress();
        initCmdSetSettings();
    }

    /**
     * 使所有配置生效
     */
    private void setAllSettings() {
        EventManagement eventManagement = EventManagement.getInstance();
        AllConfigs allConfigs = AllConfigs.getInstance();
        eventManagement.putEvent(new ConfigsChangedEvent(
                allConfigs.getDefaultBackgroundColor(),
                allConfigs.getLabelColor(),
                allConfigs.getBorderColor()));
        eventManagement.putEvent(new RegisterHotKeyEvent(configEntity.getHotkey()));
        eventManagement.putEvent(new ResponseCtrlEvent(configEntity.isDoubleClickCtrlOpen()));
        eventManagement.putEvent(new SetSearchBarTransparencyEvent(configEntity.getTransparency()));
        eventManagement.putEvent(new SetSearchBarDefaultBackgroundEvent(configEntity.getDefaultBackgroundColor()));
        eventManagement.putEvent(new SetSearchBarLabelColorEvent(configEntity.getLabelColor()));
        eventManagement.putEvent(new SetSearchBarFontColorWithCoverageEvent(configEntity.getFontColorWithCoverage()));
        eventManagement.putEvent(new SetSearchBarLabelFontColorEvent(configEntity.getFontColor()));
        eventManagement.putEvent(new SetSearchBarColorEvent(configEntity.getSearchBarColor()));
        eventManagement.putEvent(new SetSearchBarFontColorEvent(configEntity.getSearchBarFontColor()));
        eventManagement.putEvent(new SetBorderEvent(allConfigs.getBorderType(), configEntity.getBorderColor(), configEntity.getBorderThickness()));
    }

    /**
     * 设置swing的主题
     *
     * @param theme theme
     */
    private static void setSwingLaf(Constants.Enums.SwingThemes theme) {
        SwingUtilities.invokeLater(() -> {
            if (theme == Constants.Enums.SwingThemes.CoreFlatIntelliJLaf) {
                FlatIntelliJLaf.setup();
            } else if (theme == Constants.Enums.SwingThemes.CoreFlatLightLaf) {
                FlatLightLaf.setup();
            } else if (theme == Constants.Enums.SwingThemes.CoreFlatDarkLaf) {
                FlatDarkLaf.setup();
            } else if (theme == Constants.Enums.SwingThemes.Arc) {
                FlatArcIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.ArcDark) {
                FlatArcDarkIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.DarkFlat) {
                FlatDarkFlatIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.Carbon) {
                FlatCarbonIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.CyanLight) {
                FlatCyanLightIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.DarkPurple) {
                FlatDarkPurpleIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.LightFlat) {
                FlatLightFlatIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.Monocai) {
                FlatMonocaiIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.OneDark) {
                FlatOneDarkIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.Gray) {
                FlatGrayIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.MaterialDesignDark) {
                FlatMaterialDesignDarkIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.MaterialLighter) {
                FlatMaterialLighterIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.MaterialDarker) {
                FlatMaterialDarkerIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.ArcDarkOrange) {
                FlatArcDarkOrangeIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.Dracula) {
                FlatDraculaIJTheme.setup();
            } else if (theme == Constants.Enums.SwingThemes.Nord) {
                FlatNordIJTheme.setup();
            } else {
                FlatDarculaLaf.setup();
            }
            ArrayList<Component> components = new ArrayList<>(Arrays.asList(JFrame.getFrames()));
            EventManagement eventManagement = EventManagement.getInstance();
            GetExcludeComponentEvent event = new GetExcludeComponentEvent();
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
            return null;
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
    }

    @EventRegister(registerClass = DeleteCmdEvent.class)
    private static void deleteCmdEvent(Event event) {
        AllConfigs allConfigs = AllConfigs.getInstance();
        DeleteCmdEvent deleteCmdEvent = (DeleteCmdEvent) event;
        allConfigs.cmdSet.remove(deleteCmdEvent.cmd);
    }

    @EventRegister(registerClass = ReadConfigsEvent.class)
    private static void readConfigsEvent(Event event) {
        AllConfigs allConfigs = AllConfigs.getInstance();
        allConfigs.readAllSettings();
        allConfigs.saveAllSettings();
    }

    @EventRegister(registerClass = CheckConfigsEvent.class)
    private static void checkConfigsEvent(Event event) {
        event.setReturnValue(checkSettings(getInstance().configEntity));
    }

    @EventRegister(registerClass = BootSystemEvent.class)
    private static void bootSystemEvent(Event event) {
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(new StartMonitorDiskEvent());
        eventManagement.putEvent(new ShowTrayIconEvent());
        eventManagement.putEvent(new LoadAllPluginsEvent("plugins"));
        eventManagement.putEvent(new SetSwingLaf("current"));
        if (isFirstRunApp) {
            checkRunningDirAtDiskC();
        }
    }

    @EventRegister(registerClass = SetConfigsEvent.class)
    private static void setAllConfigsEvent(Event event) {
        getInstance().setAllSettings();
    }

    @EventRegister(registerClass = SetSwingLaf.class)
    private static void setSwingLafEvent(Event event) {
        AllConfigs instance = getInstance();
        String theme = ((SetSwingLaf) event).theme;
        setSwingLaf(instance.swingThemesMapper(theme));
    }

    @EventRegister(registerClass = SaveConfigsEvent.class)
    private static void saveConfigsEvent(Event event) {
        AllConfigs allConfigs = getInstance();
        ConfigEntity tempConfigEntity = ((SaveConfigsEvent) event).configEntity;
        if (allConfigs.noNullValue(tempConfigEntity)) {
            allConfigs.configEntity = tempConfigEntity;
            allConfigs.saveAllSettings();
        } else {
            throw new NullPointerException("configEntity中有Null值");
        }
    }

    @EventListener(listenClass = BootSystemEvent.class)
    private static void shutdownListener(Event event) {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> EventManagement.getInstance().putEvent(new CloseEvent())));
    }

    @Data
    public static class AddressUrl {
        public final String fileEngineVersionUrl;
        public final String pluginListUrl;
    }
}
