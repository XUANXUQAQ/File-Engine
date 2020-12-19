package FileEngine.configs;

import FileEngine.IsDebug;
import FileEngine.database.SQLiteUtil;
import FileEngine.eventHandler.EventUtil;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventHandler;
import FileEngine.eventHandler.impl.ReadConfigsAndBootSystemEvent;
import FileEngine.eventHandler.impl.configs.SaveConfigsEvent;
import FileEngine.eventHandler.impl.daemon.StartDaemonEvent;
import FileEngine.eventHandler.impl.frame.searchBar.*;
import FileEngine.eventHandler.impl.hotkey.RegisterHotKeyEvent;
import FileEngine.eventHandler.impl.monitorDisk.StartMonitorDiskEvent;
import FileEngine.eventHandler.impl.plugin.LoadAllPluginsEvent;
import FileEngine.eventHandler.impl.plugin.SetPluginsCurrentThemeEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarIconEvent;
import FileEngine.r.R;
import FileEngine.translate.TranslateUtil;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SerializerFeature;
import com.formdev.flatlaf.FlatDarculaLaf;
import com.formdev.flatlaf.FlatDarkLaf;
import com.formdev.flatlaf.FlatIntelliJLaf;
import com.formdev.flatlaf.FlatLightLaf;
import com.formdev.flatlaf.intellijthemes.*;
import com.formdev.flatlaf.intellijthemes.materialthemeuilite.FlatMaterialDarkerIJTheme;
import com.formdev.flatlaf.intellijthemes.materialthemeuilite.FlatMaterialLighterIJTheme;
import com.sun.istack.internal.NotNull;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.LinkedHashSet;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 保存软件运行时的所有配置信息
 */
public class AllConfigs {
    public static final String version = "3.0"; //TODO 更改版本号

    public static final String FILE_NAME = "File-Engine-x64.exe";

    private static final int allSetMethodsNum = 27;

    public static final int defaultLabelColor = 0xcccccc;
    public static final int defaultWindowBackgroundColor = 0x333333;
    public static final int defaultBorderColor = 0xcccccc;
    public static final int defaultFontColor = 0xcccccc;
    public static final int defaultFontColorWithCoverage = 0;
    public static final int defaultSearchbarColor = 0x333333;
    public static final int defaultSearchbarFontColor = 0xffffff;

    private static volatile int cacheNumLimit;
    private static volatile boolean isShowTipCreatingLnk;
    private static volatile String hotkey;
    private static volatile int updateTimeLimit;
    private static volatile String ignorePath;
    private static volatile String priorityFolder;
    private static volatile int searchDepth;
    private static volatile boolean isDefaultAdmin;
    private static volatile boolean isLoseFocusClose;
    private static volatile int openLastFolderKeyCode;
    private static volatile int runAsAdminKeyCode;
    private static volatile int copyPathKeyCode;
    private static volatile float transparency;
    private static volatile String proxyAddress;
    private static volatile int proxyPort;
    private static volatile String proxyUserName;
    private static volatile String proxyPassword;
    private static volatile int proxyType;
    private static final File tmp = new File("tmp");
    private static final File settings = new File("user/settings.json");
    private static final LinkedHashSet<String> cmdSet = new LinkedHashSet<>();
    private static volatile int labelColor;
    private static volatile int defaultBackgroundColor;
    private static volatile int fontColorWithCoverage;
    private static volatile int fontColor;
    private static volatile int searchBarColor;
    private static volatile int searchBarFontColor;
    private static volatile int borderColor;
    private static String updateAddress;
    private static int setterCount = 0;
    private static boolean isAllowChangeSettings = false;
    private static final AtomicInteger cacheNum = new AtomicInteger(0);
    private static boolean isFirstRunApp = false;
    private static Enums.SwingThemes swingTheme;

    private static volatile AllConfigs instance = null;

    private AllConfigs() {}

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

    private void showError() {
        System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
    }

    public void allowChangeSettings() {
        isAllowChangeSettings = true;
    }

    public void denyChangeSettings() {
        isAllowChangeSettings = false;
        if (setterCount != allSetMethodsNum) {
            System.err.println("警告：set方法并未被完全调用");
        }
        setterCount = 0;
    }

    public String getSwingTheme() {
        return swingTheme.toString();
    }

    public void setSwingTheme(String swingTheme) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.swingTheme = swingThemesMapper(swingTheme);
        } else {
            showError();
        }
    }

    private Enums.SwingThemes swingThemesMapper(String swingTheme) {
        for (Enums.SwingThemes each : Enums.SwingThemes.values()) {
            if (each.toString().equals(swingTheme)) {
                return each;
            }
        }
        return Enums.SwingThemes.CoreFlatDarculaLaf;
    }

    public boolean isShowTipOnCreatingLnk() {
        return isShowTipCreatingLnk;
    }

    public boolean isFirstRun() {
        return isFirstRunApp;
    }

    public int getCacheNum() {
        return cacheNum.get();
    }

    public int getProxyPort() {
        return proxyPort;
    }

    public String getProxyUserName() {
        return proxyUserName;
    }

    public String getProxyPassword() {
        return proxyPassword;
    }

    public int getProxyType() {
        return proxyType;
    }

    public String getProxyAddress() {
        return proxyAddress;
    }

    public void setIsShowTipCreatingLnk(boolean b) {
        if (isAllowChangeSettings) {
            setterCount++;
            isShowTipCreatingLnk = b;
        } else {
            showError();
        }
    }

    public void setCacheNumLimit(int cacheNumLimit) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.cacheNumLimit = cacheNumLimit;
        } else {
            showError();
        }
    }

    public void setHotkey(String hotkey) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.hotkey = hotkey;
        } else {
            showError();
        }
    }

    public void setUpdateTimeLimit(int updateTimeLimit) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.updateTimeLimit = updateTimeLimit;
        } else {
            showError();
        }
    }

    public void setIgnorePath(String ignorePath) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.ignorePath = ignorePath;
        } else {
            showError();
        }
    }

    public void setPriorityFolder(String priorityFolder) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.priorityFolder = priorityFolder;
        } else {
            showError();
        }
    }

    public void setSearchDepth(int searchDepth) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.searchDepth = searchDepth;
        } else {
            showError();
        }
    }

    public void setIsDefaultAdmin(boolean isDefaultAdmin) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.isDefaultAdmin = isDefaultAdmin;
        } else {
            showError();
        }
    }

    public void setIsLoseFocusClose(boolean isLoseFocusClose) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.isLoseFocusClose = isLoseFocusClose;
        } else {
            showError();
        }
    }

    public void setOpenLastFolderKeyCode(int openLastFolderKeyCode) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.openLastFolderKeyCode = openLastFolderKeyCode;
        } else {
            showError();
        }
    }

    public void setRunAsAdminKeyCode(int runAsAdminKeyCode) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.runAsAdminKeyCode = runAsAdminKeyCode;
        } else {
            showError();
        }
    }

    public void setCopyPathKeyCode(int copyPathKeyCode) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.copyPathKeyCode = copyPathKeyCode;
        } else {
            showError();
        }
    }

    public void setTransparency(float transparency) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.transparency = transparency;
        } else {
            showError();
        }
    }

    public void setProxyAddress(String proxyAddress) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyAddress = proxyAddress;
        } else {
            showError();
        }
    }

    public void setProxyPort(int proxyPort) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyPort = proxyPort;
        } else {
            showError();
        }
    }

    public void setProxyUserName(String proxyUserName) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyUserName = proxyUserName;
        } else {
            showError();
        }
    }

    public void setProxyPassword(String proxyPassword) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyPassword = proxyPassword;
        } else {
            showError();
        }
    }

    public void setProxyType(int proxyType) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyType = proxyType;
        } else {
            showError();
        }
    }

    public void setLabelColor(int labelColor) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.labelColor = labelColor;
        } else {
            showError();
        }
    }

    public void setDefaultBackgroundColor(int defaultBackgroundColor) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.defaultBackgroundColor = defaultBackgroundColor;
        } else {
            showError();
        }
    }

    public void setLabelFontColorWithCoverage(int fontColorWithCoverage) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.fontColorWithCoverage = fontColorWithCoverage;
        } else {
            showError();
        }
    }

    public void setSearchBarFontColor(int color) {
        if (isAllowChangeSettings) {
            setterCount++;
            searchBarFontColor = color;
        } else {
            showError();
        }
    }

    public void setLabelFontColor(int fontColor) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.fontColor = fontColor;
        } else {
            showError();
        }
    }

    public void setBorderColor(int color) {
        if (isAllowChangeSettings) {
            setterCount++;
            borderColor = color;
        } else {
            showError();
        }
    }

    public void setSearchBarColor(int searchBarColor) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.searchBarColor = searchBarColor;
        } else {
            showError();
        }
    }

    public void setUpdateAddress(String updateAddress) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.updateAddress = updateAddress;
        } else {
            showError();
        }
    }

    public int getSearchBarFontColor() {
        return searchBarFontColor;
    }

    public int getSearchBarColor() {
        return searchBarColor;
    }

    public String getHotkey() {
        return hotkey;
    }

    public int getCacheNumLimit() {
        return cacheNumLimit;
    }

    public int getUpdateTimeLimit() {
        return updateTimeLimit;
    }

    public String getIgnorePath() {
        return ignorePath;
    }

    public String getPriorityFolder() {
        return priorityFolder;
    }

    public int getSearchDepth() {
        return searchDepth;
    }

    public boolean isDefaultAdmin() {
        return isDefaultAdmin;
    }

    public boolean isLoseFocusClose() {
        return isLoseFocusClose;
    }

    public int getOpenLastFolderKeyCode() {
        return openLastFolderKeyCode;
    }

    public int getRunAsAdminKeyCode() {
        return runAsAdminKeyCode;
    }

    public int getCopyPathKeyCode() {
        return copyPathKeyCode;
    }

    public float getTransparency() {
        return transparency;
    }

    public File getTmp() {
        return tmp;
    }

    public LinkedHashSet<String> getCmdSet() {
        return cmdSet;
    }

    public void addToCmdSet(String cmd) {
        cmdSet.add(cmd);
    }

    public int getLabelColor() {
        return labelColor;
    }

    public String getUpdateAddress() {
        return updateAddress;
    }

    public int getDefaultBackgroundColor() {
        return defaultBackgroundColor;
    }

    public int getLabelFontColorWithCoverage() {
        return fontColorWithCoverage;
    }

    public int getLabelFontColor() {
        return fontColor;
    }

    public int getBorderColor() {
        return borderColor;
    }

    public ProxyInfo getProxy() {
        return new ProxyInfo(proxyAddress, proxyPort, proxyUserName, proxyPassword, proxyType);
    }

    public void resetCacheNumToZero() {
        cacheNum.set(0);
    }

    public void decrementCacheNum() {
        cacheNum.decrementAndGet();
    }

    public void incrementCacheNum() {
        cacheNum.incrementAndGet();
    }

    /**
     * 获取数据库缓存条目数量，用于判断软件是否还能继续写入缓存
     */
    private void initCacheNum() {
        try (PreparedStatement stmt = SQLiteUtil.getPreparedStatement("SELECT COUNT(PATH) FROM cache;");
             ResultSet resultSet = stmt.executeQuery()) {
            cacheNum.set(resultSet.getInt(1));
        } catch (Exception throwables) {
            if (IsDebug.isDebug()) {
                throwables.printStackTrace();
            }
        }
    }

    private void readUpdateAddress(JSONObject settingsInJson) {
        updateAddress = (String) getFromJson(settingsInJson, "updateAddress", "jsdelivr CDN");
    }

    private void readCacheNumLimit(JSONObject settingsInJson) {
        cacheNumLimit = (int) getFromJson(settingsInJson, "cacheNumLimit", 1000);
    }

    private void readHotKey(JSONObject settingsInJson) {
        hotkey = (String) getFromJson(settingsInJson, "hotkey", "Ctrl + Alt + K");
    }

    private void readPriorityFolder(JSONObject settingsInJson) {
        priorityFolder = (String) getFromJson(settingsInJson, "priorityFolder", "");
    }

    private void readSearchDepth(JSONObject settingsInJson) {
        searchDepth = (int) getFromJson(settingsInJson, "searchDepth", 8);
    }

    private void readIgnorePath(JSONObject settingsInJson) {
        ignorePath = (String) getFromJson(settingsInJson, "ignorePath", "C:\\Windows,");
    }

    private void readUpdateTimeLimit(JSONObject settingsInJson) {
        updateTimeLimit = (int) getFromJson(settingsInJson, "updateTimeLimit", 5);
    }

    private void readIsDefaultAdmin(JSONObject settingsInJson) {
        isDefaultAdmin = (boolean) getFromJson(settingsInJson, "isDefaultAdmin", false);
    }

    private void readIsLoseFocusClose(JSONObject settingsInJson) {
        isLoseFocusClose = (boolean) getFromJson(settingsInJson, "isLoseFocusClose", true);
    }

    private void readOpenLastFolderKeyCode(JSONObject settingsInJson) {
        openLastFolderKeyCode = (int) getFromJson(settingsInJson, "openLastFolderKeyCode", 17);
    }

    private void readRunAsAdminKeyCode(JSONObject settingsInJson) {
        runAsAdminKeyCode = (int) getFromJson(settingsInJson, "runAsAdminKeyCode", 16);
    }

    private void readCopyPathKeyCode(JSONObject settingsInJson) {
        copyPathKeyCode = (int) getFromJson(settingsInJson, "copyPathKeyCode", 18);
    }

    private void readTransparency(JSONObject settingsInJson) {
        transparency = Float.parseFloat(getFromJson(settingsInJson, "transparency", 0.8f).toString());
    }

    private void readSearchBarColor(JSONObject settingsInJson) {
        searchBarColor = (int) getFromJson(settingsInJson, "searchBarColor", defaultSearchbarColor);
    }

    private void readDefaultBackground(JSONObject settingsInJson) {
        defaultBackgroundColor = (int) getFromJson(settingsInJson, "defaultBackground", defaultWindowBackgroundColor);
    }

    private void readBorderColor(JSONObject settingsInJson) {
        borderColor = (int) getFromJson(settingsInJson, "borderColor", defaultBorderColor);
    }

    private void readFontColorWithCoverage(JSONObject settingsInJson) {
        fontColorWithCoverage = (int) getFromJson(settingsInJson, "fontColorWithCoverage", defaultFontColorWithCoverage);
    }

    private void readLabelColor(JSONObject settingsInJson) {
        labelColor = (int) getFromJson(settingsInJson, "labelColor", defaultLabelColor);
    }

    private void readFontColor(JSONObject settingsInJson) {
        fontColor = (int) getFromJson(settingsInJson, "fontColor", defaultFontColor);
    }

    private void readSearchBarFontColor(JSONObject settingsInJson) {
        searchBarFontColor = (int) getFromJson(settingsInJson, "searchBarFontColor", defaultSearchbarFontColor);
    }

    private void readLanguage(JSONObject settingsInJson) {
        String language = (String) getFromJson(settingsInJson, "language", TranslateUtil.getInstance().getDefaultLang());
        TranslateUtil.getInstance().setLanguage(language);
    }

    private void readProxy(JSONObject settingsInJson) {
        proxyAddress = (String) getFromJson(settingsInJson, "proxyAddress", "");
        proxyPort = (int) getFromJson(settingsInJson, "proxyPort", 0);
        proxyUserName = (String) getFromJson(settingsInJson, "proxyUserName", "");
        proxyPassword = (String) getFromJson(settingsInJson, "proxyPassword", "");
        proxyType = (int) getFromJson(settingsInJson, "proxyType", Enums.ProxyType.PROXY_DIRECT);
    }

    private void readSwingTheme(JSONObject settingsInJson) {
        swingTheme = swingThemesMapper((String) getFromJson(settingsInJson, "swingTheme", "CoreFlatDarculaLaf"));
    }

    private void readShowTipOnCreatingLnk(JSONObject settingsInJson) {
        isShowTipCreatingLnk = (boolean) getFromJson(settingsInJson, "isShowTipOnCreatingLnk", true);
    }

    private JSONObject getSettingsJSON() {
        File settings = new File("user/settings.json");
        if (settings.exists()) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(settings), StandardCharsets.UTF_8))) {
                String line;
                StringBuilder result = new StringBuilder();
                while (null != (line = br.readLine())) {
                    result.append(line);
                }
                return JSONObject.parseObject(result.toString());
            } catch (IOException e) {
                return null;
            }
        } else {
            isFirstRunApp = true;
            return null;
        }
    }

    private Object getFromJson(JSONObject json,String key, @NotNull Object defaultObj) {
        if (json == null) {
            return defaultObj;
        }
        Object tmp = json.getOrDefault(key, defaultObj);
        return tmp != null ? tmp : defaultObj;
    }

    private void readAllSettings() {
        JSONObject settingsInJson = getSettingsJSON();
        readProxy(settingsInJson);
        readLabelColor(settingsInJson);
        readLanguage(settingsInJson);
        readBorderColor(settingsInJson);
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
        readSearchDepth(settingsInJson);
        readPriorityFolder(settingsInJson);
        readCacheNumLimit(settingsInJson);
        readUpdateAddress(settingsInJson);
        readHotKey(settingsInJson);
        readShowTipOnCreatingLnk(settingsInJson);
        readSwingTheme(settingsInJson);
        initCacheNum();
    }

    private void setAllSettings() {
        setSwing(swingTheme);
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.putTask(new SetPluginsCurrentThemeEvent(
                AllConfigs.getInstance().getDefaultBackgroundColor(),
                AllConfigs.getInstance().getLabelColor(),
                AllConfigs.getInstance().getBorderColor()));
        eventUtil.putTask(new RegisterHotKeyEvent(hotkey));
        eventUtil.putTask(new SetSearchBarTransparencyEvent(transparency));
        eventUtil.putTask(new SetSearchBarDefaultBackgroundEvent(defaultBackgroundColor));
        eventUtil.putTask(new SetSearchBarLabelColorEvent(labelColor));
        eventUtil.putTask(new SetSearchBarFontColorWithCoverageEvent(fontColorWithCoverage));
        eventUtil.putTask(new SetSearchBarLabelFontColorEvent(fontColor));
        eventUtil.putTask(new SetSearchBarColorEvent(searchBarColor));
        eventUtil.putTask(new SetSearchBarFontColorEvent(searchBarFontColor));
        eventUtil.putTask(new SetBorderColorEvent(borderColor));
    }

    public void setSwingPreview(String theme) {
        Enums.SwingThemes t = swingThemesMapper(theme);
        setSwing(t);
    }

    private void setSwing(Enums.SwingThemes theme) {
        if (theme == Enums.SwingThemes.CoreFlatIntelliJLaf) {
            FlatIntelliJLaf.install();
        } else if (theme == Enums.SwingThemes.CoreFlatLightLaf) {
            FlatLightLaf.install();
        } else if (theme == Enums.SwingThemes.CoreFlatDarkLaf) {
            FlatDarkLaf.install();
        } else if (theme == Enums.SwingThemes.Arc) {
            FlatArcIJTheme.install();
        } else if (theme == Enums.SwingThemes.ArcDark) {
            FlatArcDarkIJTheme.install();
        } else if (theme == Enums.SwingThemes.DarkFlat) {
            FlatDarkFlatIJTheme.install();
        } else if (theme == Enums.SwingThemes.Carbon) {
            FlatCarbonIJTheme.install();
        } else if (theme == Enums.SwingThemes.CyanLight) {
            FlatCyanLightIJTheme.install();
        } else if (theme == Enums.SwingThemes.DarkPurple) {
            FlatDarkPurpleIJTheme.install();
        } else if (theme == Enums.SwingThemes.LightFlat) {
            FlatLightFlatIJTheme.install();
        } else if (theme == Enums.SwingThemes.Monocai) {
            FlatMonocaiIJTheme.install();
        } else if (theme == Enums.SwingThemes.OneDark) {
            FlatOneDarkIJTheme.install();
        } else if (theme == Enums.SwingThemes.Gray) {
            FlatGrayIJTheme.install();
        } else if (theme == Enums.SwingThemes.MaterialDesignDark) {
            FlatMaterialDesignDarkIJTheme.install();
        } else if (theme == Enums.SwingThemes.MaterialLighter) {
            FlatMaterialLighterIJTheme.install();
        } else if (theme == Enums.SwingThemes.MaterialDarker) {
            FlatMaterialDarkerIJTheme.install();
        } else if (theme == Enums.SwingThemes.ArcDarkOrange) {
            FlatArcDarkOrangeIJTheme.install();
        } else if (theme == Enums.SwingThemes.Dracula) {
            FlatDraculaIJTheme.install();
        } else if (theme == Enums.SwingThemes.Nord) {
            FlatNordIJTheme.install();
        } else {
            FlatDarculaLaf.install();
        }
        for (Component c : R.getInstance().getAllComponents()) {
            SwingUtilities.updateComponentTreeUI(c);
        }
    }

    private void saveAllSettings() {
        JSONObject allSettings = new JSONObject();
        //保存设置
        allSettings.put("hotkey", hotkey);
        allSettings.put("cacheNumLimit", cacheNumLimit);
        allSettings.put("updateTimeLimit", updateTimeLimit);
        allSettings.put("ignorePath", ignorePath);
        allSettings.put("searchDepth", searchDepth);
        allSettings.put("priorityFolder", priorityFolder);
        allSettings.put("isDefaultAdmin", isDefaultAdmin);
        allSettings.put("isLoseFocusClose", isLoseFocusClose);
        allSettings.put("runAsAdminKeyCode", runAsAdminKeyCode);
        allSettings.put("openLastFolderKeyCode", openLastFolderKeyCode);
        allSettings.put("copyPathKeyCode", copyPathKeyCode);
        allSettings.put("transparency", transparency);
        allSettings.put("labelColor", labelColor);
        allSettings.put("defaultBackground", defaultBackgroundColor);
        allSettings.put("searchBarColor", searchBarColor);
        allSettings.put("fontColorWithCoverage", fontColorWithCoverage);
        allSettings.put("fontColor", fontColor);
        allSettings.put("language", TranslateUtil.getInstance().getLanguage());
        allSettings.put("proxyAddress", proxyAddress);
        allSettings.put("proxyPort", proxyPort);
        allSettings.put("proxyUserName", proxyUserName);
        allSettings.put("proxyPassword", proxyPassword);
        allSettings.put("proxyType", proxyType);
        allSettings.put("updateAddress", updateAddress);
        allSettings.put("searchBarFontColor", searchBarFontColor);
        allSettings.put("borderColor", borderColor);
        allSettings.put("isShowTipOnCreatingLnk", isShowTipCreatingLnk);
        allSettings.put("swingTheme", swingTheme.toString());
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(settings), StandardCharsets.UTF_8))) {
            String format = JSON.toJSONString(allSettings, SerializerFeature.PrettyFormat, SerializerFeature.WriteMapNullValue, SerializerFeature.WriteDateUseDateFormat);
            buffW.write(format);
        } catch (IOException ignored) {
        }
    }

    public boolean hasStartup() {
        try {
            String command = "cmd.exe /c chcp 65001 & schtasks /query /tn \"File-Engine\"";
            Process p = Runtime.getRuntime().exec(command);
            StringBuilder strBuilder = new StringBuilder();
            String line;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                while ((line = reader.readLine()) != null) {
                    strBuilder.append(line);
                }
            }
            return strBuilder.toString().contains("File-Engine");
        } catch (IOException e) {
            return false;
        }
    }

    public static void registerEventHandler() {
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.register(ReadConfigsAndBootSystemEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().readAllSettings();
                getInstance().saveAllSettings();

                eventUtil.putTask(new LoadAllPluginsEvent("plugins"));
                eventUtil.putTask(new StartMonitorDiskEvent());
                eventUtil.putTask(new ShowTaskBarIconEvent());
                if (!IsDebug.isDebug()) {
                    eventUtil.putTask(new StartDaemonEvent(new File("").getAbsolutePath()));
                }
                getInstance().setAllSettings();
            }
        });
        eventUtil.register(SaveConfigsEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().saveAllSettings();
                getInstance().setAllSettings();
            }
        });
    }
}
