package FileEngine.configs;

import FileEngine.IsDebug;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventHandler;
import FileEngine.eventHandler.impl.ReadConfigsAndBootSystemEvent;
import FileEngine.eventHandler.impl.SetSwingLaf;
import FileEngine.eventHandler.impl.configs.SaveConfigsEvent;
import FileEngine.eventHandler.impl.configs.SetConfigsEvent;
import FileEngine.eventHandler.impl.daemon.StartDaemonEvent;
import FileEngine.eventHandler.impl.frame.searchBar.*;
import FileEngine.eventHandler.impl.hotkey.RegisterHotKeyEvent;
import FileEngine.eventHandler.impl.monitorDisk.StartMonitorDiskEvent;
import FileEngine.eventHandler.impl.plugin.LoadAllPluginsEvent;
import FileEngine.eventHandler.impl.plugin.SetPluginsCurrentThemeEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTrayIconEvent;
import FileEngine.utils.EventUtil;
import FileEngine.utils.TranslateUtil;
import FileEngine.utils.database.SQLiteUtil;
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

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.lang.reflect.Field;
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

    public static final int defaultLabelColor = 16777215;
    public static final int defaultWindowBackgroundColor = 13421772;
    public static final int defaultBorderColor = 13421772;
    public static final int defaultFontColor = 0;
    public static final int defaultFontColorWithCoverage = 10066431;
    public static final int defaultSearchbarColor = 13421772;
    public static final int defaultSearchbarFontColor = 0;
    private volatile ConfigEntity configEntity;
    private final File settings = new File("user/settings.json");
    private final LinkedHashSet<String> cmdSet = new LinkedHashSet<>();
    private final AtomicInteger cacheNum = new AtomicInteger(0);
    private boolean isFirstRunApp = false;

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

    private Enums.SwingThemes swingThemesMapper(String swingTheme) {
        if ("current".equals(swingTheme)) {
            return swingThemesMapper(configEntity.getSwingTheme());
        }
        for (Enums.SwingThemes each : Enums.SwingThemes.values()) {
            if (each.toString().equals(swingTheme)) {
                return each;
            }
        }
        return Enums.SwingThemes.MaterialLighter;
    }

    public boolean isShowTipOnCreatingLnk() {
        return configEntity.isShowTipCreatingLnk();
    }

    public boolean isFirstRun() {
        return isFirstRunApp;
    }

    public int getCacheNum() {
        return cacheNum.get();
    }

    public int getProxyPort() {
        return configEntity.getProxyPort();
    }

    public String getProxyUserName() {
        return configEntity.getProxyUserName();
    }

    public String getProxyPassword() {
        return configEntity.getProxyPassword();
    }

    public int getProxyType() {
        return configEntity.getProxyType();
    }

    public String getProxyAddress() {
        return configEntity.getProxyAddress();
    }

    public int getSearchBarFontColor() {
        return configEntity.getSearchBarFontColor();
    }

    public int getSearchBarColor() {
        return configEntity.getSearchBarColor();
    }

    public String getHotkey() {
        return configEntity.getHotkey();
    }

    public int getCacheNumLimit() {
        return configEntity.getCacheNumLimit();
    }

    public int getUpdateTimeLimit() {
        return configEntity.getUpdateTimeLimit();
    }

    public String getIgnorePath() {
        return configEntity.getIgnorePath();
    }

    public String getPriorityFolder() {
        return configEntity.getPriorityFolder();
    }

    public int getSearchDepth() {
        return configEntity.getSearchDepth();
    }

    public boolean isDefaultAdmin() {
        return configEntity.isDefaultAdmin();
    }

    public boolean isLoseFocusClose() {
        return configEntity.isLoseFocusClose();
    }

    public String getSwingTheme() {
        return configEntity.getSwingTheme();
    }

    public int getOpenLastFolderKeyCode() {
        return configEntity.getOpenLastFolderKeyCode();
    }

    public int getRunAsAdminKeyCode() {
        return configEntity.getRunAsAdminKeyCode();
    }

    public int getCopyPathKeyCode() {
        return configEntity.getCopyPathKeyCode();
    }

    public float getTransparency() {
        return configEntity.getTransparency();
    }

    public LinkedHashSet<String> getCmdSet() {
        return cmdSet;
    }

    public void addToCmdSet(String cmd) {
        cmdSet.add(cmd);
    }

    public int getLabelColor() {
        return configEntity.getLabelColor();
    }

    public String getUpdateAddress() {
        return configEntity.getUpdateAddress();
    }

    public int getDefaultBackgroundColor() {
        return configEntity.getDefaultBackgroundColor();
    }

    public int getLabelFontColorWithCoverage() {
        return configEntity.getFontColorWithCoverage();
    }

    public int getLabelFontColor() {
        return configEntity.getFontColor();
    }

    public int getBorderColor() {
        return configEntity.getBorderColor();
    }

    public ProxyInfo getProxy() {
        return new ProxyInfo(
                configEntity.getProxyAddress(),
                configEntity.getProxyPort(),
                configEntity.getProxyUserName(),
                configEntity.getProxyPassword(),
                configEntity.getProxyType());
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
        configEntity.setUpdateAddress((String) getFromJson(settingsInJson, "updateAddress", "jsdelivr CDN"));
    }

    private void readCacheNumLimit(JSONObject settingsInJson) {
        configEntity.setCacheNumLimit((int) getFromJson(settingsInJson, "cacheNumLimit", 1000));
    }

    private void readHotKey(JSONObject settingsInJson) {
        configEntity.setHotkey((String) getFromJson(settingsInJson, "hotkey", "Ctrl + Alt + K"));
    }

    private void readPriorityFolder(JSONObject settingsInJson) {
        configEntity.setPriorityFolder((String) getFromJson(settingsInJson, "priorityFolder", ""));
    }

    private void readSearchDepth(JSONObject settingsInJson) {
        configEntity.setSearchDepth((int) getFromJson(settingsInJson, "searchDepth", 8));
    }

    private void readIgnorePath(JSONObject settingsInJson) {
        configEntity.setIgnorePath((String) getFromJson(settingsInJson, "ignorePath", "C:\\Windows,"));
    }

    private void readUpdateTimeLimit(JSONObject settingsInJson) {
        configEntity.setUpdateTimeLimit((int) getFromJson(settingsInJson, "updateTimeLimit", 5));
    }

    private void readIsDefaultAdmin(JSONObject settingsInJson) {
        configEntity.setDefaultAdmin((boolean) getFromJson(settingsInJson, "isDefaultAdmin", false));
    }

    private void readIsLoseFocusClose(JSONObject settingsInJson) {
        configEntity.setLoseFocusClose((boolean) getFromJson(settingsInJson, "isLoseFocusClose", true));
    }

    private void readOpenLastFolderKeyCode(JSONObject settingsInJson) {
        configEntity.setOpenLastFolderKeyCode((int) getFromJson(settingsInJson, "openLastFolderKeyCode", 17));
    }

    private void readRunAsAdminKeyCode(JSONObject settingsInJson) {
        configEntity.setRunAsAdminKeyCode((int) getFromJson(settingsInJson, "runAsAdminKeyCode", 16));
    }

    private void readCopyPathKeyCode(JSONObject settingsInJson) {
        configEntity.setCopyPathKeyCode((int) getFromJson(settingsInJson, "copyPathKeyCode", 18));
    }

    private void readTransparency(JSONObject settingsInJson) {
        configEntity.setTransparency(Float.parseFloat(getFromJson(settingsInJson, "transparency", 0.8f).toString()));
    }

    private void readSearchBarColor(JSONObject settingsInJson) {
        configEntity.setSearchBarColor((int) getFromJson(settingsInJson, "searchBarColor", defaultSearchbarColor));
    }

    private void readDefaultBackground(JSONObject settingsInJson) {
        configEntity.setDefaultBackgroundColor((int) getFromJson(settingsInJson, "defaultBackground", defaultWindowBackgroundColor));
    }

    private void readBorderColor(JSONObject settingsInJson) {
        configEntity.setBorderColor((int) getFromJson(settingsInJson, "borderColor", defaultBorderColor));
    }

    private void readFontColorWithCoverage(JSONObject settingsInJson) {
        configEntity.setFontColorWithCoverage((int) getFromJson(settingsInJson, "fontColorWithCoverage", defaultFontColorWithCoverage));
    }

    private void readLabelColor(JSONObject settingsInJson) {
        configEntity.setLabelColor((int) getFromJson(settingsInJson, "labelColor", defaultLabelColor));
    }

    private void readFontColor(JSONObject settingsInJson) {
        configEntity.setFontColor((int) getFromJson(settingsInJson, "fontColor", defaultFontColor));
    }

    private void readSearchBarFontColor(JSONObject settingsInJson) {
        configEntity.setSearchBarFontColor((int) getFromJson(settingsInJson, "searchBarFontColor", defaultSearchbarFontColor));
    }

    private void readLanguage(JSONObject settingsInJson) {
        String language = (String) getFromJson(settingsInJson, "language", TranslateUtil.getInstance().getDefaultLang());
        configEntity.setLanguage(language);
        TranslateUtil.getInstance().setLanguage(language);
    }

    private void readProxy(JSONObject settingsInJson) {
        configEntity.setProxyAddress((String) getFromJson(settingsInJson, "proxyAddress", ""));
        configEntity.setProxyPort((int) getFromJson(settingsInJson, "proxyPort", 0));
        configEntity.setProxyUserName((String) getFromJson(settingsInJson, "proxyUserName", ""));
        configEntity.setProxyPassword((String) getFromJson(settingsInJson, "proxyPassword", ""));
        configEntity.setProxyType((int) getFromJson(settingsInJson, "proxyType", Enums.ProxyType.PROXY_DIRECT));
    }

    private void readSwingTheme(JSONObject settingsInJson) {
        configEntity.setSwingTheme((String) getFromJson(settingsInJson, "swingTheme", "CoreFlatDarculaLaf"));
    }

    private void readShowTipOnCreatingLnk(JSONObject settingsInJson) {
        configEntity.setShowTipCreatingLnk((boolean) getFromJson(settingsInJson, "isShowTipOnCreatingLnk", true));
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

    private Object getFromJson(JSONObject json, String key, Object defaultObj) {
        if (json == null) {
            return defaultObj;
        }
        Object tmp = json.get(key);
        if (tmp == null) {
            if (IsDebug.isDebug()) {
                System.err.println("配置文件读取到null值   key : " + key);
            }
            return defaultObj;
        }
        return tmp;
    }

    private void readAllSettings() {
        configEntity = new ConfigEntity();
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
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.putEvent(new SetPluginsCurrentThemeEvent(
                AllConfigs.getInstance().getDefaultBackgroundColor(),
                AllConfigs.getInstance().getLabelColor(),
                AllConfigs.getInstance().getBorderColor()));
        eventUtil.putEvent(new RegisterHotKeyEvent(configEntity.getHotkey()));
        eventUtil.putEvent(new SetSearchBarTransparencyEvent(configEntity.getTransparency()));
        eventUtil.putEvent(new SetSearchBarDefaultBackgroundEvent(configEntity.getDefaultBackgroundColor()));
        eventUtil.putEvent(new SetSearchBarLabelColorEvent(configEntity.getLabelColor()));
        eventUtil.putEvent(new SetSearchBarFontColorWithCoverageEvent(configEntity.getFontColorWithCoverage()));
        eventUtil.putEvent(new SetSearchBarLabelFontColorEvent(configEntity.getFontColor()));
        eventUtil.putEvent(new SetSearchBarColorEvent(configEntity.getSearchBarColor()));
        eventUtil.putEvent(new SetSearchBarFontColorEvent(configEntity.getSearchBarFontColor()));
        eventUtil.putEvent(new SetBorderColorEvent(configEntity.getBorderColor()));
        eventUtil.putEvent(new SetSwingLaf("current"));
    }

    private void setSwingLaf(Enums.SwingThemes theme) {
        SwingUtilities.invokeLater(() -> {
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
            for (Frame frame : JFrame.getFrames()) {
                SwingUtilities.updateComponentTreeUI(frame);
            }
        });
    }

    private void saveAllSettings() {
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(settings), StandardCharsets.UTF_8))) {
            String format = JSON.toJSONString(
                    configEntity,
                    SerializerFeature.PrettyFormat,
                    SerializerFeature.WriteMapNullValue,
                    SerializerFeature.WriteDateUseDateFormat
            );
            buffW.write(format);
        } catch (IOException e) {
            e.printStackTrace();
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

    public static void registerEventHandler() {
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.register(ReadConfigsAndBootSystemEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                Event tmpEvent;

                AllConfigs allConfigs = AllConfigs.getInstance();

                allConfigs.readAllSettings();
                allConfigs.saveAllSettings();

                tmpEvent = new LoadAllPluginsEvent("plugins");
                eventUtil.putEvent(tmpEvent);
                eventUtil.waitForEvent(tmpEvent);

                eventUtil.putEvent(new StartMonitorDiskEvent());
                eventUtil.putEvent(new ShowTrayIconEvent());

                tmpEvent = new SetConfigsEvent();
                eventUtil.putEvent(tmpEvent);
                eventUtil.waitForEvent(tmpEvent);

                if (!IsDebug.isDebug()) {
                    eventUtil.putEvent(new StartDaemonEvent(new File("").getAbsolutePath()));
                }
            }
        });

        eventUtil.register(SetConfigsEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().setAllSettings();
            }
        });

        eventUtil.register(SetSwingLaf.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                try {
                    AllConfigs instance = getInstance();
                    String theme = ((SetSwingLaf) event).theme;
                    instance.setSwingLaf(instance.swingThemesMapper(theme));
                } catch (Exception ignored) {
                }
            }
        });

        eventUtil.register(SaveConfigsEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                AllConfigs allConfigs = getInstance();
                ConfigEntity tempConfigEntity = ((SaveConfigsEvent) event).configEntity;
                if (allConfigs.noNullValue(tempConfigEntity)) {
                    allConfigs.configEntity = tempConfigEntity;
                    allConfigs.saveAllSettings();
                } else {
                    throw new NullPointerException("configEntity中有Null值");
                }
            }
        });
    }
}
