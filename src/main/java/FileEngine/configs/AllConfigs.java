package FileEngine.configs;

import FileEngine.SQLiteConfig.SQLiteUtil;
import FileEngine.checkHotkey.CheckHotKeyUtil;
import FileEngine.frames.SearchBar;
import FileEngine.translate.TranslateUtil;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SerializerFeature;

import java.io.*;
import java.net.Proxy;
import java.nio.charset.StandardCharsets;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.LinkedHashSet;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 保存软件运行时的所有配置信息
 */
public class AllConfigs {
    public static final String version = "2.9"; //TODO 更改版本号

    private static final int allSetMethodsNum = 25;

    private static volatile boolean mainExit = false;
    private static volatile int cacheNumLimit;
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

    public static void allowChangeSettings() {
        isAllowChangeSettings = true;
    }

    public static void denyChangeSettings() {
        isAllowChangeSettings = false;
        if (setterCount != allSetMethodsNum) {
            System.err.println("警告：set方法并未被完全调用");
        }
        setterCount = 0;
    }

    public static boolean isFirstRun() {
        return isFirstRunApp;
    }

    public static int getCacheNum() {
        return cacheNum.get();
    }

    public static int getProxyPort() {
        return proxyPort;
    }

    public static String getProxyUserName() {
        return proxyUserName;
    }

    public static String getProxyPassword() {
        return proxyPassword;
    }

    public static int getProxyType() {
        return proxyType;
    }

    public static String getProxyAddress() {
        return proxyAddress;
    }

    public static void setCacheNumLimit(int cacheNumLimit) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.cacheNumLimit = cacheNumLimit;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setHotkey(String hotkey) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.hotkey = hotkey;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setUpdateTimeLimit(int updateTimeLimit) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.updateTimeLimit = updateTimeLimit;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setIgnorePath(String ignorePath) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.ignorePath = ignorePath;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setPriorityFolder(String priorityFolder) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.priorityFolder = priorityFolder;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setSearchDepth(int searchDepth) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.searchDepth = searchDepth;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setIsDefaultAdmin(boolean isDefaultAdmin) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.isDefaultAdmin = isDefaultAdmin;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setIsLoseFocusClose(boolean isLoseFocusClose) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.isLoseFocusClose = isLoseFocusClose;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setOpenLastFolderKeyCode(int openLastFolderKeyCode) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.openLastFolderKeyCode = openLastFolderKeyCode;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setRunAsAdminKeyCode(int runAsAdminKeyCode) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.runAsAdminKeyCode = runAsAdminKeyCode;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setCopyPathKeyCode(int copyPathKeyCode) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.copyPathKeyCode = copyPathKeyCode;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setTransparency(float transparency) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.transparency = transparency;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setProxyAddress(String proxyAddress) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyAddress = proxyAddress;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setProxyPort(int proxyPort) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyPort = proxyPort;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setProxyUserName(String proxyUserName) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyUserName = proxyUserName;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setProxyPassword(String proxyPassword) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyPassword = proxyPassword;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setProxyType(int proxyType) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.proxyType = proxyType;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setLabelColor(int labelColor) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.labelColor = labelColor;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setDefaultBackgroundColor(int defaultBackgroundColor) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.defaultBackgroundColor = defaultBackgroundColor;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setLabelFontColorWithCoverage(int fontColorWithCoverage) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.fontColorWithCoverage = fontColorWithCoverage;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setSearchBarFontColor(int color) {
        if (isAllowChangeSettings) {
            setterCount++;
            searchBarFontColor = color;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setLabelFontColor(int fontColor) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.fontColor = fontColor;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setBorderColor(int color) {
        if (isAllowChangeSettings) {
            setterCount++;
            borderColor = color;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setSearchBarColor(int searchBarColor) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.searchBarColor = searchBarColor;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static void setUpdateAddress(String updateAddress) {
        if (isAllowChangeSettings) {
            setterCount++;
            AllConfigs.updateAddress = updateAddress;
        } else {
            System.err.println("你应该在修改设置前先调用allowChangeSettings()方法，您的设置并未生效");
        }
    }

    public static int getSearchBarFontColor() {
        return searchBarFontColor;
    }

    public static int getSearchBarColor() {
        return searchBarColor;
    }

    public static String getHotkey() {
        return hotkey;
    }

    public static boolean isNotMainExit() {
        return !mainExit;
    }

    public static int getCacheNumLimit() {
        return cacheNumLimit;
    }

    public static int getUpdateTimeLimit() {
        return updateTimeLimit;
    }

    public static String getIgnorePath() {
        return ignorePath;
    }

    public static String getPriorityFolder() {
        return priorityFolder;
    }

    public static int getSearchDepth() {
        return searchDepth;
    }

    public static boolean isDefaultAdmin() {
        return isDefaultAdmin;
    }

    public static boolean isLoseFocusClose() {
        return isLoseFocusClose;
    }

    public static int getOpenLastFolderKeyCode() {
        return openLastFolderKeyCode;
    }

    public static int getRunAsAdminKeyCode() {
        return runAsAdminKeyCode;
    }

    public static int getCopyPathKeyCode() {
        return copyPathKeyCode;
    }

    public static float getTransparency() {
        return transparency;
    }

    public static File getTmp() {
        return tmp;
    }

    public static LinkedHashSet<String> getCmdSet() {
        return cmdSet;
    }

    public static void addToCmdSet(String cmd) {
        cmdSet.add(cmd);
    }

    public static int getLabelColor() {
        return labelColor;
    }

    public static String getUpdateAddress() {
        return updateAddress;
    }

    public static int getDefaultBackgroundColor() {
        return defaultBackgroundColor;
    }

    public static int getLabelFontColorWithCoverage() {
        return fontColorWithCoverage;
    }

    public static int getLabelFontColor() {
        return fontColor;
    }

    public static int getBorderColor() {
        return borderColor;
    }

    public static String getName() {
        return "File-Engine-x64.exe";
    }

    public static ProxyInfo getProxy() {
        return new ProxyInfo(proxyAddress, proxyPort, proxyUserName, proxyPassword, proxyType);
    }

    public static void setMainExit(boolean b) {
        mainExit = b;
    }

    public static void resetCacheNumToZero() {
        cacheNum.set(0);
    }

    public static void decrementCacheNum() {
        cacheNum.decrementAndGet();
    }

    public static void incrementCacheNum() {
        cacheNum.incrementAndGet();
    }

    /**
     * 获取数据库缓存条目数量，用于判断软件是否还能继续写入缓存
     */
    private static void initCacheNum() {
        try (PreparedStatement pStmt = SQLiteUtil.getConnection().prepareStatement("SELECT COUNT(PATH) FROM cache;");
             ResultSet resultSet = pStmt.executeQuery()) {
            cacheNum.set(resultSet.getInt(1));
        } catch (Exception throwables) {
            if (AllConfigs.isDebug()) {
                throwables.printStackTrace();
            }
        }
    }

    private static void readUpdateAddress(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("updateAddress")) {
                updateAddress = settingsInJson.getString("updateAddress");
            } else {
                updateAddress = "jsdelivr CDN";
            }
        } else {
            updateAddress = "jsdelivr CDN";
        }
    }

    private static void readCacheNumLimit(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("cacheNumLimit")) {
                cacheNumLimit = settingsInJson.getInteger("cacheNumLimit");
            } else {
                cacheNumLimit = 1000;
            }
        }else {
            cacheNumLimit = 1000;
        }
    }

    private static void readHotKey(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("hotkey")) {
                String tmp = settingsInJson.getString("hotkey");
                if (tmp == null) {
                    hotkey = "Ctrl + Alt + K";
                } else {
                    hotkey = tmp;
                }
            } else {
                hotkey = "Ctrl + Alt + K";
            }
        }else {
            hotkey = "Ctrl + Alt + K";
        }
    }

    private static void readPriorityFolder(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("priorityFolder")) {
                String tmp = settingsInJson.getString("priorityFolder");
                if (tmp == null) {
                    priorityFolder = "";
                } else {
                    priorityFolder = tmp;
                }
            } else {
                priorityFolder = "";
            }
        }else {
            priorityFolder = "";
        }
    }

    private static void readSearchDepth(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("searchDepth")) {
                searchDepth = settingsInJson.getInteger("searchDepth");
            } else {
                searchDepth = 8;
            }
        }else {
            searchDepth = 8;
        }
    }

    private static void readIgnorePath(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("ignorePath")) {
                String tmp = settingsInJson.getString("ignorePath");
                if (tmp == null) {
                    ignorePath = "C:\\Windows,";
                } else {
                    ignorePath = tmp;
                }
            } else {
                ignorePath = "C:\\Windows,";
            }
        }else {
            ignorePath = "C:\\Windows,";
        }
    }

    private static void readUpdateTimeLimit(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("updateTimeLimit")) {
                updateTimeLimit = settingsInJson.getInteger("updateTimeLimit");
            } else {
                updateTimeLimit = 5;
            }
        } else {
            updateTimeLimit = 5;
        }
    }

    private static void readIsDefaultAdmin(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("isDefaultAdmin")) {
                isDefaultAdmin = settingsInJson.getBoolean("isDefaultAdmin");
            } else {
                isDefaultAdmin = false;
            }
        } else {
            isDefaultAdmin = false;
        }
    }

    private static void readIsLoseFocusClose(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("isLoseFocusClose")) {
                isLoseFocusClose = settingsInJson.getBoolean("isLoseFocusClose");
            } else {
                isLoseFocusClose = true;
            }
        } else {
            isLoseFocusClose = true;
        }
    }

    private static void readOpenLastFolderKeyCode(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("openLastFolderKeyCode")) {
                openLastFolderKeyCode = settingsInJson.getInteger("openLastFolderKeyCode");
            } else {
                openLastFolderKeyCode = 17;
            }
        } else {
            openLastFolderKeyCode = 17;
        }
    }

    private static void readRunAsAdminKeyCode(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("runAsAdminKeyCode")) {
                runAsAdminKeyCode = settingsInJson.getInteger("runAsAdminKeyCode");
            } else {
                runAsAdminKeyCode = 16;
            }
        }else {
            runAsAdminKeyCode = 16;
        }
    }

    private static void readCopyPathKeyCode(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("copyPathKeyCode")) {
                copyPathKeyCode = settingsInJson.getInteger("copyPathKeyCode");
            } else {
                copyPathKeyCode = 18;
            }
        } else {
            copyPathKeyCode = 18;
        }
    }

    private static void readTransparency(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("transparency")) {
                transparency = settingsInJson.getFloat("transparency");
            } else {
                transparency = 0.8f;
            }
        }else {
            transparency = 0.8f;
        }
    }

    private static void readSearchBarColor(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("searchBarColor")) {
                searchBarColor = settingsInJson.getInteger("searchBarColor");
            } else {
                searchBarColor = 0xffffff;
            }
        } else {
            searchBarColor = 0xffffff;
        }
    }

    private static void readDefaultBackground(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("defaultBackground")) {
                defaultBackgroundColor = settingsInJson.getInteger("defaultBackground");
            } else {
                defaultBackgroundColor = 0xffffff;
            }
        } else {
            defaultBackgroundColor = 0xffffff;
        }
    }

    private static void readBorderColor(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("borderColor")) {
                borderColor = settingsInJson.getInteger("borderColor");
            } else {
                borderColor = 0xffffff;
            }
        } else {
            borderColor = 0xffffff;
        }
    }

    private static void readFontColorWithCoverage(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("fontColorWithCoverage")) {
                fontColorWithCoverage = settingsInJson.getInteger("fontColorWithCoverage");
            } else {
                fontColorWithCoverage = 0x6666ff;
            }
        } else {
            fontColorWithCoverage = 0x6666ff;
        }
    }

    private static void readLabelColor(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("labelColor")) {
                labelColor = settingsInJson.getInteger("labelColor");
            } else {
                labelColor = 0xcccccc;
            }
        }else {
            labelColor = 0xcccccc;
        }
    }

    private static void readFontColor(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("fontColor")) {
                fontColor = settingsInJson.getInteger("fontColor");
            } else {
                fontColor = 0;
            }
        } else {
            fontColor = 0;
        }
    }

    private static void readSearchBarFontColor(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("searchBarFontColor")) {
                searchBarFontColor = settingsInJson.getInteger("searchBarFontColor");
            } else {
                searchBarFontColor = 0;
            }
        } else {
            searchBarFontColor = 0;
        }
    }

    private static void readLanguage(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("language")) {
                String language = settingsInJson.getString("language");
                if (language == null || language.isEmpty()) {
                    language = TranslateUtil.getInstance().getDefaultLang();
                }
                TranslateUtil.getInstance().setLanguage(language);
            } else {
                TranslateUtil.getInstance().setLanguage(TranslateUtil.getInstance().getDefaultLang());
            }
        } else {
            TranslateUtil.getInstance().setLanguage(TranslateUtil.getInstance().getDefaultLang());
        }
    }

    private static void readProxy(JSONObject settingsInJson) {
        if (settingsInJson != null) {
            if (settingsInJson.containsKey("proxyAddress")) {
                proxyAddress = settingsInJson.getString("proxyAddress");
            } else {
                proxyAddress = "";
            }
            if (settingsInJson.containsKey("proxyPort")) {
                proxyPort = settingsInJson.getInteger("proxyPort");
            } else {
                proxyPort = 0;
            }
            if (settingsInJson.containsKey("proxyUserName")) {
                proxyUserName = settingsInJson.getString("proxyUserName");
            } else {
                proxyUserName = "";
            }
            if (settingsInJson.containsKey("proxyPassword")) {
                proxyPassword = settingsInJson.getString("proxyPassword");
            } else {
                proxyPassword = "";
            }
            if (settingsInJson.containsKey("proxyType")) {
                proxyType = settingsInJson.getInteger("proxyType");
            } else {
                proxyType = AllConfigs.ProxyType.PROXY_DIRECT;
            }
        } else {
            proxyAddress = "";
            proxyPort = 0;
            proxyUserName = "";
            proxyPassword = "";
            proxyType = ProxyType.PROXY_DIRECT;
        }
    }

    private static JSONObject readSettingsJSON() {
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

    public static void readAllSettings() {
        JSONObject settingsInJson = readSettingsJSON();
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
        initCacheNum();
        setAllSettings();
        saveAllSettings();
    }

    public static void setAllSettings() {
        CheckHotKeyUtil.getInstance().registerHotkey(hotkey);
        SearchBar searchBar = SearchBar.getInstance();
        searchBar.setTransparency(transparency);
        searchBar.setDefaultBackgroundColor(defaultBackgroundColor);
        searchBar.setLabelColor(labelColor);
        searchBar.setFontColorWithCoverage(fontColorWithCoverage);
        searchBar.setLabelFontColor(fontColor);
        searchBar.setSearchBarColor(searchBarColor);
        searchBar.setSearchBarFontColor(searchBarFontColor);
        searchBar.setBorderColor(borderColor);
    }

    public static void saveAllSettings() {
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
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(settings), StandardCharsets.UTF_8))) {
            String format = JSON.toJSONString(allSettings, SerializerFeature.PrettyFormat, SerializerFeature.WriteMapNullValue, SerializerFeature.WriteDateUseDateFormat);
            buffW.write(format);
        } catch (IOException ignored) {
        }
    }

    public static boolean isDebug() {
        try {
            String res = System.getProperty("File_Engine_Debug");
            return "true".equalsIgnoreCase(res);
        } catch (NullPointerException e) {
            return false;
        }
    }

    public static boolean hasStartup() {
        try {
            String command = "cmd.exe /c schtasks /query /tn \"File-Engine\"";
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

    public enum DownloadStatus {
        DOWNLOAD_DONE, DOWNLOAD_ERROR, DOWNLOAD_DOWNLOADING, DOWNLOAD_INTERRUPTED, DOWNLOAD_NO_TASK
    }

    public static class ProxyInfo {
        public final String address;
        public final int port;
        public final String userName;
        public final String password;
        public final Proxy.Type type;

        private ProxyInfo(String proxyAddress, int proxyPort, String proxyUserName, String proxyPassword, int proxyType) {
            this.address = proxyAddress;
            this.port = proxyPort;
            this.userName = proxyUserName;
            this.password = proxyPassword;
            if (ProxyType.PROXY_HTTP == proxyType) {
                this.type = Proxy.Type.HTTP;
            } else if (ProxyType.PROXY_SOCKS == proxyType) {
                this.type = Proxy.Type.SOCKS;
            } else {
                this.type = Proxy.Type.DIRECT;
            }
        }
    }

    public static class ShowingSearchBarMode {
        public static final int NORMAL_SHOWING = 0;
        public static final int EXPLORER_ATTACH = 1;
    }

    public static class RunningMode {
        public static final int NORMAL_MODE = 2;
        public static final int COMMAND_MODE = 3;
        public static final int PLUGIN_MODE = 4;
    }

    public static class ProxyType {
        public static final int PROXY_HTTP = 0x100;
        public static final int PROXY_SOCKS = 0x200;
        public static final int PROXY_DIRECT = 0x300;
    }
}
