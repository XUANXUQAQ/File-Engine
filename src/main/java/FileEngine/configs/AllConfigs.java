package FileEngine.configs;

import FileEngine.modesAndStatus.Enums;
import FileEngine.translate.TranslateUtil;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SerializerFeature;

import java.io.*;
import java.net.Proxy;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashSet;

/**
 * 保存软件运行时的所有配置信息
 */
public class AllConfigs {
    public static final String version = "2.7"; //TODO 更改版本号

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
    private static String updateAddress;

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
        AllConfigs.cacheNumLimit = cacheNumLimit;
    }

    public static void setHotkey(String hotkey) {
        AllConfigs.hotkey = hotkey;
    }

    public static void setUpdateTimeLimit(int updateTimeLimit) {
        AllConfigs.updateTimeLimit = updateTimeLimit;
    }

    public static void setIgnorePath(String ignorePath) {
        AllConfigs.ignorePath = ignorePath;
    }

    public static void setPriorityFolder(String priorityFolder) {
        AllConfigs.priorityFolder = priorityFolder;
    }

    public static void setSearchDepth(int searchDepth) {
        AllConfigs.searchDepth = searchDepth;
    }

    public static void setIsDefaultAdmin(boolean isDefaultAdmin) {
        AllConfigs.isDefaultAdmin = isDefaultAdmin;
    }

    public static void setIsLoseFocusClose(boolean isLoseFocusClose) {
        AllConfigs.isLoseFocusClose = isLoseFocusClose;
    }

    public static void setOpenLastFolderKeyCode(int openLastFolderKeyCode) {
        AllConfigs.openLastFolderKeyCode = openLastFolderKeyCode;
    }

    public static void setRunAsAdminKeyCode(int runAsAdminKeyCode) {
        AllConfigs.runAsAdminKeyCode = runAsAdminKeyCode;
    }

    public static void setCopyPathKeyCode(int copyPathKeyCode) {
        AllConfigs.copyPathKeyCode = copyPathKeyCode;
    }

    public static void setTransparency(float transparency) {
        AllConfigs.transparency = transparency;
    }

    public static void setProxyAddress(String proxyAddress) {
        AllConfigs.proxyAddress = proxyAddress;
    }

    public static void setProxyPort(int proxyPort) {
        AllConfigs.proxyPort = proxyPort;
    }

    public static void setProxyUserName(String proxyUserName) {
        AllConfigs.proxyUserName = proxyUserName;
    }

    public static void setProxyPassword(String proxyPassword) {
        AllConfigs.proxyPassword = proxyPassword;
    }

    public static void setProxyType(int proxyType) {
        AllConfigs.proxyType = proxyType;
    }

    public static void setLabelColor(int labelColor) {
        AllConfigs.labelColor = labelColor;
    }

    public static void setDefaultBackgroundColor(int defaultBackgroundColor) {
        AllConfigs.defaultBackgroundColor = defaultBackgroundColor;
    }

    public static void setFontColorWithCoverage(int fontColorWithCoverage) {
        AllConfigs.fontColorWithCoverage = fontColorWithCoverage;
    }

    public static void setFontColor(int fontColor) {
        AllConfigs.fontColor = fontColor;
    }

    public static int getSearchBarColor() {
        return searchBarColor;
    }

    public static void setSearchBarColor(int searchBarColor) {
        AllConfigs.searchBarColor = searchBarColor;
    }

    public static void setUpdateAddress(String updateAddress) {
        AllConfigs.updateAddress = updateAddress;
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

    public static int getFontColorWithCoverage() {
        return fontColorWithCoverage;
    }

    public static int getFontColor() {
        return fontColor;
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
            if (Enums.ProxyType.PROXY_HTTP == proxyType) {
                this.type = Proxy.Type.HTTP;
            } else if (Enums.ProxyType.PROXY_SOCKS == proxyType) {
                this.type = Proxy.Type.SOCKS;
            } else {
                this.type = Proxy.Type.DIRECT;
            }
        }
    }
}
