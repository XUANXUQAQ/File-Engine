package FileEngine.configs;

import com.alibaba.fastjson.annotation.JSONField;

public class ConfigEntity {
    @JSONField(name = "cacheNumLimit")
    private int cacheNumLimit;

    @JSONField(name = "isShowTipOnCreatingLnk")
    private boolean isShowTipCreatingLnk;

    @JSONField(name = "hotkey")
    private String hotkey;

    @JSONField(name = "updateTimeLimit")
    private int updateTimeLimit;

    @JSONField(name = "ignorePath")
    private String ignorePath;

    @JSONField(name = "priorityFolder")
    private String priorityFolder;

    @JSONField(name = "searchDepth")
    private int searchDepth;

    @JSONField(name = "isDefaultAdmin")
    private boolean isDefaultAdmin;

    @JSONField(name = "isLoseFocusClose")
    private boolean isLoseFocusClose;

    @JSONField(name = "openLastFolderKeyCode")
    private int openLastFolderKeyCode;

    @JSONField(name = "runAsAdminKeyCode")
    private int runAsAdminKeyCode;

    @JSONField(name = "copyPathKeyCode")
    private int copyPathKeyCode;

    @JSONField(name = "transparency")
    private float transparency;

    @JSONField(name = "proxyAddress")
    private String proxyAddress;

    @JSONField(name = "proxyPort")
    private int proxyPort;

    @JSONField(name = "proxyUserName")
    private String proxyUserName;

    @JSONField(name = "proxyPassword")
    private String proxyPassword;

    @JSONField(name = "proxyType")
    private int proxyType;

    @JSONField(name = "labelColor")
    private int labelColor;

    @JSONField(name = "defaultBackground")
    private int defaultBackgroundColor;

    @JSONField(name = "fontColorWithCoverage")
    private int fontColorWithCoverage;

    @JSONField(name = "fontColor")
    private int fontColor;

    @JSONField(name = "searchBarColor")
    private int searchBarColor;

    @JSONField(name = "searchBarFontColor")
    private int searchBarFontColor;

    @JSONField(name = "borderColor")
    private int borderColor;

    @JSONField(name = "updateAddress")
    private String updateAddress;

    @JSONField(name = "swingTheme")
    private String swingTheme;

    @JSONField(name = "language")
    private String language;

    public String getLanguage() {
        return language;
    }

    public int getCacheNumLimit() {
        return cacheNumLimit;
    }

    public boolean isShowTipCreatingLnk() {
        return isShowTipCreatingLnk;
    }

    public String getHotkey() {
        return hotkey;
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

    public String getProxyAddress() {
        return proxyAddress;
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

    public int getLabelColor() {
        return labelColor;
    }

    public int getDefaultBackgroundColor() {
        return defaultBackgroundColor;
    }

    public int getFontColorWithCoverage() {
        return fontColorWithCoverage;
    }

    public int getFontColor() {
        return fontColor;
    }

    public int getSearchBarColor() {
        return searchBarColor;
    }

    public int getSearchBarFontColor() {
        return searchBarFontColor;
    }

    public int getBorderColor() {
        return borderColor;
    }

    public String getUpdateAddress() {
        return updateAddress;
    }

    public void setLanguage(String language) {
        this.language = language;
    }

    public String getSwingTheme() {
        return swingTheme;
    }

    public void setSwingTheme(String swingTheme) {
        this.swingTheme = swingTheme;
    }

    public void setCacheNumLimit(int cacheNumLimit) {
        this.cacheNumLimit = cacheNumLimit;
    }

    public void setShowTipCreatingLnk(boolean showTipCreatingLnk) {
        isShowTipCreatingLnk = showTipCreatingLnk;
    }

    public void setHotkey(String hotkey) {
        this.hotkey = hotkey;
    }

    public void setUpdateTimeLimit(int updateTimeLimit) {
        this.updateTimeLimit = updateTimeLimit;
    }

    public void setIgnorePath(String ignorePath) {
        this.ignorePath = ignorePath;
    }

    public void setPriorityFolder(String priorityFolder) {
        this.priorityFolder = priorityFolder;
    }

    public void setSearchDepth(int searchDepth) {
        this.searchDepth = searchDepth;
    }

    public void setDefaultAdmin(boolean defaultAdmin) {
        isDefaultAdmin = defaultAdmin;
    }

    public void setLoseFocusClose(boolean loseFocusClose) {
        isLoseFocusClose = loseFocusClose;
    }

    public void setOpenLastFolderKeyCode(int openLastFolderKeyCode) {
        this.openLastFolderKeyCode = openLastFolderKeyCode;
    }

    public void setRunAsAdminKeyCode(int runAsAdminKeyCode) {
        this.runAsAdminKeyCode = runAsAdminKeyCode;
    }

    public void setCopyPathKeyCode(int copyPathKeyCode) {
        this.copyPathKeyCode = copyPathKeyCode;
    }

    public void setTransparency(float transparency) {
        this.transparency = transparency;
    }

    public void setProxyAddress(String proxyAddress) {
        this.proxyAddress = proxyAddress;
    }

    public void setProxyPort(int proxyPort) {
        this.proxyPort = proxyPort;
    }

    public void setProxyUserName(String proxyUserName) {
        this.proxyUserName = proxyUserName;
    }

    public void setProxyPassword(String proxyPassword) {
        this.proxyPassword = proxyPassword;
    }

    public void setProxyType(int proxyType) {
        this.proxyType = proxyType;
    }

    public void setLabelColor(int labelColor) {
        this.labelColor = labelColor;
    }

    public void setDefaultBackgroundColor(int defaultBackgroundColor) {
        this.defaultBackgroundColor = defaultBackgroundColor;
    }

    public void setFontColorWithCoverage(int fontColorWithCoverage) {
        this.fontColorWithCoverage = fontColorWithCoverage;
    }

    public void setFontColor(int fontColor) {
        this.fontColor = fontColor;
    }

    public void setSearchBarColor(int searchBarColor) {
        this.searchBarColor = searchBarColor;
    }

    public void setSearchBarFontColor(int searchBarFontColor) {
        this.searchBarFontColor = searchBarFontColor;
    }

    public void setBorderColor(int borderColor) {
        this.borderColor = borderColor;
    }

    public void setUpdateAddress(String updateAddress) {
        this.updateAddress = updateAddress;
    }
}
