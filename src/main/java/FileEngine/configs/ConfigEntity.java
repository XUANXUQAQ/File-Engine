package FileEngine.configs;

import com.alibaba.fastjson.annotation.JSONField;
import lombok.Data;

public @Data class ConfigEntity {
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
}
