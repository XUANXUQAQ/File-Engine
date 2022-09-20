package file.engine.configs;

import com.google.gson.annotations.SerializedName;
import lombok.Data;

@Data
public class ConfigEntity {
    @SerializedName("cacheNumLimit")
    private int cacheNumLimit;

    @SerializedName("isShowTipOnCreatingLnk")
    private boolean isShowTipCreatingLnk;

    @SerializedName("hotkey")
    private String hotkey;

    @SerializedName("updateTimeLimit")
    private int updateTimeLimit;

    @SerializedName("ignorePath")
    private String ignorePath;

    @SerializedName("priorityFolder")
    private String priorityFolder;

    @SerializedName("isDefaultAdmin")
    private boolean isDefaultAdmin;

    @SerializedName("isLoseFocusClose")
    private boolean isLoseFocusClose;

    @SerializedName("openLastFolderKeyCode")
    private int openLastFolderKeyCode;

    @SerializedName("runAsAdminKeyCode")
    private int runAsAdminKeyCode;

    @SerializedName("copyPathKeyCode")
    private int copyPathKeyCode;

    @SerializedName("transparency")
    private float transparency;

    @SerializedName("proxyAddress")
    private String proxyAddress;

    @SerializedName("proxyPort")
    private int proxyPort;

    @SerializedName("proxyUserName")
    private String proxyUserName;

    @SerializedName("proxyPassword")
    private String proxyPassword;

    @SerializedName("proxyType")
    private int proxyType;

    @SerializedName("labelColor")
    private int labelColor;

    @SerializedName("defaultBackground")
    private int defaultBackgroundColor;

    @SerializedName("fontColorWithCoverage")
    private int fontColorWithCoverage;

    @SerializedName("fontColor")
    private int fontColor;

    @SerializedName("searchBarColor")
    private int searchBarColor;

    @SerializedName("searchBarFontColor")
    private int searchBarFontColor;

    @SerializedName("borderColor")
    private int borderColor;

    @SerializedName("updateAddress")
    private String updateAddress;

    @SerializedName("swingTheme")
    private String swingTheme;

    @SerializedName("language")
    private String language;

    @SerializedName("doubleClickCtrlOpen")
    private boolean doubleClickCtrlOpen;

    @SerializedName("disks")
    private String disks;

    @SerializedName("isCheckUpdateStartup")
    private boolean isCheckUpdateStartup;

    @SerializedName("borderType")
    private String borderType;

    @SerializedName("borderThickness")
    private int borderThickness;

    @SerializedName("roundRadius")
    private double roundRadius;

    @SerializedName("isAttachExplorer")
    private boolean isAttachExplorer;

    @SerializedName("isEnableCuda")
    private boolean isEnableCuda;

    @SerializedName("gpuDevice")
    private String gpuDevice;

    @SerializedName("searchThreadNumber")
    private int searchThreadNumber;
}
