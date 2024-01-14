package file.engine.configs.core;

import com.google.gson.annotations.SerializedName;
import lombok.Data;

@Data
public class CoreConfigEntity {
    @SerializedName("cacheNumLimit")
    private int cacheNumLimit;

    @SerializedName("updateTimeLimit")
    private int updateTimeLimit;

    @SerializedName("ignorePath")
    private String ignorePath;

    @SerializedName("priorityFolder")
    private String priorityFolder;

    @SerializedName("disks")
    private String disks;

    @SerializedName("isEnableGpuAccelerate")
    private boolean isEnableGpuAccelerate;

    @SerializedName("gpuDevice")
    private String gpuDevice;

    @SerializedName("searchThreadNumber")
    private int searchThreadNumber;

    @SerializedName("port")
    private int port;

    @SerializedName("advancedConfigs")
    private CoreAdvancedConfigEntity advancedConfigEntity;
}
