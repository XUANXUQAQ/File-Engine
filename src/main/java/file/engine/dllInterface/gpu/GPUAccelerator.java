package file.engine.dllInterface.gpu;

import file.engine.configs.AllConfigs;
import file.engine.utils.RegexUtil;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

public enum GPUAccelerator {
    INSTANCE;
    private static IGPUAccelerator gpuAccelerator;
    private static final CudaAccelerator cudaAccelerator = CudaAccelerator.INSTANCE;
    private static final OpenclAccelerator openclAccelerator = OpenclAccelerator.INSTANCE;

    enum Category {
        CUDA("cuda"), OPENCL("opencl");
        final String category;

        Category(String category) {
            this.category = category;
        }

        @Override
        public String toString() {
            return this.category;
        }

        static Category categoryFromString(String c) {
            switch (c) {
                case "cuda":
                    return CUDA;
                case "opencl":
                    return OPENCL;
                default:
                    return null;
            }
        }
    }

    public void resetAllResultStatus() {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            gpuAccelerator.resetAllResultStatus();
        }
    }

    public void match(String[] searchCase,
                      boolean isIgnoreCase,
                      String searchText,
                      String[] keywords,
                      String[] keywordsLowerCase,
                      boolean[] isKeywordPath,
                      int maxResultNumber,
                      BiConsumer<String, String> resultCollector) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            gpuAccelerator.match(searchCase, isIgnoreCase, searchText, keywords, keywordsLowerCase, isKeywordPath, maxResultNumber, resultCollector);
        }
    }

    public boolean isMatchDone(String key) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            return gpuAccelerator.isMatchDone(key);
        }
        return false;
    }

    public int matchedNumber(String key) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            return gpuAccelerator.matchedNumber(key);
        }
        return 0;
    }

    public void stopCollectResults() {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            gpuAccelerator.stopCollectResults();
        }
    }

    public boolean isGPUAvailableOnSystem() {
        return cudaAccelerator.isGPUAvailableOnSystem() || openclAccelerator.isGPUAvailableOnSystem();
    }

    public boolean hasCache() {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            return gpuAccelerator.hasCache();
        }
        return false;
    }

    public boolean isCacheExist(String key) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            return gpuAccelerator.isCacheExist(key);
        }
        return false;
    }

    public void initCache(String key, Supplier<String> recordSupplier) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            gpuAccelerator.initCache(key, recordSupplier);
        }
    }

    public void addRecordsToCache(String key, Object[] records) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            gpuAccelerator.addRecordsToCache(key, records);
        }
    }

    public void removeRecordsFromCache(String key, Object[] records) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            gpuAccelerator.removeRecordsFromCache(key, records);
        }
    }

    public void clearCache(String key) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            gpuAccelerator.clearCache(key);
        }
    }

    public void clearAllCache() {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            gpuAccelerator.clearAllCache();
        }
    }

    public boolean isCacheValid(String key) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            return gpuAccelerator.isCacheValid(key);
        }
        return false;
    }

    public int getGPUMemUsage() {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            return gpuAccelerator.getGPUMemUsage();
        }
        return 0;
    }

    public void initialize() {
        if (AllConfigs.getInstance().isEnableGpuAccelerate()) {
            if (cudaAccelerator.isGPUAvailableOnSystem()) {
                cudaAccelerator.initialize();
            }
            if (openclAccelerator.isGPUAvailableOnSystem()) {
                openclAccelerator.initialize();
            }
        }
    }

    public void release() {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            gpuAccelerator.release();
        }
    }

    /**
     * key: 设备名
     * value: [设备种类(cuda, opencl)];[设备id]
     *
     * @return map
     */
    public Map<String, String> getDevices() {
        LinkedHashMap<String, String> deviceMap = new LinkedHashMap<>();
        getDeviceToMap(cudaAccelerator, deviceMap, Category.CUDA);
        getDeviceToMap(openclAccelerator, deviceMap, Category.OPENCL);
        return deviceMap;
    }

    private void getDeviceToMap(IGPUAccelerator igpuAccelerator, HashMap<String, String> deviceMap, Category category) {
        if (igpuAccelerator.isGPUAvailableOnSystem()) {
            String devices = igpuAccelerator.getDevices();
            if (devices.isBlank())
                return;
            String[] deviceInfo = RegexUtil.semicolon.split(devices);
            if (deviceInfo == null || deviceInfo.length == 0) {
                return;
            }
            for (var eachDeviceInfo : deviceInfo) {
                if (eachDeviceInfo.isBlank())
                    continue;
                String[] nameAndId = RegexUtil.comma.split(eachDeviceInfo);
                if (null == nameAndId || nameAndId.length != 2)
                    continue;
                try {
                    String deviceName = nameAndId[0];
                    int deviceId = Integer.parseInt(nameAndId[1]);
                    if (!deviceMap.containsKey(deviceName)) {
                        deviceMap.put(deviceName, category + ";" + deviceId);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public boolean setDevice(String deviceCategoryAndId) {
        if (gpuAccelerator != null && AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            // 切换GPU设备重启生效，运行中不允许切换
            return true;
        }
        if (deviceCategoryAndId.isEmpty()) {
            if (cudaAccelerator.isGPUAvailableOnSystem() && cudaAccelerator.setDevice(0)) {
                gpuAccelerator = cudaAccelerator;
                return true;
            }
            if (openclAccelerator.isGPUAvailableOnSystem() && openclAccelerator.setDevice(0)) {
                gpuAccelerator = openclAccelerator;
                return true;
            }
            return false;
        }
        String[] info = RegexUtil.semicolon.split(deviceCategoryAndId);
        String deviceCategory = info[0];
        int id = Integer.parseInt(info[1]);
        var category = Category.categoryFromString(deviceCategory);
        if (category != null) {
            switch (category) {
                case CUDA:
                    if (cudaAccelerator.isGPUAvailableOnSystem() && cudaAccelerator.setDevice(id)) {
                        gpuAccelerator = cudaAccelerator;
                        return true;
                    }
                case OPENCL:
                    if (openclAccelerator.isGPUAvailableOnSystem() && openclAccelerator.setDevice(id)) {
                        gpuAccelerator = openclAccelerator;
                        return true;
                    }
            }
        }
        return false;
    }
}
