package file.engine.dllInterface.gpu;

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
        checkAvailable();
        gpuAccelerator.resetAllResultStatus();
    }

    public void match(String[] searchCase,
                      boolean isIgnoreCase,
                      String searchText,
                      String[] keywords,
                      String[] keywordsLowerCase,
                      boolean[] isKeywordPath,
                      int maxResultNumber,
                      BiConsumer<String, String> resultCollector) {
        checkAvailable();
        gpuAccelerator.match(searchCase, isIgnoreCase, searchText, keywords, keywordsLowerCase, isKeywordPath, maxResultNumber, resultCollector);
    }

    public boolean isMatchDone(String key) {
        checkAvailable();
        return gpuAccelerator.isMatchDone(key);
    }

    public int matchedNumber(String key) {
        checkAvailable();
        return gpuAccelerator.matchedNumber(key);
    }

    public void stopCollectResults() {
        checkAvailable();
        gpuAccelerator.stopCollectResults();
    }

    public boolean isGPUAvailableOnSystem() {
        return cudaAccelerator.isGPUAvailableOnSystem() || openclAccelerator.isGPUAvailableOnSystem();
    }

    public boolean hasCache() {
        checkAvailable();
        return gpuAccelerator.hasCache();
    }

    public boolean isCacheExist(String key) {
        checkAvailable();
        return gpuAccelerator.isCacheExist(key);
    }

    public void initCache(String key, Supplier<String> recordSupplier) {
        checkAvailable();
        gpuAccelerator.initCache(key, recordSupplier);
    }

    public void addRecordsToCache(String key, Object[] records) {
        checkAvailable();
        gpuAccelerator.addRecordsToCache(key, records);
    }

    public void removeRecordsFromCache(String key, Object[] records) {
        checkAvailable();
        gpuAccelerator.removeRecordsFromCache(key, records);
    }

    public void clearCache(String key) {
        checkAvailable();
        gpuAccelerator.clearCache(key);
    }

    public void clearAllCache() {
        checkAvailable();
        gpuAccelerator.clearAllCache();
    }

    public boolean isCacheValid(String key) {
        checkAvailable();
        return gpuAccelerator.isCacheValid(key);
    }

    public int getGPUMemUsage() {
        checkAvailable();
        return gpuAccelerator.getGPUMemUsage();
    }

    public void initialize() {
        if (cudaAccelerator.isGPUAvailableOnSystem()) {
            cudaAccelerator.initialize();
        }
        if (openclAccelerator.isGPUAvailableOnSystem()) {
            openclAccelerator.initialize();
        }
    }

    public void release() {
        checkAvailable();
        gpuAccelerator.release();
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
            String[] deviceInfo = RegexUtil.semicolon.split(devices);
            if (deviceInfo != null && deviceInfo.length != 0) {
                for (var eachDeviceInfo : deviceInfo) {
                    String[] nameAndId = RegexUtil.comma.split(eachDeviceInfo);
                    String deviceName = nameAndId[0];
                    int deviceId = Integer.parseInt(nameAndId[1]);
                    if (!deviceMap.containsKey(deviceName)) {
                        deviceMap.put(deviceName, category + ";" + deviceId);
                    }
                }
            }
        }
    }

    public boolean setDevice(String deviceCategoryAndId) {
        if (gpuAccelerator != null) {
            // 切换GPU设备重启生效，运行中不允许切换
            return true;
        }
        // FIXME: 测试OpenCL注释代码
        if (deviceCategoryAndId.isEmpty()) {
//            if (cudaAccelerator.isGPUAvailableOnSystem() && cudaAccelerator.setDevice(0)) {
//                gpuAccelerator = cudaAccelerator;
//                return true;
//            }
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
//                case CUDA:
//                    if (cudaAccelerator.isGPUAvailableOnSystem() && cudaAccelerator.setDevice(id)) {
//                        gpuAccelerator = cudaAccelerator;
//                        return true;
//                    }
                case OPENCL:
                    if (openclAccelerator.isGPUAvailableOnSystem() && openclAccelerator.setDevice(id)) {
                        gpuAccelerator = openclAccelerator;
                        return true;
                    }
            }
        }
        return false;
    }

    private void checkAvailable() {
        if (gpuAccelerator == null) {
            throw new RuntimeException("gpu accelerate not available");
        }
    }
}
