package file.engine.dllInterface.gpu;

import file.engine.configs.AllConfigs;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.stop.RestartEvent;
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
    private static boolean isEnableGpuAccelerate = false;

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
            return switch (c) {
                case "cuda" -> CUDA;
                case "opencl" -> OPENCL;
                default -> null;
            };
        }
    }

    public void resetAllResultStatus() {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
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
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            gpuAccelerator.match(searchCase, isIgnoreCase, searchText, keywords, keywordsLowerCase, isKeywordPath, maxResultNumber, resultCollector);
        }
    }

    public boolean isMatchDone(String key) {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            return gpuAccelerator.isMatchDone(key);
        }
        return false;
    }

    public int matchedNumber(String key) {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            return gpuAccelerator.matchedNumber(key);
        }
        return 0;
    }

    public void stopCollectResults() {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            gpuAccelerator.stopCollectResults();
        }
    }

    public boolean isGPUAvailableOnSystem() {
        return cudaAccelerator.isGPUAvailableOnSystem() || openclAccelerator.isGPUAvailableOnSystem();
    }

    public boolean hasCache() {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            return gpuAccelerator.hasCache();
        }
        return false;
    }

    public boolean isCacheExist(String key) {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            return gpuAccelerator.isCacheExist(key);
        }
        return false;
    }

    public void initCache(String key, Supplier<String> recordSupplier) {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            gpuAccelerator.initCache(key, recordSupplier);
        }
    }

    public void addRecordsToCache(String key, Object[] records) {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            gpuAccelerator.addRecordsToCache(key, records);
        }
    }

    public void removeRecordsFromCache(String key, Object[] records) {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            gpuAccelerator.removeRecordsFromCache(key, records);
        }
    }

    public void clearCache(String key) {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            gpuAccelerator.clearCache(key);
        }
    }

    public void clearAllCache() {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            gpuAccelerator.clearAllCache();
        }
    }

    public boolean isCacheValid(String key) {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            return gpuAccelerator.isCacheValid(key);
        }
        return false;
    }

    public int getGPUMemUsage() {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
            return gpuAccelerator.getGPUMemUsage();
        }
        return 0;
    }

    public void initialize() {
        isEnableGpuAccelerate = AllConfigs.getInstance().getConfigEntity().isEnableGpuAccelerate();
        if (isEnableGpuAccelerate) {
            if (cudaAccelerator.isGPUAvailableOnSystem()) {
                cudaAccelerator.initialize();
            }
            if (openclAccelerator.isGPUAvailableOnSystem()) {
                openclAccelerator.initialize();
            }
        }
    }

    public void release() {
        if (gpuAccelerator != null && isEnableGpuAccelerate) {
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
            var devices = igpuAccelerator.getDevices();
            if (devices == null || devices.length == 0) {
                return;
            }
            for (int i = 0; i < devices.length; ++i) {
                var deviceName = devices[i];
                if (deviceName.isBlank()) {
                    continue;
                }
                try {
                    if (!deviceMap.containsKey(deviceName)) {
                        deviceMap.put(deviceName, category + ";" + i);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public boolean setDevice(String deviceCategoryAndId) {
        if (gpuAccelerator != null) {
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

    @SuppressWarnings("unused")
    public static void sendRestartOnError0() {
        EventManagement.getInstance().putEvent(new RestartEvent());
    }
}
