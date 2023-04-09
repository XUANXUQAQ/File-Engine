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

    /**
     * 之所以采用双重检验锁机制，是由于要实现懒加载，并且不能在加载类的时候进行加载
     * 由于在事务管理器扫描@EventRegister和@EventListener的阶段将会尝试加载所有类，此时配置中心还不可用
     * 因此采用getInstance()方法来实现懒加载，在BootSystem事件发出后再进行初始化。
     *
     * @param isEnableGpuAccelerate
     */
    record IsEnabledWrapper(boolean isEnableGpuAccelerate) {
        private static volatile IsEnabledWrapper instance;

        public static IsEnabledWrapper getInstance() {
            if (instance == null) {
                synchronized (IsEnabledWrapper.class) {
                    if (instance == null) {
                        instance = new IsEnabledWrapper(AllConfigs.getInstance().getConfigEntity().isEnableGpuAccelerate());
                    }
                }
            }
            return instance;
        }
    }

    public void resetAllResultStatus() {
        if (gpuAccelerator != null) {
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
        if (gpuAccelerator != null) {
            gpuAccelerator.match(searchCase, isIgnoreCase, searchText, keywords, keywordsLowerCase, isKeywordPath, maxResultNumber, resultCollector);
        }
    }

    public boolean isMatchDone(String key) {
        if (gpuAccelerator != null) {
            return gpuAccelerator.isMatchDone(key);
        }
        return false;
    }

    public int matchedNumber(String key) {
        if (gpuAccelerator != null) {
            return gpuAccelerator.matchedNumber(key);
        }
        return 0;
    }

    public void stopCollectResults() {
        if (gpuAccelerator != null) {
            gpuAccelerator.stopCollectResults();
        }
    }

    public boolean isGPUAvailableOnSystem() {
        return CudaAccelerator.INSTANCE.isGPUAvailableOnSystem();
    }

    public boolean hasCache() {
        if (gpuAccelerator != null) {
            return gpuAccelerator.hasCache();
        }
        return false;
    }

    public boolean isCacheExist(String key) {
        if (gpuAccelerator != null) {
            return gpuAccelerator.isCacheExist(key);
        }
        return false;
    }

    public void initCache(String key, Supplier<String> recordSupplier) {
        if (gpuAccelerator != null) {
            gpuAccelerator.initCache(key, recordSupplier);
        }
    }

    public void addRecordsToCache(String key, Object[] records) {
        if (gpuAccelerator != null) {
            gpuAccelerator.addRecordsToCache(key, records);
        }
    }

    public void removeRecordsFromCache(String key, Object[] records) {
        if (gpuAccelerator != null) {
            gpuAccelerator.removeRecordsFromCache(key, records);
        }
    }

    public void clearCache(String key) {
        if (gpuAccelerator != null) {
            gpuAccelerator.clearCache(key);
        }
    }

    public void clearAllCache() {
        if (gpuAccelerator != null) {
            gpuAccelerator.clearAllCache();
        }
    }

    public boolean isCacheValid(String key) {
        if (gpuAccelerator != null) {
            return gpuAccelerator.isCacheValid(key);
        }
        return false;
    }

    public int getGPUMemUsage() {
        if (gpuAccelerator != null) {
            return gpuAccelerator.getGPUMemUsage();
        }
        return 100;
    }

    public void release() {
        if (gpuAccelerator != null) {
            gpuAccelerator.release();
        }
    }

    /**
     * TODO 添加其他API
     * key: 设备名
     * value: [设备种类(cuda, opencl)];[设备id]
     *
     * @return map
     */
    public Map<String, String> getDevices() {
        LinkedHashMap<String, String> deviceMap = new LinkedHashMap<>();
        getDeviceToMap(CudaAccelerator.INSTANCE, deviceMap, GPUApiCategory.CUDA);
        return deviceMap;
    }

    private void getDeviceToMap(IGPUAccelerator igpuAccelerator, HashMap<String, String> deviceMap, GPUApiCategory category) {
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

    /**
     * TODO 添加其他API
     *
     * @param deviceCategoryAndId 设备类型及id，如cuda;0
     * @return 是否设置成功
     */
    public boolean setDevice(String deviceCategoryAndId) {
        if (gpuAccelerator != null) {
            // 切换GPU设备重启生效，运行中不允许切换
            return true;
        }
        if (!IsEnabledWrapper.getInstance().isEnableGpuAccelerate()) {
            return false;
        }
        if (deviceCategoryAndId == null || deviceCategoryAndId.isEmpty()) {
            if (CudaAccelerator.INSTANCE.isGPUAvailableOnSystem()) {
                CudaAccelerator.INSTANCE.initialize();
                if (CudaAccelerator.INSTANCE.setDevice(0)) {
                    gpuAccelerator = CudaAccelerator.INSTANCE;
                    return true;
                }
            }
            return false;
        }
        String[] info = RegexUtil.semicolon.split(deviceCategoryAndId);
        String deviceCategory = info[0];
        int id = Integer.parseInt(info[1]);
        var category = GPUApiCategory.categoryFromString(deviceCategory);
        if (category != null) {
            switch (category) {
                case CUDA:
                    if (CudaAccelerator.INSTANCE.isGPUAvailableOnSystem()) {
                        CudaAccelerator.INSTANCE.initialize();
                        if (CudaAccelerator.INSTANCE.setDevice(id)) {
                            gpuAccelerator = CudaAccelerator.INSTANCE;
                            return true;
                        }
                    }
            }
        }
        return false;
    }

    @SuppressWarnings("unused")
    public static void sendRestartOnError0() {
        System.err.println("GPU缓存出错，自动重启");
        EventManagement.getInstance().putEvent(new RestartEvent());
    }
}
