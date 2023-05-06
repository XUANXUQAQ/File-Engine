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

    /**
     * 清除搜索状态，为下一次搜索做准备
     */
    public void resetAllResultStatus() {
        if (gpuAccelerator != null) {
            gpuAccelerator.resetAllResultStatus();
        }
    }

    /**
     * 开始搜索
     *
     * @param searchCase        搜索条件
     * @param isIgnoreCase      是否忽略大小写
     * @param searchText        搜索关键字（全字匹配）
     * @param keywords          搜索关键字
     * @param keywordsLowerCase 搜索关键字（小写字母）防止重复计算
     * @param isKeywordPath     与上方的搜索关键字一一对应，记录关键字是文件名关键字还是文件路径关键字
     * @param maxResultNumber   最多匹配数量
     * @param resultCollector   匹配回调函数
     * @see file.engine.services.utils.PathMatchUtil
     */
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

    /**
     * 判断某个缓存是否搜索完成
     *
     * @param key 缓存key
     * @return true如果搜索完成
     */
    public boolean isMatchDone(String key) {
        if (gpuAccelerator != null) {
            return gpuAccelerator.isMatchDone(key);
        }
        return false;
    }

    /**
     * 缓存搜索成功匹配关键字的数量
     *
     * @param key 缓存key
     * @return int
     */
    public int matchedNumber(String key) {
        if (gpuAccelerator != null) {
            return gpuAccelerator.matchedNumber(key);
        }
        return 0;
    }

    /**
     * 手动停止搜索
     */
    public void stopCollectResults() {
        if (gpuAccelerator != null) {
            gpuAccelerator.stopCollectResults();
        }
    }

    /**
     * GPU加速是否可用
     *
     * @return true如果GPU加速可用，CUDA或者OpenCL
     */
    public boolean isGPUAvailableOnSystem() {
        return CudaAccelerator.INSTANCE.isGPUAvailableOnSystem() || OpenclAccelerator.INSTANCE.isGPUAvailableOnSystem();
    }

    /**
     * 是否存在缓存
     *
     * @return true如果有一个以上缓存
     */
    public boolean hasCache() {
        if (gpuAccelerator != null) {
            return gpuAccelerator.hasCache();
        }
        return false;
    }

    /**
     * 缓存是否存在
     *
     * @param key 缓存key
     * @return true如果存在
     */
    public boolean isCacheExist(String key) {
        if (gpuAccelerator != null) {
            return gpuAccelerator.isCacheExist(key);
        }
        return false;
    }

    /**
     * 添加一个缓存
     *
     * @param key            缓存key
     * @param recordSupplier supplier
     */
    public void initCache(String key, Supplier<String> recordSupplier) {
        if (gpuAccelerator != null) {
            gpuAccelerator.initCache(key, recordSupplier);
        }
    }

    /**
     * 向缓存添加记录
     *
     * @param key     缓存key
     * @param records 需要添加的记录，类型为String[]
     */
    public void addRecordsToCache(String key, Object[] records) {
        if (gpuAccelerator != null) {
            gpuAccelerator.addRecordsToCache(key, records);
        }
    }

    /**
     * 从缓存中删除记录
     *
     * @param key     缓存key
     * @param records 需要删除的记录
     */
    public void removeRecordsFromCache(String key, Object[] records) {
        if (gpuAccelerator != null) {
            gpuAccelerator.removeRecordsFromCache(key, records);
        }
    }

    /**
     * 删除一个缓存
     *
     * @param key 缓存key
     */
    public void clearCache(String key) {
        if (gpuAccelerator != null) {
            gpuAccelerator.clearCache(key);
        }
    }

    /**
     * 删除所有缓存
     */
    public void clearAllCache() {
        if (gpuAccelerator != null) {
            gpuAccelerator.clearAllCache();
        }
    }

    /**
     * 缓存是否有效，当addRecordsToCache函数调用，发现缓存已经无法再添加更多记录，则会使缓存失效
     *
     * @param key 缓存key
     * @return true如果缓存有效，否则返回false
     */
    public boolean isCacheValid(String key) {
        if (gpuAccelerator != null) {
            return gpuAccelerator.isCacheValid(key);
        }
        return false;
    }

    /**
     * 获取显存占用状态
     *
     * @return 0-100的值，对应百分比
     */
    public int getGPUMemUsage() {
        if (gpuAccelerator != null) {
            return gpuAccelerator.getGPUMemUsage();
        }
        return 100;
    }

    /**
     * 释放所有缓存以及预分配的内存，用于最后退出程序
     */
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
        getDeviceToMap(OpenclAccelerator.INSTANCE, deviceMap, GPUApiCategory.OPENCL);
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
            if (OpenclAccelerator.INSTANCE.isGPUAvailableOnSystem()) {
                OpenclAccelerator.INSTANCE.initialize();
                if (OpenclAccelerator.INSTANCE.setDevice(0)) {
                    gpuAccelerator = OpenclAccelerator.INSTANCE;
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
                case CUDA -> {
                    if (CudaAccelerator.INSTANCE.isGPUAvailableOnSystem()) {
                        CudaAccelerator.INSTANCE.initialize();
                        if (CudaAccelerator.INSTANCE.setDevice(id)) {
                            gpuAccelerator = CudaAccelerator.INSTANCE;
                            return true;
                        }
                    }
                }
                case OPENCL -> {
                    if (OpenclAccelerator.INSTANCE.isGPUAvailableOnSystem()) {
                        OpenclAccelerator.INSTANCE.initialize();
                        if (OpenclAccelerator.INSTANCE.setDevice(id)) {
                            gpuAccelerator = OpenclAccelerator.INSTANCE;
                            return true;
                        }
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
