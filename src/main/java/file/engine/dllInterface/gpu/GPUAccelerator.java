package file.engine.dllInterface.gpu;

import java.util.function.BiConsumer;
import java.util.function.Supplier;

public enum GPUAccelerator implements IGPUInterface {
    INSTANCE;
    private static IGPUInterface gpuAccelerator = null;

    static {
        if (CudaAccelerator.INSTANCE.isGPUAvailableOnSystem()) {
            gpuAccelerator = CudaAccelerator.INSTANCE;
        } else if (OpenclAccelerator.INSTANCE.isGPUAvailableOnSystem()) {
            gpuAccelerator = OpenclAccelerator.INSTANCE;
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
        if (gpuAccelerator == null)
            return false;
        return gpuAccelerator.isGPUAvailableOnSystem();
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
        checkAvailable();
        gpuAccelerator.initialize();
    }

    public void release() {
        checkAvailable();
        gpuAccelerator.release();
    }

    public String getDevices() {
        checkAvailable();
        return gpuAccelerator.getDevices();
    }

    public boolean setDevice(int deviceNum) {
        checkAvailable();
        return gpuAccelerator.setDevice(deviceNum);
    }

    private void checkAvailable() {
        if (gpuAccelerator == null) {
            throw new RuntimeException("gpu accelerate not available");
        }
    }
}
