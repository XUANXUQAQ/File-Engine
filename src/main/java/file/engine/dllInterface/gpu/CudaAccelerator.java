package file.engine.dllInterface.gpu;

import java.nio.file.Path;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

enum CudaAccelerator implements IGPUAccelerator {
    INSTANCE;

    private volatile boolean isCudaLoaded;

    CudaAccelerator() {
        try {
            System.load(Path.of("user/cudaAccelerator.dll").toAbsolutePath().toString());
            isCudaLoaded = true;
        } catch (UnsatisfiedLinkError | Exception e) {
            isCudaLoaded = false;
        }
    }

    public native void resetAllResultStatus();

    public native void match(String[] searchCase,
                             boolean isIgnoreCase,
                             String searchText,
                             String[] keywords,
                             String[] keywordsLowerCase,
                             boolean[] isKeywordPath,
                             int maxResultNumber,
                             int resultCollectThreadNum,
                             BiConsumer<String, String> resultCollector);

    public boolean isGPUAvailableOnSystem() {
        if (isCudaLoaded) {
            return isCudaAvailable();
        }
        return false;
    }

    private native boolean isCudaAvailable();

    public native boolean isMatchDone(String key);

    public native int matchedNumber(String key);

    public native void stopCollectResults();

    public native boolean hasCache();

    public native boolean isCacheExist(String key);

    public native void initCache(String key, Supplier<String> recordSupplier);

    public native void addRecordsToCache(String key, Object[] records);

    public native void removeRecordsFromCache(String key, Object[] records);

    public native void clearCache(String key);

    public native void clearAllCache();

    public native boolean isCacheValid(String key);

    public native int getGPUMemUsage();

    public native void initialize();

    public native void release();

    public native String[] getDevices();

    public native boolean setDevice(int deviceNum);
}
