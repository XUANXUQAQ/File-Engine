package file.engine.dllInterface;

import java.nio.file.Path;

public enum CudaAccelerator {
    INSTANCE;

    private static boolean isCudaLoaded;

    static {
        try {
            System.load(Path.of("user/cudaAccelerator.dll").toAbsolutePath().toString());
            isCudaLoaded = true;
        } catch (UnsatisfiedLinkError | Exception e) {
            e.printStackTrace();
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
                             int maxResultNumber);

    public native String getOneResult(String key);

    public boolean isCudaAvailableOnSystem() {
        if (isCudaLoaded) {
          return isCudaAvailable();
        }
        return false;
    }
    public native boolean isMatchDone(String key);

    public native void stopCollectResults();

    private native boolean isCudaAvailable();

    public native boolean hasCache(String key);

    public native void initCache(String key, Object[] records);

    public native void addOneRecordToCache(String key, String record);

    public native void removeOneRecordFromCache(String key, String record);

    public native void clearCache(String key);

    public native void clearAllCache();

    public native boolean isCacheValid(String key);

    public native int getCudaMemUsage();

    public native void initialize();

    public native void release();
}
