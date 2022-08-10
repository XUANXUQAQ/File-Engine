package file.engine.dllInterface;

import java.nio.file.Path;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

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

    public native void match(String[] searchCase,
                             boolean isIgnoreCase,
                             String searchText,
                             String[] keywords,
                             String[] keywordsLowerCase,
                             boolean[] isKeywordPath,
                             ConcurrentHashMap<String, ConcurrentLinkedQueue<String>> output);

    public boolean isCudaAvailableOnSystem() {
        if (isCudaLoaded) {
          return isCudaAvailable();
        }
        return false;
    }
    public native boolean isMatchDone(String key);

    private native boolean isCudaAvailable();

    public native boolean hasCache(String key);

    public native void initCache(String key, Object[] records);

    public native void addOneRecordToCache(String key, String record);

    public native void clearCache(String key);

    public native void clearAllCache();

    public native boolean isCacheValid(String key);

    public native int getCudaMemUsage();
}
