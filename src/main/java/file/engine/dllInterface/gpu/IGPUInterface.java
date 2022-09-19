package file.engine.dllInterface.gpu;

import java.util.function.BiConsumer;
import java.util.function.Supplier;

public interface IGPUInterface {

    void resetAllResultStatus();

    void match(String[] searchCase,
               boolean isIgnoreCase,
               String searchText,
               String[] keywords,
               String[] keywordsLowerCase,
               boolean[] isKeywordPath,
               int maxResultNumber,
               BiConsumer<String, String> resultCollector);

    boolean isGPUAvailableOnSystem();

    boolean isMatchDone(String key);

    int matchedNumber(String key);

    void stopCollectResults();

    boolean hasCache();

    boolean isCacheExist(String key);

    void initCache(String key, Supplier<String> recordSupplier);

    void addRecordsToCache(String key, Object[] records);

    void removeRecordsFromCache(String key, Object[] records);

    void clearCache(String key);

    void clearAllCache();

    boolean isCacheValid(String key);

    int getGPUMemUsage();

    void initialize();

    void release();

    String getDevices();

    boolean setDevice(int deviceNum);
}
