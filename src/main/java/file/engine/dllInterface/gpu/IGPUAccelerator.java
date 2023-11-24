package file.engine.dllInterface.gpu;

import java.util.function.BiConsumer;
import java.util.function.Supplier;

public interface IGPUAccelerator {

    /**
     * 重置搜索的标志
     */
    void resetAllResultStatus();

    /**
     * 进行搜索，每当有一个结果匹配后调用resultCollector
     *
     * @param searchCase：有D          F Full Case四种，分别代表只搜索文件夹(Directory)，只搜索文件(File)，全字匹配(Full)，大小写敏感(Case Sensitive)
     * @param isIgnoreCase           是否忽略大小写，当searchCase中存在case时该字段为true
     * @param searchText             原始搜索关键字
     * @param keywords               搜索关键字，将searchText用";"切割得到
     * @param keywordsLowerCase      搜索关键字，全小写字母，内容与keywords相同，但字母为全小写
     * @param isKeywordPath          保存keywords中的关键字是路径判断还是文件名判断
     * @param maxResultNumber        最大匹配结果数量限制
     * @param resultCollectThreadNum 收集GPU搜索结果并发线程数
     * @param resultCollector        有一个结果匹配后的回调方法
     */
    void match(String[] searchCase,
               boolean isIgnoreCase,
               String searchText,
               String[] keywords,
               String[] keywordsLowerCase,
               boolean[] isKeywordPath,
               int maxResultNumber,
               int resultCollectThreadNum,
               BiConsumer<String, String> resultCollector);

    /**
     * 判断GPU加速是否可用
     *
     * @return true如果可以进行GPU加速
     */
    boolean isGPUAvailableOnSystem();

    /**
     * 判断某一个缓存是否搜索完成
     *
     * @param key 缓存key
     * @return true如果全部完成
     */
    boolean isMatchDone(String key);

    /**
     * 获取某一个缓存搜索完成后匹配的结果数量
     *
     * @param key 缓存key
     * @return 匹配的结果数量
     */
    int matchedNumber(String key);

    /**
     * 停止搜索
     */
    void stopCollectResults();

    /**
     * 判断缓存是否存在
     *
     * @return true如果至少保存了一个缓存
     */
    boolean hasCache();

    /**
     * 判断某一个缓存是否存在
     *
     * @param key 缓存key
     * @return true如果缓存名为key的缓存存在
     */
    boolean isCacheExist(String key);

    /**
     * 添加缓存到GPU显存
     *
     * @param key            缓存key
     * @param recordSupplier 字符串supplier，由GPU加速dll通过jni进行调用，防止字符串过多导致OOM
     */
    void initCache(String key, Supplier<String> recordSupplier);

    /**
     * 向某个缓存添加数据
     *
     * @param key     缓存key
     * @param records 待添加的数据
     */
    void addRecordsToCache(String key, Object[] records);

    /**
     * 删除某一个缓存中的数据
     *
     * @param key     缓存key
     * @param records 待删除的数据
     */
    void removeRecordsFromCache(String key, Object[] records);

    /**
     * 删除某一个缓存
     *
     * @param key 缓存key
     */
    void clearCache(String key);

    /**
     * 删除所有缓存
     */
    void clearAllCache();

    /**
     * 缓存是否有效
     *
     * @param key 缓存key
     * @return true如果缓存仍然有效，当addRecordsToCache()失败，表示当前缓存的空位已经不足，出现了数据丢失，缓存被标记为无效。
     * @see #addRecordsToCache(String, Object[])
     */
    boolean isCacheValid(String key);

    /**
     * 获取系统显存占用，用于在用户使用过多显存时释放缓存。
     *
     * @return 内存占用百分比，数值从0-100
     */
    int getGPUMemUsage();

    /**
     * 初始化
     */
    void initialize();

    /**
     * 释放所有资源
     */
    void release();

    /**
     * 获取GPU设备名
     *
     * @return GPU设备名
     */
    String[] getDevices();

    /**
     * 设置使用的GPU设备，deviceNum为getDevices()返回的数组的下标
     *
     * @param deviceNum GPU设备id
     * @return true如果成功使用设备并初始化
     */
    boolean setDevice(int deviceNum);
}
