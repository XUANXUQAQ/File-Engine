package file.engine.services;

import com.google.gson.Gson;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.dllInterface.*;
import file.engine.dllInterface.gpu.GPUAccelerator;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BootSystemEvent;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.database.*;
import file.engine.event.handler.impl.database.gpu.GPUAddRecordEvent;
import file.engine.event.handler.impl.database.gpu.GPUClearCacheEvent;
import file.engine.event.handler.impl.database.gpu.GPURemoveRecordEvent;
import file.engine.event.handler.impl.frame.searchBar.SearchBarReadyEvent;
import file.engine.event.handler.impl.monitor.disk.StartMonitorDiskEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.frames.SearchBar;
import file.engine.services.utils.PathMatchUtil;
import file.engine.services.utils.SystemInfoUtil;
import file.engine.services.utils.connection.SQLiteUtil;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.utils.ProcessUtil;
import file.engine.utils.RegexUtil;
import file.engine.utils.Bit;
import file.engine.utils.gson.GsonUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDate;
import java.time.Period;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.stream.Collectors;

public class DatabaseService {
    private final ConcurrentLinkedQueue<SQLWithTaskId> commandQueue = new ConcurrentLinkedQueue<>();
    private final AtomicReference<Constants.Enums.DatabaseStatus> status = new AtomicReference<>(Constants.Enums.DatabaseStatus.NORMAL);
    private final AtomicBoolean isExecuteImmediately = new AtomicBoolean(false);
    private final AtomicInteger databaseCacheNum = new AtomicInteger(0);
    private final Set<TableNameWeightInfo> tableSet = ConcurrentHashMap.newKeySet();    //保存从0-40数据库的表，使用频率和名字对应，使经常使用的表最快被搜索到
    private final AtomicBoolean isDatabaseUpdated = new AtomicBoolean(false);
    private ConcurrentLinkedQueue<String> tempResults;  //在优先文件夹和数据库cache未搜索完时暂时保存结果，搜索完后会立即被转移到listResults
    private Set<String> tempResultsForEvent; //在SearchDoneEvent中保存的容器
    private final AtomicBoolean shouldStopSearch = new AtomicBoolean(false);
    private final ConcurrentLinkedQueue<Pair> priorityMap = new ConcurrentLinkedQueue<>();
    //tableCache 数据表缓存，在初始化时将会放入所有的key和一个空的cache，后续需要缓存直接放入空的cache中，不再创建新的cache实例
    private final ConcurrentHashMap<String, Cache> tableCache = new ConcurrentHashMap<>();
    private final AtomicInteger tableCacheCount = new AtomicInteger();
    private static volatile String[] searchCase;
    private static volatile boolean isIgnoreCase;
    private static volatile String searchText;
    private static volatile String[] keywords;
    private static volatile String[] keywordsLowerCase;
    private static volatile boolean[] isKeywordPath;
    private final AtomicBoolean isSharedMemoryCreated = new AtomicBoolean(false);
    private final LinkedHashSet<String> databaseCacheSet = new LinkedHashSet<>();
    private final AtomicInteger searchThreadCount = new AtomicInteger(0);
    private final AtomicLong startSearchTimeMills = new AtomicLong(0);
    private static final int MAX_TEMP_QUERY_RESULT_CACHE = 128;
    private static final int MAX_CACHED_RECORD_NUM = 10240 * 5;
    private static final int MAX_SQL_NUM = 5000;
    private static final int MAX_RESULTS = 200;

    private static volatile DatabaseService INSTANCE = null;

    @AllArgsConstructor
    private static class Pair {
        private final String suffix;
        private final int priority;
    }

    private static class Cache {
        private final AtomicBoolean isCached = new AtomicBoolean(false);
        private final AtomicBoolean isFileLost = new AtomicBoolean(false);
        private ConcurrentLinkedQueue<String> data = null;

        private boolean isCacheValid() {
            return isCached.get() && !isFileLost.get();
        }
    }

    private static class TableNameWeightInfo {
        private final String tableName;
        private final AtomicLong weight;

        private TableNameWeightInfo(String tableName, int weight) {
            this.tableName = tableName;
            this.weight = new AtomicLong(weight);
        }
    }

    private DatabaseService() {
    }

    private void invalidateAllCache() {
        GPUClearCacheEvent gpuClearCacheEvent = new GPUClearCacheEvent();
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(gpuClearCacheEvent);
        eventManagement.waitForEvent(gpuClearCacheEvent, 60_000);
        tableCache.values().forEach(each -> {
            each.isCached.set(false);
            each.data = null;
        });
        tableCacheCount.set(0);
    }

    /**
     * 通过表名获得表的权重信息
     *
     * @param tableName 表名
     * @return 权重信息
     */
    private TableNameWeightInfo getInfoByName(String tableName) {
        for (TableNameWeightInfo each : tableSet) {
            if (each.tableName.equals(tableName)) {
                return each;
            }
        }
        return null;
    }

    /**
     * 更新权重信息
     *
     * @param tableName 表名
     * @param weight    权重
     */
    private void updateTableWeight(String tableName, long weight) {
        TableNameWeightInfo origin = getInfoByName(tableName);
        if (origin == null) {
            return;
        }
        origin.weight.addAndGet(weight);
        String format = String.format("UPDATE weight SET TABLE_WEIGHT=%d WHERE TABLE_NAME=\"%s\"", origin.weight.get(), tableName);
        addToCommandQueue(new SQLWithTaskId(format, SqlTaskIds.UPDATE_WEIGHT, "weight"));
        if (IsDebug.isDebug()) {
            System.err.println("已更新" + tableName + "权重, 之前为" + origin + "***增加了" + weight);
        }
    }

    /**
     * 获取数据库缓存条目数量，用于判断软件是否还能继续写入缓存
     */
    private void initDatabaseCacheNum() {
        try (Statement stmt = SQLiteUtil.getStatement("cache");
             ResultSet resultSet = stmt.executeQuery("SELECT COUNT(PATH) FROM cache;")) {
            databaseCacheNum.set(resultSet.getInt(1));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 处理所有sql线程
     */
    private void executeSqlCommandsThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.notMainExit()) {
                    if (isExecuteImmediately.get()) {
                        try {
                            isExecuteImmediately.set(false);
                            executeAllCommands();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    public static DatabaseService getInstance() {
        if (INSTANCE == null) {
            synchronized (DatabaseService.class) {
                if (INSTANCE == null) {
                    INSTANCE = new DatabaseService();
                }
            }
        }
        return INSTANCE;
    }

    /**
     * 开始监控磁盘文件变化
     */
    private static void startMonitorDisk() {
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        EventManagement eventManagement = EventManagement.getInstance();
        TranslateService translateService = TranslateService.getInstance();
        String disks = AllConfigs.getInstance().getAvailableDisks();
        String[] splitDisks = RegexUtil.comma.split(disks);
        if (isAdmin()) {
            for (String root : splitDisks) {
                cachedThreadPoolUtil.executeTask(() -> FileMonitor.INSTANCE.monitor(root), false);
            }
        } else {
            eventManagement.putEvent(new ShowTaskBarMessageEvent(
                    translateService.getTranslation("Warning"),
                    translateService.getTranslation("Not administrator, file monitoring function is turned off")));
        }
    }

    private void searchFolder(String folder) {
        File path = new File(folder);
        if (!path.exists()) {
            return;
        }
        File[] files = path.listFiles();
        if (null == files || files.length == 0) {
            return;
        }
        List<File> filesList = List.of(files);
        LinkedList<File> dirsToSearch = new LinkedList<>(filesList);
        LinkedList<File> listRemainDir = new LinkedList<>(filesList);
        do {
            File remain = listRemainDir.poll();
            if (remain == null) {
                continue;
            }
            if (remain.isDirectory()) {
                File[] subFiles = remain.listFiles();
                if (subFiles != null) {
                    List<File> subFilesList = List.of(subFiles);
                    listRemainDir.addAll(subFilesList);
                    dirsToSearch.addAll(subFilesList);
                }
            } else {
                checkIsMatchedAndAddToList(remain.getAbsolutePath(), null);
            }
        } while (!listRemainDir.isEmpty() && !shouldStopSearch.get());
        dirsToSearch.forEach(eachDir -> checkIsMatchedAndAddToList(eachDir.getAbsolutePath(), null));
    }

    /**
     * 搜索文件夹
     */
    private void searchStartMenu() {
        searchFolder(GetWindowsKnownFolder.INSTANCE.getKnownFolder("{A4115719-D62E-491D-AA7C-E74B8BE3B067}"));
        searchFolder(GetWindowsKnownFolder.INSTANCE.getKnownFolder("{625B53C3-AB48-4EC1-BA1F-A1EF4146FC19}"));
    }

    /**
     * 搜索桌面
     */
    private void searchDesktop() {
        searchFolder(GetWindowsKnownFolder.INSTANCE.getKnownFolder("{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}"));
        searchFolder(GetWindowsKnownFolder.INSTANCE.getKnownFolder("{C4AA340D-F20F-4863-AFEF-F87EF2E6BA25}"));
    }

    /**
     * 搜索优先文件夹
     */
    private void searchPriorityFolder() {
        searchFolder(AllConfigs.getInstance().getPriorityFolder());
    }

    /**
     * 返回满足数据在minRecordNum-maxRecordNum之间的表可以被缓存的表
     *
     * @param disks                硬盘盘符
     * @param tableQueueByPriority 后缀优先级表，从高到低优先级逐渐降低
     * @param isStopCreateCache    是否停止
     * @param minRecordNum         最小数据量
     * @param maxRecordNum         最大数据量
     * @return key为[盘符, 表名, 优先级]，例如 [C,list10,9]，value为实际数据量所占的字节数
     */
    private LinkedHashMap<String, Integer> scanDatabaseAndSelectCacheTable(String[] disks,
                                                                           ConcurrentLinkedQueue<String> tableQueueByPriority,
                                                                           Supplier<Boolean> isStopCreateCache,
                                                                           @SuppressWarnings("SameParameterValue") int minRecordNum,
                                                                           int maxRecordNum) {
        if (minRecordNum > maxRecordNum) {
            throw new RuntimeException("minRecordNum > maxRecordNum");
        }
        //检查哪些表符合缓存条件，通过表权重依次向下排序
        LinkedHashMap<String, Integer> tableNeedCache = new LinkedHashMap<>();
        for (String diskPath : disks) {
            String disk = String.valueOf(diskPath.charAt(0));
            try (Statement stmt = SQLiteUtil.getStatement(disk)) {
                for (String tableName : tableQueueByPriority) {
                    for (Pair pair : priorityMap) {
                        if (isStopCreateCache.get()) {
                            return tableNeedCache;
                        }
                        boolean canBeCached;
                        try (ResultSet resultCount = stmt.executeQuery("SELECT COUNT(*) as total_num FROM " + tableName + " WHERE PRIORITY=" + pair.priority)) {
                            if (resultCount.next()) {
                                final int num = resultCount.getInt("total_num");
                                canBeCached = num >= minRecordNum && num <= maxRecordNum;
                            } else {
                                canBeCached = false;
                            }
                        }
                        if (!canBeCached)
                            continue;
                        try (ResultSet resultsLength = stmt.executeQuery("SELECT SUM(LENGTH(PATH)) as total_bytes FROM " + tableName + " WHERE PRIORITY=" + pair.priority)) {
                            if (resultsLength.next()) {
                                final int resultsBytes = resultsLength.getInt("total_bytes");
                                tableNeedCache.put(disk + "," + tableName + "," + pair.priority, resultsBytes);
                            }
                        }
                    }
                }
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        return tableNeedCache;
    }

    /**
     * 扫描数据库并添加缓存
     */
    private void saveTableCacheThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            final int checkTimeInterval = 10 * 60 * 1000; // 10 min
            final int startUpLatency = 10 * 1000; // 10s
            var startCheckInfo = new Object() {
                long startCheckTimeMills = System.currentTimeMillis() - checkTimeInterval + startUpLatency;
            };
            final Supplier<Boolean> isStopCreateCache =
                    () -> !eventManagement.notMainExit() || searchThreadCount.get() != 0 ||
                            status.get() == Constants.Enums.DatabaseStatus.MANUAL_UPDATE ||
                            status.get() == Constants.Enums.DatabaseStatus.VACUUM;
            final Supplier<Boolean> isStartSaveCache =
                    () -> (System.currentTimeMillis() - startCheckInfo.startCheckTimeMills > checkTimeInterval &&
                            status.get() == Constants.Enums.DatabaseStatus.NORMAL &&
                            !GetHandle.INSTANCE.isForegroundFullscreen()) ||
                            (isDatabaseUpdated.get());
            try {
                AllConfigs allConfigs = AllConfigs.getInstance();
                final int createMemoryThreshold = 70;
                final int createGPUCacheThreshold = 50;
                final int freeGPUCacheThreshold = 70;
                while (eventManagement.notMainExit()) {
                    if (isStartSaveCache.get()) {
                        if (isDatabaseUpdated.get()) {
                            isDatabaseUpdated.set(false);
                        }
                        startCheckInfo.startCheckTimeMills = System.currentTimeMillis();
                        if (allConfigs.isGPUAcceleratorEnabled()) {
                            final int gpuMemUsage = GPUAccelerator.INSTANCE.getGPUMemUsage();
                            if (gpuMemUsage < createGPUCacheThreshold) {
                                createGpuCache(isStopCreateCache, createGPUCacheThreshold);
                            }
                        } else {
                            final double memoryUsage = SystemInfoUtil.getMemoryUsage();
                            if (memoryUsage * 100 < createMemoryThreshold) {
                                createMemoryCache(isStopCreateCache);
                            }
                        }
                        if (status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
                            casSetStatus(Constants.Enums.DatabaseStatus._SHARED_MEMORY, Constants.Enums.DatabaseStatus.NORMAL);
                        }
                    } else {
                        if (allConfigs.isGPUAcceleratorEnabled()) {
                            final int gpuMemUsage = GPUAccelerator.INSTANCE.getGPUMemUsage();
                            if (gpuMemUsage >= freeGPUCacheThreshold) {
                                // 防止显存占用超过70%后仍然扫描数据库
                                startCheckInfo.startCheckTimeMills = System.currentTimeMillis();
                                if (GPUAccelerator.INSTANCE.hasCache()) {
                                    GPUClearCacheEvent gpuClearCacheEvent = new GPUClearCacheEvent();
                                    eventManagement.putEvent(gpuClearCacheEvent);
                                    eventManagement.waitForEvent(gpuClearCacheEvent);
                                }
                            }
                        }
                    }
                    TimeUnit.SECONDS.sleep(1);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    private void createMemoryCache(Supplier<Boolean> isStopCreateCache) {
        System.out.println("添加缓存");
        String availableDisks = AllConfigs.getInstance().getAvailableDisks();
        ConcurrentLinkedQueue<String> tableQueueByPriority = initTableQueueByPriority();
        // 系统内存使用少于70%
        String[] disks = RegexUtil.comma.split(availableDisks);
        LinkedHashMap<String, Integer> tableNeedCache = scanDatabaseAndSelectCacheTable(disks,
                tableQueueByPriority,
                isStopCreateCache,
                100,
                2000);
        saveTableCache(isStopCreateCache, tableNeedCache);
    }

    @SuppressWarnings("SameParameterValue")
    private void createGpuCache(Supplier<Boolean> isStopCreateCache, int createGpuCacheThreshold) {
        System.out.println("添加gpu缓存");
        String availableDisks = AllConfigs.getInstance().getAvailableDisks();
        ConcurrentLinkedQueue<String> tableQueueByPriority = initTableQueueByPriority();
        String[] disks = RegexUtil.comma.split(availableDisks);
        LinkedHashMap<String, Integer> tableNeedCache = scanDatabaseAndSelectCacheTable(disks,
                tableQueueByPriority,
                isStopCreateCache,
                1,
                Integer.MAX_VALUE);
        saveTableCacheForGPU(isStopCreateCache, tableNeedCache, createGpuCacheThreshold);
    }

    /**
     * 缓存数据表到显存中
     *
     * @param isStopCreateCache 是否停止
     * @param tableNeedCache    需要缓存的表
     */
    private void saveTableCacheForGPU(Supplier<Boolean> isStopCreateCache, LinkedHashMap<String, Integer> tableNeedCache, int createGpuCacheThreshold) {
        for (Map.Entry<String, Cache> entry : tableCache.entrySet()) {
            String key = entry.getKey();
            if (tableNeedCache.containsKey(key)) {
                //超过128M字节或已存在缓存
                if (GPUAccelerator.INSTANCE.isCacheExist(key) || tableNeedCache.get(key) > 128 * 1024 * 1024) {
                    continue;
                }
                String[] info = RegexUtil.comma.split(key);
                try (Statement stmt = SQLiteUtil.getStatement(info[0]);
                     ResultSet resultSet = stmt.executeQuery("SELECT PATH FROM " + info[1] + " " + "WHERE PRIORITY=" + info[2])) {
                    EventManagement eventManagement = EventManagement.getInstance();
                    GPUAccelerator.INSTANCE.initCache(key, () -> {
                        try {
                            if (resultSet.next() && eventManagement.notMainExit()) {
                                return resultSet.getString("PATH");
                            }
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        return null;
                    });
                    if (isStopCreateCache.get()) {
                        break;
                    }
                    var usage = GPUAccelerator.INSTANCE.getGPUMemUsage();
                    if (usage > createGpuCacheThreshold) {
                        break;
                    }
                } catch (SQLException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * 缓存数据表
     *
     * @param isStopCreateCache 是否停止
     * @param tableNeedCache    需要缓存的表
     */
    private void saveTableCache(Supplier<Boolean> isStopCreateCache, LinkedHashMap<String, Integer> tableNeedCache) {
        //开始缓存数据库表
        out:
        for (Map.Entry<String, Cache> entry : tableCache.entrySet()) {
            String key = entry.getKey();
            Cache cache = entry.getValue();
            if (tableNeedCache.containsKey(key)) {
                //当前表可以被缓存
                if (tableCacheCount.get() + tableNeedCache.get(key) < MAX_CACHED_RECORD_NUM - 1000 && !cache.isCacheValid()) {
                    cache.data = new ConcurrentLinkedQueue<>();
                    String[] info = RegexUtil.comma.split(key);
                    try (Statement stmt = SQLiteUtil.getStatement(info[0]);
                         ResultSet resultSet = stmt.executeQuery("SELECT PATH FROM " + info[1] + " " + "WHERE PRIORITY=" + info[2])) {
                        while (resultSet.next()) {
                            if (isStopCreateCache.get()) {
                                break out;
                            }
                            cache.data.add(resultSet.getString("PATH"));
                            tableCacheCount.incrementAndGet();
                        }
                    } catch (SQLException e) {
                        e.printStackTrace();
                    }
                    cache.isCached.compareAndSet(cache.isCached.get(), true);
                    cache.isFileLost.set(false);
                }
            } else {
                if (cache.isCached.get()) {
                    cache.isCached.compareAndSet(cache.isCached.get(), false);
                    int num = cache.data.size();
                    tableCacheCount.compareAndSet(tableCacheCount.get(), tableCacheCount.get() - num);
                    cache.data = null;
                }
            }
        }
    }

    private void syncFileChangesThread() {
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        EventManagement eventManagement = EventManagement.getInstance();
        var fileChanges = new SynchronousQueue<Runnable>();
        cachedThreadPoolUtil.executeTask(() -> {
            try {
                while (eventManagement.notMainExit()) {
                    String addFile = FileMonitor.INSTANCE.pop_add_file();
                    String deleteFile = FileMonitor.INSTANCE.pop_del_file();
                    if (addFile == null && deleteFile == null) {
                        TimeUnit.MILLISECONDS.sleep(1);
                        continue;
                    }
                    fileChanges.put(() -> {
                        addFileToDatabase(addFile);
                        removeFileFromDatabase(deleteFile);
                    });
                }
                fileChanges.put(() -> {
                });
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });
        cachedThreadPoolUtil.executeTask(() -> {
            try {
                while (eventManagement.notMainExit()) {
                    fileChanges.take().run();
                }
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });
    }

    public Set<String> getCache() {
        return new LinkedHashSet<>(databaseCacheSet);
    }

    /**
     * 将缓存中的文件保存到cacheSet中
     */
    private void prepareDatabaseCache() {
        String eachLine;
        try (Statement statement = SQLiteUtil.getStatement("cache");
             ResultSet resultSet = statement.executeQuery("SELECT PATH FROM cache;")) {
            while (resultSet.next()) {
                eachLine = resultSet.getString("PATH");
                databaseCacheSet.add(eachLine);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 从缓存中搜索结果并将匹配的放入listResults
     */
    private void searchCache() {
        for (String each : databaseCacheSet) {
            if (Files.exists(Path.of(each))) {
                checkIsMatchedAndAddToList(each, null);
            } else {
                EventManagement.getInstance().putEvent(new DeleteFromCacheEvent(each));
            }
        }
    }

    /**
     * * 检查文件路径是否匹配然后加入到列表
     *
     * @param path 文件路径
     * @return true如果匹配成功
     */
    private boolean checkIsMatchedAndAddToList(String path, Collection<String> container) {
        boolean ret = false;
        if (PathMatchUtil.check(path,
                searchCase,
                isIgnoreCase,
                searchText,
                keywords,
                keywordsLowerCase,
                isKeywordPath)) {
            //字符串匹配通过
            ret = true;
            if (container == null) {
                if (tempResultsForEvent.add(path)) {
                    tempResults.add(path);
                    if (!shouldStopSearch.get() && tempResultsForEvent.size() >= MAX_RESULTS) {
                        stopSearch();
                    }
                }
            } else {
                if (!tempResultsForEvent.contains(path) && !container.contains(path) && container.add(path)) {
                    if (!shouldStopSearch.get() && tempResultsForEvent.size() + container.size() >= MAX_RESULTS) {
                        stopSearch();
                    }
                }
            }
        }
        return ret;
    }

    /**
     * 搜索数据库并加入到tempQueue中
     *
     * @param sql sql
     */
    private int searchAndAddToTempResults(String sql, Collection<String> container, String diskStr) {
        int matchedResultCount = 0;
        //结果太多则不再进行搜索
        if (shouldStopSearch.get()) {
            return matchedResultCount;
        }
        String[] tmpQueryResultsCache = new String[MAX_TEMP_QUERY_RESULT_CACHE];
        EventManagement eventManagement = EventManagement.getInstance();
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement(sql, diskStr);
             ResultSet resultSet = pStmt.executeQuery()) {
            boolean isExit = false;
            while (!isExit && eventManagement.notMainExit()) {
                int i = 0;
                // 先将结果查询出来，再进行字符串匹配，提高吞吐量
                while (i < MAX_TEMP_QUERY_RESULT_CACHE && eventManagement.notMainExit()) {
                    if (resultSet.next()) {
                        tmpQueryResultsCache[i] = resultSet.getString("PATH");
                        ++i;
                    } else {
                        isExit = true;
                        break;
                    }
                }
                //结果太多则不再进行搜索
                //用户重新输入了信息
                if (shouldStopSearch.get()) {
                    return matchedResultCount;
                }
                for (int j = 0; j < i; ++j) {
                    if (checkIsMatchedAndAddToList(tmpQueryResultsCache[j], container)) {
                        ++matchedResultCount;
                    }
                }
            }
        } catch (SQLException e) {
            System.err.println("error sql : " + sql);
            e.printStackTrace();
        }
        return matchedResultCount;
    }

    /**
     * 根据优先级将表排序放入tableQueue
     */
    private ConcurrentLinkedQueue<String> initTableQueueByPriority() {
        ConcurrentLinkedQueue<String> tableQueue = new ConcurrentLinkedQueue<>();
        LinkedList<TableNameWeightInfo> tmpCommandList = new LinkedList<>(tableSet);
        //将tableSet通过权重排序
        tmpCommandList.sort((o1, o2) -> Long.compare(o2.weight.get(), o1.weight.get()));
        for (TableNameWeightInfo each : tmpCommandList) {
            if (IsDebug.isDebug()) {
                System.out.println("已添加表" + each.tableName + "----权重" + each.weight.get());
            }
            tableQueue.add(each.tableName);
        }
        return tableQueue;
    }

    /**
     * 初始化所有表名和权重信息，不要移动到构造函数中，否则会造成死锁
     * 在该任务前可能会有设置搜索框颜色等各种任务，这些任务被设置为异步，若在构造函数未执行完成时，会造成无法构造实例
     */
    private void initTableMap() {
        boolean isNeedSubtract = false;
        HashMap<String, Integer> weights = queryAllWeights();
        if (!weights.isEmpty()) {
            for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
                Integer weight = weights.get("list" + i);
                if (weight == null) {
                    weight = 0;
                }
                if (weight > 100_000_000) {
                    isNeedSubtract = true;
                }
                tableSet.add(new TableNameWeightInfo("list" + i, weight));
            }
        } else {
            for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
                tableSet.add(new TableNameWeightInfo("list" + i, 0));
            }
        }
        if (isNeedSubtract) {
            tableSet.forEach(tableNameWeightInfo -> tableNameWeightInfo.weight.set(tableNameWeightInfo.weight.get() / 2));
        }
    }

    /**
     * 根据上面分配的位信息，从第二位开始，与taskStatus做与运算，并向右偏移，若结果为1，则表示该任务完成
     */
    private void waitForTasks() {
        try {
            EventManagement eventManagement = EventManagement.getInstance();
            while (!PrepareSearchInfo.taskStatus.equals(PrepareSearchInfo.allTaskStatus) && eventManagement.notMainExit()) {
                TimeUnit.MILLISECONDS.sleep(10);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            searchDone();
        }
    }

    private void searchDone() {
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(new SearchDoneEvent(new ConcurrentLinkedQueue<>(tempResultsForEvent)));
        if (AllConfigs.getInstance().isGPUAcceleratorEnabled() && eventManagement.notMainExit()) {
            GPUAccelerator.INSTANCE.stopCollectResults();
        }
    }

    /**
     * 解析出sql中的priority
     *
     * @param sql sql
     */
    private String getPriorityFromSql(String sql) {
        final int pos = sql.indexOf('=');
        if (pos == -1) {
            throw new RuntimeException("error sql no priority");
        }
        return sql.substring(pos + 1);
    }

    /**
     * 创建搜索任务
     * nonFormattedSql将会生成从list0-40，根据priority从高到低排序的SQL语句，第一个map中key保存未格式化的sql，value保存表名称
     * containerMap中key为每个任务分配到的位，value为nonFormattedSql中的临时存储容器
     * 生成任务顺序会根据list的权重和priority来生成
     * <p>
     * PrepareSearchInfo.taskStatus      用于保存任务信息，这是一个通用变量，从第二个位开始，每一个位代表一个任务，当任务完成，该位将被置为1，否则为0，例如第一个和第三个任务完成，第二个未完成，则为1010
     * PrepareSearchInfo.allTaskStatus   所有的任务信息，从第二位开始，只要有任务被创建，该位就为1，例如三个任务被创建，则为1110
     * PrepareSearchInfo.containerMap    每个任务搜索出来的结果都会被放到一个属于它自己的一个容器中，该容器保存任务与容器的映射关系
     * PrepareSearchInfo.taskMap         任务
     *
     * @param nonFormattedSql 未格式化搜索字段的SQL
     * @return gpu搜索容器
     */
    private ConcurrentHashMap<String, Set<String>> addSearchTasks(LinkedList<LinkedHashMap<String, String>> nonFormattedSql) {
        ConcurrentHashMap<String, Set<String>> gpuSearchContainer = new ConcurrentHashMap<>();
        Bit number = new Bit(new byte[]{1});
        AllConfigs allConfigs = AllConfigs.getInstance();
        String availableDisks = allConfigs.getAvailableDisks();
        for (String eachDisk : RegexUtil.comma.split(availableDisks)) {
            ConcurrentLinkedQueue<Runnable> tasks = new ConcurrentLinkedQueue<>();
            PrepareSearchInfo.taskMap.put(eachDisk, tasks);
            //向任务队列tasks添加任务
            for (var commandsMap : nonFormattedSql) {
                //为每个任务分配的位，不断左移以不断进行分配
                number.shiftLeft(1);
                Bit currentTaskNum = new Bit(number);
                {//记录当前任务信息到allTaskStatus
                    byte[] origin;
                    while ((origin = PrepareSearchInfo.allTaskStatus.getBytes()) != null) {
                        Bit or = Bit.or(origin, currentTaskNum.getBytes());
                        if (PrepareSearchInfo.allTaskStatus.set(origin, or)) {
                            break;
                        }
                    }
                }
                Set<String> container = ConcurrentHashMap.newKeySet();
                //每一个任务负责查询一个priority和list0-list40生成的41个SQL
                if (status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
                    addTaskForSharedMemory0(eachDisk, tasks, container, commandsMap, currentTaskNum);
                } else {
                    addTaskForDatabase0(eachDisk, tasks, container, commandsMap, currentTaskNum);
                }
                //为gpu搜索生成的container
                for (var sqlAndTableName : commandsMap.entrySet()) {
                    String eachSql = sqlAndTableName.getKey();
                    String tableName = sqlAndTableName.getValue();
                    String priority = getPriorityFromSql(eachSql);
                    String key = eachDisk.charAt(0) + "," + tableName + "," + priority;
                    gpuSearchContainer.put(key, container);
                }
            }
        }
        return gpuSearchContainer;
    }

    private void addTaskForDatabase0(String diskChar,
                                     ConcurrentLinkedQueue<Runnable> tasks,
                                     Collection<String> resultContainer,
                                     LinkedHashMap<String, String> sqlToExecute,
                                     Bit currentTaskNum) {
        var tempResultsLocal = tempResults;
        var tempResultsForEventLocal = tempResultsForEvent;
        AllConfigs allConfigs = AllConfigs.getInstance();
        tasks.add(() -> {
            try {
                for (var sqlAndTableName : sqlToExecute.entrySet()) {
                    String diskStr = String.valueOf(diskChar.charAt(0));
                    String eachSql = sqlAndTableName.getKey();
                    String tableName = sqlAndTableName.getValue();
                    String priority = getPriorityFromSql(eachSql);
                    String key = diskStr + "," + tableName + "," + priority;
                    final long matchedNum;
                    if (allConfigs.isGPUAcceleratorEnabled() && GPUAccelerator.INSTANCE.isMatchDone(key)) {
                        //gpu搜索已经放入container，只需要获取matchedNum修改权重即可
                        matchedNum = GPUAccelerator.INSTANCE.matchedNumber(key);
                    } else {
                        if (!shouldStopSearch.get()) {
                            matchedNum = searchFromDatabaseOrCache(resultContainer, diskStr, eachSql, key);
                        } else {
                            matchedNum = 0;
                        }
                    }
                    final long weight = Math.min(matchedNum, 5);
                    if (weight != 0L) {
                        //更新表的权重，每次搜索将会按照各个表的权重排序
                        updateTableWeight(tableName, weight);
                    }
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            } finally {
                for (String s : resultContainer) {
                    if (tempResultsForEventLocal.size() >= MAX_RESULTS)
                        break;
                    if (tempResultsForEventLocal.add(s)) {
                        tempResultsLocal.add(s);
                    }
                }
                //执行完后将对应的线程flag设为1
                byte[] originalBytes;
                while ((originalBytes = PrepareSearchInfo.taskStatus.getBytes()) != null) {
                    Bit or = Bit.or(originalBytes, currentTaskNum.getBytes());
                    if (PrepareSearchInfo.taskStatus.set(originalBytes, or)) {
                        break;
                    }
                }
            }
        });
    }

    private void addTaskForSharedMemory0(String diskChar,
                                         ConcurrentLinkedQueue<Runnable> tasks,
                                         Collection<String> resultContainer,
                                         LinkedHashMap<String, String> sqlToExecute,
                                         Bit currentTaskNum) {
        var tempResultsLocal = tempResults;
        var tempResultsForEventLocal = tempResultsForEvent;
        tasks.add(() -> {
            try {
                for (var sqlAndTableName : sqlToExecute.entrySet()) {
                    if (shouldStopSearch.get()) {
                        return;
                    }
                    String eachSql = sqlAndTableName.getKey();
                    String listName = sqlAndTableName.getValue();
                    int priority = Integer.parseInt(getPriorityFromSql(eachSql));
                    String result;
                    for (int count = 0;
                         !shouldStopSearch.get() && ((result = ResultPipe.INSTANCE.getResult(diskChar.charAt(0), listName, priority, count)) != null);
                         ++count) {
                        checkIsMatchedAndAddToList(result, resultContainer);
                    }
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            } finally {
                for (String s : resultContainer) {
                    if (tempResultsForEventLocal.size() >= MAX_RESULTS)
                        break;
                    if (tempResultsForEventLocal.add(s)) {
                        tempResultsLocal.add(s);
                    }
                }
                byte[] originalBytes;
                while ((originalBytes = PrepareSearchInfo.taskStatus.getBytes()) != null) {
                    Bit or = Bit.or(originalBytes, currentTaskNum.getBytes());
                    if (PrepareSearchInfo.taskStatus.set(originalBytes, or)) {
                        break;
                    }
                }
            }
        });
    }

    /**
     * 从数据库中查找
     *
     * @param container 存储结果容器
     * @param diskStr   磁盘盘符
     * @param sql       sql
     * @param key       查询key，例如 [C,list10,-1]
     * @return 查询的结果数量
     */
    private long searchFromDatabaseOrCache(Collection<String> container, String diskStr, String sql, String key) {
        if (shouldStopSearch.get()) {
            return 0;
        }
        long matchedNum;
        Cache cache = tableCache.get(key);
        if (cache != null && cache.isCacheValid()) {
            if (IsDebug.isDebug()) {
                System.out.println("从缓存中读取 " + key);
            }
            matchedNum = cache.data.stream()
                    .filter(eachRecord -> checkIsMatchedAndAddToList(eachRecord, container))
                    .count();
        } else {
            //格式化是为了以后的拓展性
            String formattedSql = String.format(sql, "PATH");
            //当前数据库表中有多少个结果匹配成功
            matchedNum = searchAndAddToTempResults(
                    formattedSql,
                    container,
                    diskStr);
        }
        return matchedNum;
    }

    /**
     * 生成未格式化的sql
     * 第一个map中每一个priority加上list0-list40会生成41条SQL作为key，value是搜索的表，即SELECT* FROM [list?]中的[list?];
     *
     * @return set
     */
    private LinkedList<LinkedHashMap<String, String>> getNonFormattedSqlFromTableQueue() {
        LinkedList<LinkedHashMap<String, String>> sqlColumnMap = new LinkedList<>();
        if (priorityMap.isEmpty()) {
            return sqlColumnMap;
        }
        ConcurrentLinkedQueue<String> tableQueue = initTableQueueByPriority();
        int asciiSum = 0;
        if (keywords != null) {
            for (String keyword : keywords) {
                int ascII = GetAscII.INSTANCE.getAscII(keyword); //其实是utf8编码的值
                asciiSum += Math.max(ascII, 0);
            }
        }
        int asciiGroup = asciiSum / 100;
        if (asciiGroup > Constants.ALL_TABLE_NUM) {
            asciiGroup = Constants.ALL_TABLE_NUM;
        }
        String firstTableName = "list" + asciiGroup;
        // 有d代表只需要搜索文件夹，文件夹的priority为-1
        if (searchCase != null && Arrays.asList(searchCase).contains("d")) {
            //首先根据输入的keywords找到对应的list
            LinkedHashMap<String, String> tmpPriorityMap = new LinkedHashMap<>();
            String eachSql = "SELECT %s FROM " + firstTableName + " WHERE PRIORITY=" + "-1";
            tmpPriorityMap.put(eachSql, firstTableName);
            tableQueue.stream().filter(each -> !each.equals(firstTableName)).forEach(each -> {
                // where后面=不能有空格，否则解析priority会出错
                String sql = "SELECT %s FROM " + each + " WHERE PRIORITY=" + "-1";
                tmpPriorityMap.put(sql, each);
            });
            sqlColumnMap.add(tmpPriorityMap);
        } else {
            for (Pair i : priorityMap) {
                LinkedHashMap<String, String> eachPriorityMap = new LinkedHashMap<>();
                String eachSql = "SELECT %s FROM " + firstTableName + " WHERE PRIORITY=" + i.priority;
                eachPriorityMap.put(eachSql, firstTableName);
                tableQueue.stream().filter(each -> !each.equals(firstTableName)).forEach(each -> {
                    // where后面=不能有空格，否则解析priority会出错
                    String sql = "SELECT %s FROM " + each + " WHERE PRIORITY=" + i.priority;
                    eachPriorityMap.put(sql, each);
                });
                sqlColumnMap.add(eachPriorityMap);
            }
        }
        return sqlColumnMap;
    }

    /**
     * 添加sql语句，并开始搜索
     */
    private void startSearch() {
        searchCache();
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        AtomicBoolean isSearchFolderDone = new AtomicBoolean();
        cachedThreadPoolUtil.executeTask(() -> {
            searchPriorityFolder();
            searchStartMenu();
            searchDesktop();
            isSearchFolderDone.set(true);
        });
        final long start = System.currentTimeMillis();
        final long timeout = 500;
        while (!isSearchFolderDone.get()) {
            if (System.currentTimeMillis() - start > timeout) {
                break;
            }
            Thread.onSpinWait();
        }
        var taskMap = PrepareSearchInfo.taskMap;
        EventManagement eventManagement = EventManagement.getInstance();
        final int threadNumberPerDisk = Math.max(1, AllConfigs.getInstance().getSearchThreadNumber() / taskMap.size());
        Consumer<ConcurrentLinkedQueue<Runnable>> taskHandler = (taskQueue) -> {
            while (!taskQueue.isEmpty() && eventManagement.notMainExit()) {
                var runnable = taskQueue.poll();
                if (runnable == null)
                    continue;
                runnable.run();
            }
        };
        for (var entry : taskMap.entrySet()) {
            for (int i = 0; i < threadNumberPerDisk; i++) {
                cachedThreadPoolUtil.executeTask(() -> {
                    searchThreadCount.incrementAndGet();
                    var taskQueue = entry.getValue();
                    taskHandler.accept(taskQueue);
                    taskMap.remove(entry.getKey());
                    //自身任务已经完成，开始扫描其他线程的任务
                    boolean hasTask;
                    do {
                        hasTask = false;
                        for (var e : taskMap.entrySet()) {
                            var otherTaskQueue = e.getValue();
                            // 如果hasTask为false，则检查otherTaskQueue是否为空
                            if (!hasTask) {
                                // 不为空则设置hasTask为true
                                hasTask = !otherTaskQueue.isEmpty();
                            }
                            taskHandler.accept(otherTaskQueue);
                            taskMap.remove(e.getKey());
                        }
                    } while (hasTask && eventManagement.notMainExit());
                    searchThreadCount.decrementAndGet();
                });
            }
        }
        waitForTasks();
    }

    /**
     * 生成删除记录sql
     *
     * @param asciiSum ascii
     * @param path     文件路径
     */
    private void addDeleteSqlCommandByAscii(int asciiSum, String path) {
        String command;
        int asciiGroup = asciiSum / 100;
        asciiGroup = Math.min(asciiGroup, Constants.ALL_TABLE_NUM);
        String sql = "DELETE FROM %s where PATH=\"%s\";";
        command = String.format(sql, "list" + asciiGroup, path);
        if (command != null && isCommandNotRepeat(command)) {
            addToCommandQueue(new SQLWithTaskId(command, SqlTaskIds.DELETE_FROM_LIST, String.valueOf(path.charAt(0))));
        }
    }

    /**
     * 生成添加记录sql
     *
     * @param asciiSum ascii
     * @param path     文件路径
     * @param priority 优先级
     */
    private void addAddSqlCommandByAscii(int asciiSum, String path, int priority) {
        String commandTemplate = "INSERT OR IGNORE INTO %s VALUES(%d, \"%s\", %d)";
        int asciiGroup = asciiSum / 100;
        asciiGroup = Math.min(asciiGroup, Constants.ALL_TABLE_NUM);
        String columnName = "list" + asciiGroup;
        String command = String.format(commandTemplate, columnName, asciiSum, path, priority);
        if (command != null && isCommandNotRepeat(command)) {
            addToCommandQueue(new SQLWithTaskId(command, SqlTaskIds.INSERT_TO_LIST, String.valueOf(path.charAt(0))));
        }
    }

    /**
     * 获得文件名
     *
     * @param path 文件路径
     * @return 文件名
     */
    private String getFileName(String path) {
        if (path != null) {
            int index = path.lastIndexOf(File.separator);
            return path.substring(index + 1);
        }
        return "";
    }

    private int getAscIISum(String path) {
        if (path == null || path.isEmpty()) {
            return 0;
        }
        return GetAscII.INSTANCE.getAscII(path);
    }

    /**
     * 检查要删除的文件是否还未添加
     * 防止文件刚添加就被删除
     *
     * @param path 待删除文件路径
     * @return true如果待删除文件已经在数据库中
     */
    private boolean isRemoveFileInDatabase(String path) {
        for (SQLWithTaskId each : commandQueue) {
            if (each.taskId == SqlTaskIds.INSERT_TO_LIST && each.sql.contains(path)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 从数据库中删除记录
     *
     * @param path 文件路径
     */
    private void removeFileFromDatabase(String path) {
        if (path == null || path.isEmpty()) {
            return;
        }
        int asciiSum = getAscIISum(getFileName(path));
        if (isRemoveFileInDatabase(path)) {
            addDeleteSqlCommandByAscii(asciiSum, path);
            int priorityBySuffix = getPriorityBySuffix(getSuffixByPath(path));
            int asciiGroup = asciiSum / 100;
            asciiGroup = Math.min(asciiGroup, Constants.ALL_TABLE_NUM);
            String tableName = "list" + asciiGroup;
            String key = path.charAt(0) + "," + tableName + "," + priorityBySuffix;
            if (AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
                EventManagement.getInstance().putEvent(new GPURemoveRecordEvent(key, path));
            }
            Cache cache = tableCache.get(key);
            if (cache.isCached.get()) {
                if (cache.data.remove(path)) {
                    tableCacheCount.decrementAndGet();
                }
            }
        }
    }

    public HashMap<String, Integer> getPriorityMap() {
        HashMap<String, Integer> map = new HashMap<>();
        priorityMap.forEach(p -> map.put(p.suffix, p.priority));
        return map;
    }

    /**
     * 初始化优先级表
     */
    private void initPriority() {
        priorityMap.clear();
        try (Statement stmt = SQLiteUtil.getStatement("cache");
             ResultSet resultSet = stmt.executeQuery("SELECT * FROM priority order by PRIORITY desc;")) {
            while (resultSet.next()) {
                String suffix = resultSet.getString("SUFFIX");
                String priority = resultSet.getString("PRIORITY");
                try {
                    priorityMap.add(new Pair(suffix, Integer.parseInt(priority)));
                } catch (Exception e) {
                    e.printStackTrace();
                    priorityMap.add(new Pair(suffix, 0));
                }
            }
            priorityMap.add(new Pair("dirPriority", -1));
        } catch (SQLException exception) {
            exception.printStackTrace();
        }
    }

    /**
     * 根据文件后缀获取优先级信息
     *
     * @param suffix 文件后缀名
     * @return 优先级
     */
    @SuppressWarnings("IndexOfReplaceableByContains")
    private int getPriorityBySuffix(String suffix) {
        List<Pair> result = priorityMap.stream().filter(each -> each.suffix.equals(suffix)).collect(Collectors.toList());
        if (result.isEmpty()) {
            if (suffix.indexOf(File.separator) != -1) {
                return getPriorityBySuffix("dirPriority");
            } else {
                return getPriorityBySuffix("defaultPriority");
            }
        } else {
            return result.get(0).priority;
        }
    }

    /**
     * 获取文件后缀
     *
     * @param path 文件路径
     * @return 后缀名
     */
    private String getSuffixByPath(String path) {
        return path.substring(path.lastIndexOf('.') + 1).toLowerCase();
    }

    private void addFileToDatabase(String path) {
        if (path == null || path.isEmpty()) {
            return;
        }
        int asciiSum = getAscIISum(getFileName(path));
        int priorityBySuffix = getPriorityBySuffix(getSuffixByPath(path));
        addAddSqlCommandByAscii(asciiSum, path, priorityBySuffix);
        int asciiGroup = asciiSum / 100;
        asciiGroup = Math.min(asciiGroup, Constants.ALL_TABLE_NUM);
        String tableName = "list" + asciiGroup;
        String key = path.charAt(0) + "," + tableName + "," + priorityBySuffix;
        if (AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            EventManagement.getInstance().putEvent(new GPUAddRecordEvent(key, path));
        }
        Cache cache = tableCache.get(key);
        if (cache.isCacheValid()) {
            if (tableCacheCount.get() < MAX_CACHED_RECORD_NUM) {
                if (cache.data.add(path)) {
                    tableCacheCount.incrementAndGet();
                }
            } else {
                cache.isFileLost.set(true);
            }
        }
    }

    private void addFileToCache(String path) {
        String command = "INSERT OR IGNORE INTO cache(PATH) VALUES(\"" + path + "\");";
        if (isCommandNotRepeat(command)) {
            addToCommandQueue(new SQLWithTaskId(command, SqlTaskIds.INSERT_TO_CACHE, "cache"));
            if (IsDebug.isDebug()) {
                System.out.println("添加" + path + "到缓存");
            }
        }
    }

    private void removeFileFromCache(String path) {
        String command = "DELETE from cache where PATH=" + "\"" + path + "\";";
        if (isCommandNotRepeat(command)) {
            addToCommandQueue(new SQLWithTaskId(command, SqlTaskIds.DELETE_FROM_CACHE, "cache"));
            if (IsDebug.isDebug()) {
                System.out.println("删除" + path + "到缓存");
            }
        }
    }

    /**
     * 发送立即执行所有sql信号
     */
    private void sendExecuteSQLSignal() {
        isExecuteImmediately.set(true);
    }

    /**
     * 执行sql
     */
    @SuppressWarnings("SqlNoDataSourceInspection")
    private void executeAllCommands() {
        synchronized (this) {
            if (!commandQueue.isEmpty()) {
                LinkedHashSet<SQLWithTaskId> tempCommandSet = new LinkedHashSet<>(commandQueue);
                HashMap<String, LinkedList<String>> commandMap = new HashMap<>();
                for (SQLWithTaskId sqlWithTaskId : tempCommandSet) {
                    if (commandMap.containsKey(sqlWithTaskId.key)) {
                        commandMap.get(sqlWithTaskId.key).add(sqlWithTaskId.sql);
                    } else {
                        LinkedList<String> sqls = new LinkedList<>();
                        sqls.add(sqlWithTaskId.sql);
                        commandMap.put(sqlWithTaskId.key, sqls);
                    }
                }
                commandMap.forEach((k, v) -> {
                    Statement stmt = null;
                    try {
                        stmt = SQLiteUtil.getStatement(k);
                        stmt.execute("BEGIN;");
                        for (String sql : v) {
                            if (IsDebug.isDebug()) {
                                System.out.println("----------------------------------------------");
                                System.out.println("执行SQL命令--" + sql);
                                System.out.println("----------------------------------------------");
                            }
                            stmt.execute(sql);
                        }
                    } catch (SQLException exception) {
                        exception.printStackTrace();
                        System.err.println("执行失败：" + v);
                    } finally {
                        if (stmt != null) {
                            try {
                                stmt.execute("COMMIT;");
                                stmt.close();
                            } catch (SQLException e) {
                                e.printStackTrace();
                            }
                        }
                    }
                });
                commandQueue.removeAll(tempCommandSet);
            }
        }
    }

    /**
     * 添加任务到任务列表
     *
     * @param sql 任务
     */
    private void addToCommandQueue(SQLWithTaskId sql) {
        if (commandQueue.size() < MAX_SQL_NUM) {
            if (getStatus() == Constants.Enums.DatabaseStatus.MANUAL_UPDATE) {
                return;
            }
            commandQueue.add(sql);
        } else {
            if (IsDebug.isDebug()) {
                System.err.println("添加sql语句" + sql + "失败，已达到最大上限");
            }
        }
    }

    /**
     * 检查任务是否重复
     *
     * @param sql 任务
     * @return boolean
     */
    private boolean isCommandNotRepeat(String sql) {
        for (SQLWithTaskId each : commandQueue) {
            if (each.sql.equals(sql)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 获取数据库状态
     *
     * @return 数据库状态
     */
    public Constants.Enums.DatabaseStatus getStatus() {
        var currentStatus = status.get();
        switch (currentStatus) {
            case NORMAL:
            case _TEMP:
            case _SHARED_MEMORY:
                return Constants.Enums.DatabaseStatus.NORMAL;
            default:
                return currentStatus;
        }
    }

    public boolean casSetStatus(Constants.Enums.DatabaseStatus expect, Constants.Enums.DatabaseStatus newVal) {
        final long start = System.currentTimeMillis();
        final long timeout = 1000;
        try {
            while (!status.compareAndSet(expect, newVal) && System.currentTimeMillis() - start < timeout) {
                TimeUnit.MILLISECONDS.sleep(1);
            }
        } catch (InterruptedException ignored) {
            // ignore interruptedException
        }
        return status.get() == newVal;
    }

    /**
     * 创建索引
     */
    private void createAllIndex() {
        commandQueue.add(new SQLWithTaskId("CREATE INDEX IF NOT EXISTS cache_index ON cache(PATH);", SqlTaskIds.CREATE_INDEX, "cache"));
        for (String each : RegexUtil.comma.split(AllConfigs.getInstance().getAvailableDisks())) {
            for (int i = 0; i <= Constants.ALL_TABLE_NUM; ++i) {
                String createIndex = "CREATE INDEX IF NOT EXISTS list" + i + "_index ON list" + i + "(PRIORITY);";
                commandQueue.add(new SQLWithTaskId(createIndex, SqlTaskIds.CREATE_INDEX, String.valueOf(each.charAt(0))));
            }
        }
    }

    /**
     * 调用C程序搜索并等待执行完毕
     *
     * @param paths      磁盘信息
     * @param ignorePath 忽略文件夹
     * @throws IOException exception
     */
    private Process searchByUSN(String paths, String ignorePath) throws IOException {
        File usnSearcher = new File("user/fileSearcherUSN.exe");
        String absPath = usnSearcher.getAbsolutePath();
        String start = absPath.substring(0, 2);
        String end = "\"" + absPath.substring(2) + "\"";
        File database = new File("data");
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/MFTSearchInfo.dat"), StandardCharsets.UTF_8))) {
            buffW.write(paths);
            buffW.newLine();
            buffW.write(database.getAbsolutePath());
            buffW.newLine();
            buffW.write(ignorePath);
        }
        return Runtime.getRuntime().exec(new String[]{"cmd.exe", "/c", start + end}, null, new File("user"));
    }

    /**
     * 检查索引数据库大小
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private void checkDbFileSize(boolean isDropPrevious) {
        String databaseCreateTimeFileName = "user/databaseCreateTime.dat";
        HashMap<String, String> databaseCreateTimeMap = new HashMap<>();
        String[] disks = RegexUtil.comma.split(AllConfigs.getInstance().getAvailableDisks());
        LocalDate now = LocalDate.now();
        //从文件中读取每个数据库的创建时间
        StringBuilder stringBuilder = new StringBuilder();
        Gson gson = GsonUtil.INSTANCE.getGson();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(databaseCreateTimeFileName), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                stringBuilder.append(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            for (String disk : disks) {
                databaseCreateTimeMap.put(disk, now.toString());
            }
            Map map = gson.fromJson(stringBuilder.toString(), Map.class);
            if (map != null) {
                //从文件中读取每个数据库的创建时间
                map.forEach((disk, createTime) -> databaseCreateTimeMap.put((String) disk, (String) createTime));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        final long maxDatabaseSize = 8L * 1024 * 1024 * 100;
        for (String eachDisk : disks) {
            String name = eachDisk.charAt(0) + ".db";
            try {
                Path diskDatabaseFile = Path.of("data/" + name);
                long length = Files.size(diskDatabaseFile);
                if (length > maxDatabaseSize || Period.between(LocalDate.parse(databaseCreateTimeMap.get(eachDisk)), now).getDays() > 5 || isDropPrevious) {
                    if (IsDebug.isDebug()) {
                        System.out.println("当前文件" + name + "已删除");
                    }
                    //更新创建时间
                    databaseCreateTimeMap.put(eachDisk, now.toString());
                    Files.delete(diskDatabaseFile);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        String toJson = gson.toJson(databaseCreateTimeMap);
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(databaseCreateTimeFileName), StandardCharsets.UTF_8))) {
            writer.write(toJson);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void closeSharedMemory() throws IOException {
        File closeSharedMemory = new File("tmp", "closeSharedMemory");
        if (closeSharedMemory.exists()) {
            return;
        }
        if (!closeSharedMemory.createNewFile()) {
            throw new RuntimeException("closeSharedMemory创建失败");
        }
    }

    /**
     * 当软件空闲时将共享内存关闭
     */
    private void closeSharedMemoryOnIdle() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                SearchBar searchBar = SearchBar.getInstance();
                while (isSharedMemoryCreated.get()) {
                    if (status.get() == Constants.Enums.DatabaseStatus.NORMAL && !searchBar.isVisible()) {
                        isSharedMemoryCreated.set(false);
                        ResultPipe.INSTANCE.closeAllSharedMemory();
                        closeSharedMemory();
                        if (IsDebug.isDebug()) {
                            System.out.println("已关闭共享内存");
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(250);
                }
            } catch (InterruptedException | IOException e) {
                e.printStackTrace();
            }
        });
    }

    private void stopSearch() {
        shouldStopSearch.set(true);
    }

    private void executeAllSQLAndWait(@SuppressWarnings("SameParameterValue") int timeoutMills) {// 等待剩余的sql全部执行完成
        try {
            final long time = System.currentTimeMillis();
            // 将在队列中的sql全部执行并等待搜索线程全部完成
            System.out.println("等待所有sql执行完成，并且退出搜索");
            while (searchThreadCount.get() != 0 || !commandQueue.isEmpty()) {
                sendExecuteSQLSignal();
                TimeUnit.MILLISECONDS.sleep(10);
                if (System.currentTimeMillis() - time > timeoutMills) {
                    System.out.println("等待超时");
                    break;
                }
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void waitForSearchAndSwitchDatabase(Process searchByUsn) {
        final long start = System.currentTimeMillis();
        final long timeLimit = 10 * 60 * 1000;
        // 阻塞等待程序将共享内存配置完成
        try {
            while (!ResultPipe.INSTANCE.isComplete() && ProcessUtil.isProcessExist("fileSearcherUSN.exe")) {
                if (System.currentTimeMillis() - start > timeLimit) {
                    System.out.println("等待共享内存超时");
                    break;
                }
                TimeUnit.SECONDS.sleep(1);
            }
        } catch (InterruptedException | IOException e) {
            e.printStackTrace();
        }
        casSetStatus(Constants.Enums.DatabaseStatus._TEMP, Constants.Enums.DatabaseStatus._SHARED_MEMORY);
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                while (!ResultPipe.INSTANCE.isDatabaseComplete() && eventManagement.notMainExit()) {
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (InterruptedException ignored) {
                // ignore interrupt exception
            }
            stopSearch();
            try {
                final long startWaitingTime = System.currentTimeMillis();
                //等待所有搜索线程结束，最多等待1分钟
                while (searchThreadCount.get() != 0 && System.currentTimeMillis() - startWaitingTime < 60 * 1000) {
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException ignored) {
                // ignore interrupt exception
            }
            SQLiteUtil.closeAll();
            invalidateAllCache();
            SQLiteUtil.initAllConnections();
            createAllIndex();
            waitForCommandSet(SqlTaskIds.CREATE_INDEX);
            //空闲时关闭共享内存
            closeSharedMemoryOnIdle();
            // 搜索完成，更新isDatabaseUpdated标志，结束UpdateDatabaseEvent事件等待
            isDatabaseUpdated.set(true);
            //重新初始化priority
            initPriority();
            // 搜索完成并写入数据库后，重新建立数据库连接
            try {
                ProcessUtil.waitForProcess("fileSearcherUSN.exe", 1000);
                readSearchUsnOutput(searchByUsn);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    private static void readSearchUsnOutput(Process searchByUsn) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(searchByUsn.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println("fileSearcherUSN: " + line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(searchByUsn.getErrorStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.err.println("fileSearcherUSN: " + line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 关闭数据库连接并更新数据库
     *
     * @param ignorePath     忽略文件夹
     * @param isDropPrevious 是否删除之前的记录
     */
    private boolean updateLists(String ignorePath, boolean isDropPrevious) throws IOException, InterruptedException {
        if (getStatus() == Constants.Enums.DatabaseStatus.MANUAL_UPDATE || ProcessUtil.isProcessExist("fileSearcherUSN.exe")) {
            System.out.println("already searching");
            return true;
        }
        // 复制数据库到tmp
        SQLiteUtil.copyDatabases("data", "tmp");
        if (!casSetStatus(status.get(), Constants.Enums.DatabaseStatus.MANUAL_UPDATE)) {
            throw new RuntimeException("databaseService status设置MANUAL UPDATE状态失败");
        }
        // 停止搜索
        stopSearch();
        executeAllSQLAndWait(3000);

        if (!isDropPrevious) {
            //执行VACUUM命令
            for (String eachDisk : RegexUtil.comma.split(AllConfigs.getInstance().getAvailableDisks())) {
                try (Statement stmt = SQLiteUtil.getStatement(String.valueOf(eachDisk.charAt(0)))) {
                    stmt.execute("VACUUM;");
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        }

        SQLiteUtil.closeAll();
        SQLiteUtil.initAllConnections("tmp");
        if (IsDebug.isDebug()) {
            System.out.println("成功切换到临时数据库");
        }
        if (!casSetStatus(status.get(), Constants.Enums.DatabaseStatus._TEMP)) {
            //恢复data目录的数据库
            SQLiteUtil.closeAll();
            SQLiteUtil.initAllConnections();
            casSetStatus(status.get(), Constants.Enums.DatabaseStatus.NORMAL);
            throw new RuntimeException("databaseService status设置TEMP状态失败");
        }
        // 检查数据库文件大小，过大则删除
        checkDbFileSize(isDropPrevious);

        isSharedMemoryCreated.set(true);
        Process searchByUSN = null;
        try {
            File closeSharedMemory = new File("tmp", "closeSharedMemory");
            if (closeSharedMemory.exists() && !closeSharedMemory.delete()) {
                throw new RuntimeException("删除共享内存标志失败");
            }
            // 创建搜索进程并等待
            searchByUSN = searchByUSN(AllConfigs.getInstance().getAvailableDisks(), ignorePath.toLowerCase());
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        } finally {
            waitForSearchAndSwitchDatabase(searchByUSN);
        }
        return true;
    }

    /**
      等待sql任务执行

      @param taskId 任务id
     */
    private void waitForCommandSet(@SuppressWarnings("SameParameterValue") SqlTaskIds taskId) {
        try {
            EventManagement eventManagement = EventManagement.getInstance();
            long tmpStartTime = System.currentTimeMillis();
            while (eventManagement.notMainExit()) {
                //等待
                if (System.currentTimeMillis() - tmpStartTime > 60 * 1000) {
                    System.err.println("等待SQL语句任务" + taskId + "处理超时");
                    break;
                }
                //判断commandSet中是否还有taskId存在
                if (!isTaskExistInCommandSet(taskId)) {
                    break;
                }
                TimeUnit.MILLISECONDS.sleep(10);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * 获取缓存数量
     *
     * @return cache num
     */
    public int getDatabaseCacheNum() {
        return databaseCacheNum.get();
    }

    private boolean isTaskExistInCommandSet(SqlTaskIds taskId) {
        for (SQLWithTaskId tasks : commandQueue) {
            if (tasks.taskId == taskId) {
                return true;
            }
        }
        return false;
    }

    private HashMap<String, Integer> queryAllWeights() {
        HashMap<String, Integer> stringIntegerHashMap = new HashMap<>();
        try (Statement pStmt = SQLiteUtil.getStatement("weight");
             ResultSet resultSet = pStmt.executeQuery("SELECT TABLE_NAME, TABLE_WEIGHT FROM weight;")) {
            while (resultSet.next()) {
                String tableName = resultSet.getString("TABLE_NAME");
                int weight = resultSet.getInt("TABLE_WEIGHT");
                stringIntegerHashMap.put(tableName, weight);
            }
        } catch (SQLException exception) {
            exception.printStackTrace();
        }
        return stringIntegerHashMap;
    }

    private void checkTimeAndSendExecuteSqlSignalThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            // 时间检测线程
            final long updateTimeLimit = AllConfigs.getInstance().getUpdateTimeLimit();
            final long timeout = Constants.CLOSE_DATABASE_TIMEOUT_MILLS - 30 * 1000;
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.notMainExit()) {
                    if ((getStatus() == Constants.Enums.DatabaseStatus.NORMAL && System.currentTimeMillis() - startSearchTimeMills.get() < timeout) ||
                            (getStatus() == Constants.Enums.DatabaseStatus.NORMAL && commandQueue.size() > 100)) {
                        sendExecuteSQLSignal();
                    }
                    TimeUnit.SECONDS.sleep(updateTimeLimit);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * 检查是否拥有管理员权限
     *
     * @return boolean
     */
    private static boolean isAdmin() {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("cmd.exe");
            Process process = processBuilder.start();
            try (PrintStream printStream = new PrintStream(process.getOutputStream(), true)) {
                try (Scanner scanner = new Scanner(process.getInputStream())) {
                    printStream.println("@echo off");
                    printStream.println(">nul 2>&1 \"%SYSTEMROOT%\\system32\\cacls.exe\" \"%SYSTEMROOT%\\system32\\config\\system\"");
                    printStream.println("echo %errorlevel%");
                    boolean printedErrorLevel = false;
                    while (true) {
                        String nextLine = scanner.nextLine();
                        if (printedErrorLevel) {
                            int errorLevel = Integer.parseInt(nextLine);
                            scanner.close();
                            return errorLevel == 0;
                        } else if ("echo %errorlevel%".equals(nextLine)) {
                            printedErrorLevel = true;
                        }
                    }
                }
            }
        } catch (IOException e) {
            return false;
        }
    }

    private static void prepareSearchKeywords(Supplier<String> searchTextSupplier, Supplier<String[]> searchCaseSupplier, Supplier<String[]> keywordsSupplier) {
        searchText = searchTextSupplier.get();
        searchCase = searchCaseSupplier.get();
        isIgnoreCase = searchCase == null ||
                Arrays.stream(searchCase).noneMatch(s -> s.equals(PathMatchUtil.SearchCase.CASE));
        String[] _keywords = keywordsSupplier.get();
        keywords = new String[_keywords.length];
        keywordsLowerCase = new String[_keywords.length];
        isKeywordPath = new boolean[_keywords.length];
        // 对keywords进行处理
        for (int i = 0; i < _keywords.length; ++i) {
            String eachKeyword = _keywords[i];
            // 当keywords为空，初始化为默认值
            if (eachKeyword == null || eachKeyword.isEmpty()) {
                isKeywordPath[i] = false;
                keywords[i] = "";
                keywordsLowerCase[i] = "";
                continue;
            }
            final char _firstChar = eachKeyword.charAt(0);
            final boolean isPath = _firstChar == '/' || _firstChar == File.separatorChar;
            if (isPath) {
                // 当关键字为"test;/C:/test"时，分割出来为["test", "/C:/test"]，所以需要去掉 /C:/test 前面的 "/"
                if (eachKeyword.contains(":")) {
                    eachKeyword = eachKeyword.substring(1);
                }
                // 将 / 以及 \ 替换为空字符串，以便模糊匹配文件夹路径
                Matcher matcher = RegexUtil.getPattern("/|\\\\", 0).matcher(eachKeyword);
                eachKeyword = matcher.replaceAll(Matcher.quoteReplacement(""));
            }
            isKeywordPath[i] = isPath;
            keywords[i] = eachKeyword;
            keywordsLowerCase[i] = eachKeyword.toLowerCase();
        }
    }

    @EventListener(listenClass = SearchBarReadyEvent.class)
    private static void searchBarVisibleListener(Event event) {
        SQLiteUtil.openAllConnection();
        getInstance().sendExecuteSQLSignal();
    }

    @EventRegister(registerClass = CheckDatabaseEmptyEvent.class)
    private static void checkDatabaseEmpty(Event event) {
        boolean databaseDamaged = SQLiteUtil.isDatabaseDamaged();
        event.setReturnValue(databaseDamaged);
    }

    @EventRegister(registerClass = InitializeDatabaseEvent.class)
    private static void initAllDatabases(Event event) {
        SQLiteUtil.initAllConnections();
    }

    @EventRegister(registerClass = StartMonitorDiskEvent.class)
    private static void startMonitorDiskEvent(Event event) {
        startMonitorDisk();
    }

    private static class PrepareSearchInfo {
        static AtomicBoolean isGpuThreadRunning = new AtomicBoolean(false);
        //taskMap任务队列，key为磁盘盘符，value为任务
        static ConcurrentHashMap<String, ConcurrentLinkedQueue<Runnable>> taskMap;
        static Bit taskStatus = new Bit(new byte[]{0});
        static Bit allTaskStatus = new Bit(new byte[]{0});
        static AtomicBoolean isPreparing = new AtomicBoolean(false);

        private static ConcurrentHashMap<String, Set<String>> prepareSearchTasks() {
            DatabaseService databaseService = DatabaseService.getInstance();
            //每个priority用一个线程，每一个后缀名对应一个优先级
            //按照优先级排列，key是sql和表名的对应，value是容器
            var nonFormattedSql = databaseService.getNonFormattedSqlFromTableQueue();
            taskStatus = new Bit(new byte[]{0});
            allTaskStatus = new Bit(new byte[]{0});
            //添加搜索任务到队列
            return databaseService.addSearchTasks(nonFormattedSql);
        }
    }

    @EventListener(listenClass = SetConfigsEvent.class)
    private static void setGpuDevice(Event event) {
        SetConfigsEvent setConfigsEvent = (SetConfigsEvent) event;
        if (!GPUAccelerator.INSTANCE.setDevice(setConfigsEvent.getConfigs().getGpuDevice())) {
            System.err.println("gpu设备" + setConfigsEvent.getConfigs().getGpuDevice() + "无效");
        }
    }

    @EventRegister(registerClass = PrepareSearchEvent.class)
    private static void prepareSearchEventHandle(Event event) {
        if (PrepareSearchInfo.isPreparing.compareAndSet(false, true)) {
            if (IsDebug.isDebug()) {
                System.out.println("进行预搜索并添加搜索任务");
            }
            DatabaseService databaseService = DatabaseService.getInstance();
            final long startWaiting = System.currentTimeMillis();
            final long timeout = 3000;
            while (databaseService.getStatus() != Constants.Enums.DatabaseStatus.NORMAL && System.currentTimeMillis() - startWaiting < timeout) {
                Thread.onSpinWait();
            }
            prepareSearch((PrepareSearchEvent) event);
            PrepareSearchInfo.isPreparing.set(false);
            event.setReturnValue(databaseService.tempResults);
        }
    }

    @EventRegister(registerClass = StartSearchEvent.class)
    private static void startSearchEvent(Event event) {
        DatabaseService databaseService = getInstance();
        final long startWaiting = System.currentTimeMillis();
        final long timeout = 3000;
        while (databaseService.searchThreadCount.get() != 0 ||
                databaseService.getStatus() != Constants.Enums.DatabaseStatus.NORMAL ||
                PrepareSearchInfo.isPreparing.get()) {
            if (System.currentTimeMillis() - startWaiting > timeout) {
                System.out.println("等待上次搜索结束超时");
                break;
            }
            Thread.onSpinWait();
        }
        if (PrepareSearchInfo.taskMap == null || PrepareSearchInfo.taskMap.isEmpty()) {
            prepareSearch((StartSearchEvent) event);
        }
        databaseService.shouldStopSearch.set(false);
        //tempResultsForEvent和PrepareSearchInfo.containerMap在searchDone()中发送SearchDoneEvent后就已经清除，所以不需要清除，只清理tempResults
        //启动搜索线程
        CachedThreadPoolUtil.getInstance().executeTask(databaseService::startSearch);
        databaseService.startSearchTimeMills.set(System.currentTimeMillis());
        event.setReturnValue(databaseService.tempResults);
    }

    /**
     * 防止PrepareSearchEvent未完成正在创建taskMap时，StartSearchEvent开始，导致两个线程同时添加任务（极端情况，一般不会出现）
     *
     * @param startSearchEvent startSearchEvent
     */
    private static synchronized void prepareSearch(StartSearchEvent startSearchEvent) {
        PrepareSearchInfo.taskMap = new ConcurrentHashMap<>();
        prepareSearchKeywords(startSearchEvent.searchText, startSearchEvent.searchCase, startSearchEvent.keywords);
        DatabaseService databaseService = getInstance();
        databaseService.tempResults = new ConcurrentLinkedQueue<>();
        databaseService.tempResultsForEvent = new ConcurrentSkipListSet<>();
        var gpuSearchContainer = PrepareSearchInfo.prepareSearchTasks();
        if (AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            // 退出上一次搜索
            GPUAccelerator.INSTANCE.stopCollectResults();
            long start = System.currentTimeMillis();
            while (!PrepareSearchInfo.isGpuThreadRunning.compareAndSet(false, true)) {
                if (System.currentTimeMillis() - start > 3000) {
                    System.out.println("等待上一次gpu加速完成超时");
                    break;
                }
                Thread.onSpinWait();
            }
            CachedThreadPoolUtil.getInstance().executeTask(() -> {
                // 开始进行搜索
                AtomicInteger resultCount = new AtomicInteger();
                GPUAccelerator.INSTANCE.resetAllResultStatus();
                PrepareSearchInfo.isGpuThreadRunning.set(true);
                EventManagement eventManagement = EventManagement.getInstance();
                GPUAccelerator.INSTANCE.match(searchCase,
                        isIgnoreCase,
                        searchText,
                        keywords,
                        keywordsLowerCase,
                        isKeywordPath,
                        MAX_RESULTS,
                        (key, path) -> {
                            if (gpuSearchContainer.containsKey(key) && eventManagement.notMainExit()) {
                                Set<String> gpuContainer = gpuSearchContainer.get(key);
                                if (!databaseService.tempResultsForEvent.contains(path) && gpuContainer.add(path)) {
                                    if (resultCount.incrementAndGet() >= MAX_RESULTS) {
                                        databaseService.stopSearch();
                                    }
                                }
                            }
                        });
                PrepareSearchInfo.isGpuThreadRunning.set(false);
            }, false);
        }
    }

    @EventRegister(registerClass = StopSearchEvent.class)
    private static void stopSearchEvent(Event event) {
        DatabaseService databaseService = getInstance();
        databaseService.stopSearch();
    }

    @EventListener(listenClass = BootSystemEvent.class)
    private static void databaseServiceInit(Event event) {
        DatabaseService databaseService = getInstance();
        databaseService.initDatabaseCacheNum();
        databaseService.initPriority();
        databaseService.initTableMap();
        databaseService.prepareDatabaseCache();
        for (String diskPath : RegexUtil.comma.split(AllConfigs.getInstance().getAvailableDisks())) {
            for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
                for (Pair pair : databaseService.priorityMap) {
                    databaseService.tableCache.put(diskPath.charAt(0) + "," + "list" + i + "," + pair.priority, new Cache());
                }
            }
        }
        databaseService.syncFileChangesThread();
        databaseService.checkTimeAndSendExecuteSqlSignalThread();
        databaseService.executeSqlCommandsThread();
        databaseService.saveTableCacheThread();
    }

    @EventRegister(registerClass = AddToCacheEvent.class)
    private static void addToCacheEvent(Event event) {
        DatabaseService databaseService = getInstance();
        String path = ((AddToCacheEvent) event).path;
        databaseService.databaseCacheSet.add(path);
        if (databaseService.status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
            return;
        }
        databaseService.addFileToCache(path);
        databaseService.databaseCacheNum.incrementAndGet();
    }

    @EventRegister(registerClass = DeleteFromCacheEvent.class)
    private static void deleteFromCacheEvent(Event event) {
        DatabaseService databaseService = getInstance();
        String path = ((DeleteFromCacheEvent) event).path;
        databaseService.databaseCacheSet.remove(path);
        if (databaseService.status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
            return;
        }
        databaseService.removeFileFromCache(path);
        databaseService.databaseCacheNum.decrementAndGet();
    }

    @EventRegister(registerClass = UpdateDatabaseEvent.class)
    private static void updateDatabaseEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
            return;
        }
        UpdateDatabaseEvent updateDatabaseEvent = (UpdateDatabaseEvent) event;
        // 在这里设置数据库状态为manual update
        try {
            if (!databaseService.updateLists(AllConfigs.getInstance().getIgnorePath(), updateDatabaseEvent.isDropPrevious)) {
                throw new RuntimeException("search failed");
            }
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    @EventRegister(registerClass = OptimiseDatabaseEvent.class)
    private static void optimiseDatabaseEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
            return;
        }
        if (databaseService.casSetStatus(Constants.Enums.DatabaseStatus.NORMAL, Constants.Enums.DatabaseStatus.VACUUM)) {
            throw new RuntimeException("databaseService status设置VACUUM状态失败");
        }
        //执行VACUUM命令
        for (String eachDisk : RegexUtil.comma.split(AllConfigs.getInstance().getAvailableDisks())) {
            try (Statement stmt = SQLiteUtil.getStatement(String.valueOf(eachDisk.charAt(0)))) {
                stmt.execute("VACUUM;");
            } catch (Exception ex) {
                ex.printStackTrace();
            } finally {
                if (IsDebug.isDebug()) {
                    System.out.println("结束优化");
                }
            }
        }
        if (databaseService.casSetStatus(Constants.Enums.DatabaseStatus.VACUUM, Constants.Enums.DatabaseStatus.NORMAL)) {
            throw new RuntimeException("databaseService status从VACUUM修改为NORMAL失败");
        }
    }

    @EventRegister(registerClass = AddToSuffixPriorityMapEvent.class)
    private static void addToSuffixPriorityMapEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
            return;
        }
        AddToSuffixPriorityMapEvent event1 = (AddToSuffixPriorityMapEvent) event;
        String suffix = event1.suffix;
        int priority = event1.priority;
        databaseService.addToCommandQueue(
                new SQLWithTaskId(String.format("INSERT INTO priority VALUES(\"%s\", %d);", suffix, priority), SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = ClearSuffixPriorityMapEvent.class)
    private static void clearSuffixPriorityMapEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
            return;
        }
        databaseService.addToCommandQueue(new SQLWithTaskId("DELETE FROM priority;", SqlTaskIds.UPDATE_SUFFIX, "cache"));
        databaseService.addToCommandQueue(
                new SQLWithTaskId("INSERT INTO priority VALUES(\"defaultPriority\", 0);", SqlTaskIds.UPDATE_SUFFIX, "cache"));
        databaseService.addToCommandQueue(
                new SQLWithTaskId("INSERT INTO priority VALUES(\"dirPriority\", -1);", SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = DeleteFromSuffixPriorityMapEvent.class)
    private static void deleteFromSuffixPriorityMapEvent(Event event) {
        DeleteFromSuffixPriorityMapEvent delete = (DeleteFromSuffixPriorityMapEvent) event;
        DatabaseService databaseService = getInstance();
        if (databaseService.status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
            return;
        }
        if ("dirPriority".equals(delete.suffix) || "defaultPriority".equals(delete.suffix)) {
            return;
        }
        databaseService.addToCommandQueue(new SQLWithTaskId(String.format("DELETE FROM priority where SUFFIX=\"%s\"", delete.suffix), SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = UpdateSuffixPriorityEvent.class)
    private static void updateSuffixPriorityEvent(Event event) {
        DatabaseService databaseService = DatabaseService.getInstance();
        if (databaseService.status.get() == Constants.Enums.DatabaseStatus._SHARED_MEMORY) {
            return;
        }
        EventManagement eventManagement = EventManagement.getInstance();
        UpdateSuffixPriorityEvent update = (UpdateSuffixPriorityEvent) event;
        String origin = update.originSuffix;
        String newSuffix = update.newSuffix;
        int newNum = update.newPriority;
        eventManagement.putEvent(new DeleteFromSuffixPriorityMapEvent(origin));
        eventManagement.putEvent(new AddToSuffixPriorityMapEvent(newSuffix, newNum));
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void restartEvent(Event event) {
        FileMonitor.INSTANCE.stop_monitor();
        DatabaseService databaseService = getInstance();
        databaseService.executeAllCommands();
        databaseService.stopSearch();
        SQLiteUtil.closeAll();
        if (AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
            GPUAccelerator.INSTANCE.stopCollectResults();
            GPUAccelerator.INSTANCE.release();
        }
        try {
            databaseService.closeSharedMemory();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Data
    private static class SQLWithTaskId {
        private final String sql;
        private final SqlTaskIds taskId;
        private final String key;
    }

    private enum SqlTaskIds {
        DELETE_FROM_LIST, DELETE_FROM_CACHE, INSERT_TO_LIST, INSERT_TO_CACHE,
        CREATE_INDEX, CREATE_TABLE, DROP_TABLE, DROP_INDEX, UPDATE_SUFFIX, UPDATE_WEIGHT
    }

    @SuppressWarnings("unused")
    private static class GPUCacheService {
        private static final ConcurrentLinkedQueue<String> invalidCacheKeys = new ConcurrentLinkedQueue<>();
        private static final ConcurrentHashMap<String, Set<String>> recordsToAdd = new ConcurrentHashMap<>();
        private static final ConcurrentHashMap<String, Set<String>> recordsToRemove = new ConcurrentHashMap<>();

        private static void clearInvalidCacheThread() {
            //检测缓存是否有效并删除缓存
            CachedThreadPoolUtil.getInstance().executeTask(() -> {
                EventManagement eventManagement = EventManagement.getInstance();
                DatabaseService databaseService = DatabaseService.getInstance();
                long startCheckInvalidCacheTime = System.currentTimeMillis();
                final long checkInterval = 10 * 60 * 1000; // 10min
                try {
                    while (eventManagement.notMainExit()) {
                        if (System.currentTimeMillis() - startCheckInvalidCacheTime > checkInterval && !GetHandle.INSTANCE.isForegroundFullscreen()) {
                            startCheckInvalidCacheTime = System.currentTimeMillis();
                            String eachKey;
                            while (eventManagement.notMainExit() && (eachKey = invalidCacheKeys.poll()) != null) {
                                GPUAccelerator.INSTANCE.clearCache(eachKey);
                            }
                        }
                        TimeUnit.MILLISECONDS.sleep(100);
                    }
                } catch (InterruptedException ignored) {
                    //ignored
                }
            });
        }

        private static void execWorkQueueThread() {
            CachedThreadPoolUtil.getInstance().executeTask(() -> {
                EventManagement eventManagement = EventManagement.getInstance();
                DatabaseService databaseService = DatabaseService.getInstance();
                try {
                    final int removeRecordsThreshold = 20;
                    while (eventManagement.notMainExit()) {
                        if (databaseService.getStatus() == Constants.Enums.DatabaseStatus.NORMAL &&
                                !GetHandle.INSTANCE.isForegroundFullscreen() &&
                                (!recordsToAdd.isEmpty() || !recordsToRemove.isEmpty())) {
                            for (var entry : recordsToAdd.entrySet()) {
                                String k = entry.getKey();
                                Set<String> container = entry.getValue();
                                if (container.isEmpty()) continue;
                                var records = container.toArray();
                                GPUAccelerator.INSTANCE.addRecordsToCache(k, records);
                                for (Object record : records) {
                                    container.remove((String) record);
                                }
                            }
                            for (var entry : recordsToRemove.entrySet()) {
                                String key = entry.getKey();
                                Set<String> container = entry.getValue();
                                if (container.size() < removeRecordsThreshold) continue;
                                var records = container.toArray();
                                GPUAccelerator.INSTANCE.removeRecordsFromCache(key, records);
                                for (Object record : records) {
                                    container.remove((String) record);
                                }
                            }
                        }
                        TimeUnit.SECONDS.sleep(1);
                    }
                } catch (InterruptedException ignored) {
                    // ignored
                }
            });
        }

        private static void addRecord(String key, String fileRecord) {
            if (GPUAccelerator.INSTANCE.isCacheExist(key) && !GPUAccelerator.INSTANCE.isCacheValid(key)) {
                invalidCacheKeys.add(key);
            } else {
                Set<String> container;
                if (recordsToAdd.containsKey(key)) {
                    container = recordsToAdd.get(key);
                    container.add(fileRecord);
                } else {
                    container = ConcurrentHashMap.newKeySet();
                    container.add(fileRecord);
                    recordsToAdd.put(key, container);
                }
            }
        }

        private static void removeRecord(String key, String fileRecord) {
            Set<String> container;
            if (recordsToRemove.containsKey(key)) {
                container = recordsToRemove.get(key);
                container.add(fileRecord);
            } else {
                container = ConcurrentHashMap.newKeySet();
                container.add(fileRecord);
                recordsToRemove.put(key, container);
            }
        }

        @EventListener(listenClass = BootSystemEvent.class)
        private static void startThread(Event event) {
            if (!AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
                return;
            }
            clearInvalidCacheThread();
            //向gpu缓存添加或删除记录线程
            execWorkQueueThread();
        }

        @EventRegister(registerClass = GPUAddRecordEvent.class)
        private static void addToGPUMemory(Event event) {
            if (!AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
                return;
            }
            GPUAddRecordEvent gpuAddRecordEvent = (GPUAddRecordEvent) event;
            addRecord(gpuAddRecordEvent.key, gpuAddRecordEvent.record);
        }

        @EventRegister(registerClass = GPURemoveRecordEvent.class)
        private static void removeFromGPUMemory(Event event) {
            if (!AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
                return;
            }
            GPURemoveRecordEvent gpuRemoveRecordEvent = (GPURemoveRecordEvent) event;
            removeRecord(gpuRemoveRecordEvent.key, gpuRemoveRecordEvent.record);
        }

        @EventRegister(registerClass = GPUClearCacheEvent.class)
        private static void clearCacheGPU(Event event) {
            if (!AllConfigs.getInstance().isGPUAcceleratorEnabled()) {
                return;
            }
            GPUAccelerator.INSTANCE.clearAllCache();
        }
    }
}

