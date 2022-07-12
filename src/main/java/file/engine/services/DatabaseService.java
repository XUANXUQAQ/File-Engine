package file.engine.services;

import com.google.gson.Gson;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.dllInterface.*;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BootSystemEvent;
import file.engine.event.handler.impl.database.*;
import file.engine.event.handler.impl.frame.searchBar.SearchBarReadyEvent;
import file.engine.event.handler.impl.monitor.disk.StartMonitorDiskEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.frames.SearchBar;
import file.engine.utils.*;
import file.engine.utils.bit.Bit;
import file.engine.utils.gson.GsonUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDate;
import java.time.Period;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.stream.Collectors;

public class DatabaseService {
    private static final int MAX_SQL_NUM = 5000;
    private final ConcurrentLinkedQueue<SQLWithTaskId> commandQueue = new ConcurrentLinkedQueue<>();
    private volatile Constants.Enums.DatabaseStatus status = Constants.Enums.DatabaseStatus.NORMAL;
    private final AtomicBoolean isExecuteImmediately = new AtomicBoolean(false);
    private final AtomicInteger cacheNum = new AtomicInteger(0);
    private final Set<TableNameWeightInfo> tableSet;    //保存从0-40数据库的表，使用频率和名字对应，使经常使用的表最快被搜索到
    private final AtomicBoolean isDatabaseUpdated = new AtomicBoolean(false);
    private final AtomicBoolean isReadFromSharedMemory = new AtomicBoolean(false);
    private final @Getter
    ConcurrentLinkedQueue<String> tempResults;  //在优先文件夹和数据库cache未搜索完时暂时保存结果，搜索完后会立即被转移到listResults
    private final AtomicBoolean isSearchStopped = new AtomicBoolean(true);
    private final ConcurrentLinkedQueue<Pair> priorityMap = new ConcurrentLinkedQueue<>();
    private final ConcurrentHashMap<String, Cache> tableCache = new ConcurrentHashMap<>();
    private final AtomicInteger cacheCount = new AtomicInteger();
    private volatile String[] searchCase;
    private volatile boolean isIgnoreCase;
    private volatile String searchText;
    private volatile String[] keywords;
    private volatile String[] keywordsLowerCase;
    private volatile boolean[] isKeywordPath;
    private final AtomicBoolean isSharedMemoryCreated = new AtomicBoolean(false);
    private final ConcurrentSkipListSet<String> cacheSet = new ConcurrentSkipListSet<>();
    private final AtomicInteger searchThreadCount = new AtomicInteger(0);
    private final AtomicLong startSearchTimeMills = new AtomicLong(0);
    private static final int MAX_TEMP_QUERY_RESULT_CACHE = 4096;
    private static final int MAX_CACHED_RECORD_NUM = 10240 * 5;

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
        tableSet = ConcurrentHashMap.newKeySet();
        tempResults = new ConcurrentLinkedQueue<>();
    }

    private void invalidateAllCache() {
        tableCache.values().forEach(each -> {
            each.isCached.set(false);
            each.data = null;
        });
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
        commandQueue.add(new SQLWithTaskId(format, SqlTaskIds.UPDATE_WEIGHT, "weight"));
        if (IsDebug.isDebug()) {
            System.err.println("已更新" + tableName + "权重, 之前为" + origin + "***增加了" + weight);
        }
    }

    /**
     * 获取数据库缓存条目数量，用于判断软件是否还能继续写入缓存
     */
    private void initCacheNum() {
        try (Statement stmt = SQLiteUtil.getStatement("cache");
             ResultSet resultSet = stmt.executeQuery("SELECT COUNT(PATH) FROM cache;")) {
            cacheNum.set(resultSet.getInt(1));
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
                            if (!isReadFromSharedMemory.get()) {
                                executeAllCommands();
                            }
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
     * 读取磁盘监控信息并发送删除sql线程
     */
    private void deleteRecordsFromDatabaseThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try (BufferedReader readerRemove =
                         new BufferedReader(new InputStreamReader(
                                 new FileInputStream(new File("tmp").getAbsolutePath() + File.separator + "fileRemoved.txt"),
                                 StandardCharsets.UTF_8))) {
                String tmp;
                int fileCount = 0;
                LinkedHashSet<String> deletePaths = new LinkedHashSet<>();
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.notMainExit()) {
                    if (status == Constants.Enums.DatabaseStatus.NORMAL) {
                        while ((tmp = readerRemove.readLine()) != null) {
                            fileCount++;
                            deletePaths.add(tmp);
                            if (fileCount > 3000) {
                                break;
                            }
                        }
                    }
                    if (status == Constants.Enums.DatabaseStatus.NORMAL && !deletePaths.isEmpty()) {
                        if (!isReadFromSharedMemory.get()) {
                            for (String deletePath : deletePaths) {
                                removeFileFromDatabase(deletePath);
                            }
                        }
                        deletePaths.clear();
                        fileCount = 0;
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * 开始监控磁盘文件变化
     */
    private static void startMonitorDisk() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            TranslateService translateService = TranslateService.getInstance();
            String disks = AllConfigs.getInstance().getAvailableDisks();
            String[] splitDisks = RegexUtil.comma.split(disks);
            if (isAdmin()) {
                FileMonitor.INSTANCE.set_output(new File("tmp").getAbsolutePath());
                for (String root : splitDisks) {
                    FileMonitor.INSTANCE.monitor(root);
                }
            } else {
                eventManagement.putEvent(new ShowTaskBarMessageEvent(
                        translateService.getTranslation("Warning"),
                        translateService.getTranslation("Not administrator, file monitoring function is turned off")));
            }
        });
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
        LinkedList<String> listRemainDir = new LinkedList<>();
        for (File each : files) {
            if (isSearchStopped.get()) {
                return;
            }
            checkIsMatchedAndAddToList(each.getAbsolutePath(), null);
            if (each.isDirectory()) {
                listRemainDir.add(each.getAbsolutePath());
            }
        }
        out:
        while (!listRemainDir.isEmpty()) {
            String remain = listRemainDir.poll();
            if (remain == null || remain.isEmpty()) {
                continue;
            }
            File[] allFiles = new File(remain).listFiles();
            if (allFiles == null || allFiles.length == 0) {
                continue;
            }
            for (File each : allFiles) {
                checkIsMatchedAndAddToList(each.getAbsolutePath(), null);
                if (isSearchStopped.get()) {
                    break out;
                }
                if (each.isDirectory()) {
                    listRemainDir.add(each.getAbsolutePath());
                }
            }
        }
    }

    /**
     * 搜索文件夹
     */
    private void searchStartMenu() {
        String startMenu = GetStartMenu.INSTANCE.getStartMenu();
        String[] split = RegexUtil.semicolon.split(startMenu);
        if (split == null || split.length == 0) {
            return;
        }
        for (String s : split) {
            searchFolder(s);
        }
    }

    /**
     * 搜索优先文件夹
     */
    private void searchPriorityFolder() {
        searchFolder(AllConfigs.getInstance().getPriorityFolder());
    }

    private LinkedHashMap<String, Integer> scanDatabaseAndSelectCacheTable(String[] disks,
                                                                           ConcurrentLinkedQueue<String> tableQueueByPriority,
                                                                           Supplier<Boolean> isStopCreateCache) {
        //检查哪些表符合缓存条件，通过表权重依次向下排序
        LinkedHashMap<String, Integer> tableNeedCache = new LinkedHashMap<>();
        for (String diskPath : disks) {
            String disk = String.valueOf(diskPath.charAt(0));
            try (Statement stmt = SQLiteUtil.getStatement(disk)) {
                for (String tableName : tableQueueByPriority) {
                    for (Pair pair : priorityMap) {
                        try (ResultSet resultSet = stmt.executeQuery("SELECT COUNT(*) as num FROM " + tableName + " WHERE PRIORITY=" + pair.priority)) {
                            if (resultSet.next()) {
                                int num = resultSet.getInt("num");
                                if (num > 100 && num < 2000) {
                                    tableNeedCache.put(disk + "," + tableName + "," + pair.priority, num);
                                }
                            }
                            if (isStopCreateCache.get()) {
                                return tableNeedCache;
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

    private void saveTableCacheThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            long startCheckTimeMills = 0;
            SearchBar searchBar = SearchBar.getInstance();
            final Supplier<Boolean> isStopCreateCache =
                    () -> !isSearchStopped.get() ||
                            !eventManagement.notMainExit() ||
                            searchBar.isVisible() ||
                            status == Constants.Enums.DatabaseStatus.MANUAL_UPDATE;
            final int checkTimeInterval = 10 * 60 * 1000;
            try {
                while (eventManagement.notMainExit()) {
                    if (isSearchStopped.get() &&
                            System.currentTimeMillis() - startCheckTimeMills > checkTimeInterval &&
                            !searchBar.isVisible()) {
                        final double memoryUsage = SystemInfoUtil.getMemoryUsage();
                        if (memoryUsage < 0.7) {
                            // 系统内存使用少于70%
                            startCheckTimeMills = System.currentTimeMillis();
                            String availableDisks = AllConfigs.getInstance().getAvailableDisks();
                            String[] disks = RegexUtil.comma.split(availableDisks);
                            ConcurrentLinkedQueue<String> tableQueueByPriority = initTableQueueByPriority();
                            LinkedHashMap<String, Integer> tableNeedCache = scanDatabaseAndSelectCacheTable(disks, tableQueueByPriority, isStopCreateCache);
                            //开始缓存数据库表
                            out:
                            for (Map.Entry<String, Cache> entry : tableCache.entrySet()) {
                                String key = entry.getKey();
                                Cache cache = entry.getValue();
                                if (tableNeedCache.containsKey(key)) {
                                    //当前表可以被缓存
                                    if (cacheCount.get() + tableNeedCache.get(key) < MAX_CACHED_RECORD_NUM - 1000 && !cache.isCacheValid()) {
                                        cache.data = new ConcurrentLinkedQueue<>();
                                        String[] info = RegexUtil.comma.split(key);
                                        try (Statement stmt = SQLiteUtil.getStatement(info[0]);
                                             ResultSet resultSet = stmt.executeQuery("SELECT PATH FROM " + info[1] + " " + "WHERE PRIORITY=" + info[2])) {
                                            while (resultSet.next()) {
                                                if (isStopCreateCache.get()) {
                                                    break out;
                                                }
                                                cache.data.add(resultSet.getString("PATH"));
                                                cacheCount.incrementAndGet();
                                            }
                                        } catch (SQLException e) {
                                            e.printStackTrace();
                                        }
                                        cache.isCached.compareAndSet(cache.isCached.get(), true);
                                        cache.isFileLost.set(false);
                                        System.out.println("添加缓存 " + key);
                                    }
                                } else {
                                    if (cache.isCached.get()) {
                                        cache.isCached.compareAndSet(cache.isCached.get(), false);
                                        int num = cache.data.size();
                                        cacheCount.compareAndSet(cacheCount.get(), cacheCount.get() - num);
                                        cache.data = null;
                                    }
                                }
                            }
                        }
                    }
                    TimeUnit.SECONDS.sleep(1);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * 读取磁盘监控信息并发送添加sql线程
     */
    private void addRecordsToDatabaseThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            //检测文件添加线程
            try (BufferedReader readerAdd =
                         new BufferedReader(new InputStreamReader(
                                 new FileInputStream(new File("tmp").getAbsolutePath() + File.separator + "fileAdded.txt"),
                                 StandardCharsets.UTF_8))) {
                String tmp;
                int fileCount = 0;
                LinkedHashSet<String> addPaths = new LinkedHashSet<>();
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.notMainExit()) {
                    if (status == Constants.Enums.DatabaseStatus.NORMAL) {
                        while ((tmp = readerAdd.readLine()) != null) {
                            fileCount++;
                            addPaths.add(tmp);
                            if (fileCount > 3000) {
                                break;
                            }
                        }
                    }
                    if (status == Constants.Enums.DatabaseStatus.NORMAL && !addPaths.isEmpty()) {
                        if (!isReadFromSharedMemory.get()) {
                            for (String addPath : addPaths) {
                                addFileToDatabase(addPath);
                            }
                        }
                        addPaths.clear();
                        fileCount = 0;
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        });
    }


    /**
     * 将缓存中的文件保存到cacheSet中
     */
    private void prepareCache() {
        String eachLine;
        try (Statement statement = SQLiteUtil.getStatement("cache");
             ResultSet resultSet = statement.executeQuery("SELECT PATH FROM cache;")) {
            while (resultSet.next()) {
                eachLine = resultSet.getString("PATH");
                cacheSet.add(eachLine);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 从缓存中搜索结果并将匹配的放入listResults
     */
    private void searchCache() {
        for (String each : cacheSet) {
            checkIsMatchedAndAddToList(each, null);
        }
    }

    /**
     * * 检查文件路径是否匹配然后加入到列表
     *
     * @param path 文件路径
     * @return true如果匹配成功
     */
    private boolean checkIsMatchedAndAddToList(String path, ConcurrentSkipListSet<String> container) {
        boolean ret = false;
        if (PathMatchUtil.check(path, searchCase, isIgnoreCase, searchText, keywords, keywordsLowerCase, isKeywordPath)) {
            if (Files.exists(Path.of(path))) {
                //字符串匹配通过
                ret = true;
                if (!tempResults.contains(path)) {
                    if (container == null) {
                        tempResults.add(path);
                    } else {
                        container.add(path);
                    }
                }
            } else if (container == null) {
                EventManagement.getInstance().putEvent(new DeleteFromCacheEvent(path));
            }
        }
        return ret;
    }

    /**
     * 搜索数据库并加入到tempQueue中
     *
     * @param sql sql
     */
    private int searchAndAddToTempResults(String sql, ConcurrentSkipListSet<String> container, Statement stmt) {
        int matchedResultCount = 0;
        //结果太多则不再进行搜索
        if (isSearchStopped.get()) {
            return matchedResultCount;
        }
        ArrayList<String> tmpQueryResultsCache;
        try (ResultSet resultSet = stmt.executeQuery(sql)) {
            while (resultSet.next()) {
                int i = 0;
                // 先将结果查询出来，再进行字符串匹配，提高吞吐量
                tmpQueryResultsCache = new ArrayList<>(MAX_TEMP_QUERY_RESULT_CACHE / 2);
                do {
                    tmpQueryResultsCache.add(resultSet.getString("PATH"));
                    ++i;
                } while (resultSet.next() && i < MAX_TEMP_QUERY_RESULT_CACHE);
                //结果太多则不再进行搜索
                //用户重新输入了信息
                if (isSearchStopped.get()) {
                    return matchedResultCount;
                }
                matchedResultCount += tmpQueryResultsCache.stream()
                        .filter(each -> checkIsMatchedAndAddToList(each, container))
                        .count();
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
     *
     * @param containerMap  任务搜索结果存放容器
     * @param allTaskStatus 所有的任务信息
     * @param taskStatus    实时更新的任务完成信息
     */
    private void waitForTaskAndMergeResults(LinkedHashMap<String, ConcurrentSkipListSet<String>> containerMap, Bit allTaskStatus, Bit taskStatus) {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            Bit zero = new Bit(new byte[]{0});
            Bit one = new Bit(new byte[]{1});
            int failedRetryTimes = 0;
            try {
                while (!taskStatus.equals(allTaskStatus) && !isSearchStopped.get()) {
                    //线程状态的记录从第二个位开始，所以初始值为1 0
                    Bit start = new Bit(new byte[]{1, 0});
                    //循环次数，也是下方与运算结束后需要向右偏移的位数
                    int loopCount = 1;
                    //由于任务的耗时不同，如果阻塞时间过长，则跳过该任务，在下次循环中重新拿取结果
                    long waitTime = 0;
                    ConcurrentSkipListSet<String> results;
                    while (start.length() <= allTaskStatus.length() || taskStatus.or(zero).equals(zero)) {
                        if (isSearchStopped.get()) {
                            //用户重新输入，结束当前任务
                            break;
                        }
                        //当线程完成，taskStatus中的位会被设置为1
                        //这时，将taskStatus和start做与运算，然后移到第一位，如果为1，则线程已完成搜索
                        Bit and = taskStatus.and(start);
                        boolean isFailed = System.currentTimeMillis() - waitTime > 300 && waitTime != 0;
                        if (((and).shiftRight(loopCount)).equals(one) || isFailed) {
                            // 阻塞时间过长则跳过
                            waitTime = 0;
                            results = containerMap.get(start.toString());
                            if (results != null) {
                                for (String result : results) {
                                    if (!tempResults.contains(result)) {
                                        tempResults.add(result);
                                    }
                                }
                                if (!isFailed || failedRetryTimes > 5) {
                                    failedRetryTimes = 0;
                                    results.clear();
                                    //将start左移，代表当前任务结束，继续拿下一个任务的结果
                                    start.shiftLeft(1);
                                    loopCount++;
                                } else {
                                    ++failedRetryTimes;
                                }
                            }
                        } else {
                            if (waitTime == 0) {
                                waitTime = System.currentTimeMillis();
                            }
                        }
                        TimeUnit.MILLISECONDS.sleep(10);
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                EventManagement eventManagement = EventManagement.getInstance();
                eventManagement.putEvent(new SearchDoneEvent());
                isSearchStopped.set(true);
            }
        }, false);
    }

    /**
     * 当使用共享内存时，添加搜索任务到队列
     *
     * @param nonFormattedSql sql未被格式化的所有任务
     * @param taskStatus      用于保存任务信息，这是一个通用变量，从第二个位开始，每一个位代表一个任务，当任务完成，该位将被置为1，否则为0，
     *                        例如第一个和第三个任务完成，第二个未完成，则为1010
     * @param allTaskStatus   所有的任务信息，从第二位开始，只要有任务被创建，该为就为1，例如三个任务被创建，则为1110
     * @param containerMap    每个任务搜索出来的结果都会被放到一个属于它自己的一个容器中，该容器保存任务与容器的映射关系
     * @param taskMap         任务
     */
    private void addSearchTasksForSharedMemory(LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> nonFormattedSql,
                                               Bit taskStatus,
                                               Bit allTaskStatus,
                                               LinkedHashMap<String, ConcurrentSkipListSet<String>> containerMap,
                                               ConcurrentHashMap<String, ConcurrentLinkedQueue<Runnable>> taskMap) {
        Bit number = new Bit(new byte[]{1});
        for (String eachDisk : RegexUtil.comma.split(AllConfigs.getInstance().getAvailableDisks())) {
            ConcurrentLinkedQueue<Runnable> runnables = new ConcurrentLinkedQueue<>();
            taskMap.put(eachDisk, runnables);
            for (Map.Entry<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> entry : nonFormattedSql.entrySet()) {
                LinkedHashMap<String, String> commandsMap = entry.getKey();
                ConcurrentSkipListSet<String> container = entry.getValue();
                //为每个任务分配的位，不断左移以不断进行分配
                number.shiftLeft(1);
                Bit currentTaskNum = new Bit(number);
                allTaskStatus.set(allTaskStatus.or(currentTaskNum));
                containerMap.put(currentTaskNum.toString(), container);
                runnables.add(() -> {
                    try {
                        Set<String> sqls = commandsMap.keySet();
                        String[] searchInfo = new String[2];
                        for (String each : sqls) {
                            if (isSearchStopped.get()) {
                                return;
                            }
                            getListAndPriority(each, searchInfo);
                            String listName = searchInfo[0];
                            int priority = Integer.parseInt(searchInfo[1]);
                            String result;
                            for (int count = 0;
                                 !isSearchStopped.get() && ((result = ResultPipe.INSTANCE.getResult(eachDisk.charAt(0), listName, priority, count)) != null);
                                 ++count) {
                                checkIsMatchedAndAddToList(result, container);
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        taskStatus.set(taskStatus.or(currentTaskNum));
                    }
                });
            }
        }
    }

    /**
     * 解析出sql中的list和priority
     *
     * @param sql  sql
     * @param info info[0]存放list，info[1]存放priority，需要保证info长度大于2
     */
    private void getListAndPriority(String sql, @NonNull String[] info) {
        String[] split = RegexUtil.blank.split(sql);
        Arrays.stream(split).filter(each -> !each.isBlank()).forEach(each -> {
            int val;
            if (each.contains("list")) {
                info[0] = each;
            } else if ((val = each.indexOf("=")) != -1) {
                info[1] = each.substring(val + 1);
            }
        });
    }

    private void addSearchTasks(LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> nonFormattedSql,
                                Bit taskStatus,
                                Bit allTaskStatus,
                                LinkedHashMap<String, ConcurrentSkipListSet<String>> containerMap,
                                ConcurrentHashMap<String, ConcurrentLinkedQueue<Runnable>> taskMap,
                                boolean isReadFromSharedMemory) {
        if (isReadFromSharedMemory) {
            addSearchTasksForSharedMemory(nonFormattedSql, taskStatus, allTaskStatus, containerMap, taskMap);
        } else {
            addSearchTasksForDatabase(nonFormattedSql, taskStatus, allTaskStatus, containerMap, taskMap);
        }
    }

    /**
     * 添加搜索任务到队列
     *
     * @param nonFormattedSql sql未被格式化的所有任务
     * @param taskStatus      用于保存任务信息，这是一个通用变量，从第二个位开始，每一个位代表一个任务，当任务完成，该位将被置为1，否则为0，
     *                        例如第一个和第三个任务完成，第二个未完成，则为1010
     * @param allTaskStatus   所有的任务信息，从第二位开始，只要有任务被创建，该为就为1，例如三个任务被创建，则为1110
     * @param containerMap    每个任务搜索出来的结果都会被放到一个属于它自己的一个容器中，该容器保存任务与容器的映射关系
     * @param taskMap         任务
     */
    private void addSearchTasksForDatabase(LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> nonFormattedSql,
                                           Bit taskStatus,
                                           Bit allTaskStatus,
                                           LinkedHashMap<String, ConcurrentSkipListSet<String>> containerMap,
                                           ConcurrentHashMap<String, ConcurrentLinkedQueue<Runnable>> taskMap) {
        Bit number = new Bit(new byte[]{1});
        String availableDisks = AllConfigs.getInstance().getAvailableDisks();
        for (String eachDisk : RegexUtil.comma.split(availableDisks)) {
            ConcurrentLinkedQueue<Runnable> runnables = new ConcurrentLinkedQueue<>();
            taskMap.put(eachDisk, runnables);
            for (Map.Entry<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> entry : nonFormattedSql.entrySet()) {
                LinkedHashMap<String, String> commandsMap = entry.getKey();
                ConcurrentSkipListSet<String> container = entry.getValue();
                //为每个任务分配的位，不断左移以不断进行分配
                number.shiftLeft(1);
                Bit currentTaskNum = new Bit(number);
                allTaskStatus.set(allTaskStatus.or(currentTaskNum));
                containerMap.put(currentTaskNum.toString(), container);
                runnables.add(() -> {
                    String diskStr = String.valueOf(eachDisk.charAt(0));
                    try (Statement stmt = SQLiteUtil.getStatement(diskStr)) {
                        Set<String> sqls = commandsMap.keySet();
                        String[] searchInfo = new String[2];
                        for (String eachSql : sqls) {
                            if (isSearchStopped.get()) {
//                                tableQueue.clear();
                                return;
                            }
                            getListAndPriority(eachSql, searchInfo);
                            String tableName = searchInfo[0];
                            String priority = searchInfo[1];
                            String key = diskStr + "," + tableName + "," + priority;
                            Cache cache = tableCache.get(key);
                            int matchedNum = 0;
                            if (cache.isCacheValid()) {
                                if (IsDebug.isDebug()) {
                                    System.out.println("从缓存中读取 " + key);
                                }
                                matchedNum += cache.data.stream()
                                        .filter(record -> checkIsMatchedAndAddToList(record, container))
                                        .count();
                            } else {
                                //格式化是为了以后的拓展性
                                String formattedSql = String.format(eachSql, "PATH");
                                //当前数据库表中有多少个结果匹配成功
                                matchedNum = searchAndAddToTempResults(
                                        formattedSql,
                                        container,
                                        stmt);
                            }
                            final long weight = Math.min(matchedNum, 5);
                            if (weight != 0L) {
                                //更新表的权重，每次搜索将会按照各个表的权重排序
                                updateTableWeight(commandsMap.get(eachSql), weight);
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    } finally {
                        //执行完后将对应的线程flag设为1
                        taskStatus.set(taskStatus.or(currentTaskNum));
                    }
                });
            }
        }
    }

    /**
     * 生成未格式化的sql
     * 第一个map中key保存未格式化的sql，value保存表名称，第二个map为搜索结果的暂时存储容器
     *
     * @return map
     */
    private LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> getNonFormattedSqlFromTableQueue() {
        if (isDatabaseUpdated.get()) {
            isDatabaseUpdated.set(false);
            initPriority();
        }
        LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> sqlColumnMap = new LinkedHashMap<>();
        if (priorityMap.isEmpty()) {
            return sqlColumnMap;
        }
        ConcurrentLinkedQueue<String> tableQueue = initTableQueueByPriority();
        int asciiSum = 0;
        if (keywords != null) {
            for (String keyword : keywords) {
                int ascII = GetAscII.INSTANCE.getAscII(keyword);
                asciiSum += Math.max(ascII, 0);
            }
        }
        final int asciiGroup = asciiSum / 100;
        String firstTableName = "list" + asciiGroup;
        if (searchCase != null && Arrays.asList(searchCase).contains("d")) {
            LinkedHashMap<String, String> _priorityMap = new LinkedHashMap<>();
            String _sql = "SELECT %s FROM " + firstTableName + " WHERE PRIORITY=" + "-1";
            _priorityMap.put(_sql, firstTableName);
            tableQueue.stream().filter(each -> !each.equals(firstTableName)).forEach(each -> {
                // where后面=不能有空格，否则解析priority会出错
                String sql = "SELECT %s FROM " + each + " WHERE PRIORITY=" + "-1";
                _priorityMap.put(sql, each);
            });
            ConcurrentSkipListSet<String> container;
            container = new ConcurrentSkipListSet<>();
            sqlColumnMap.put(_priorityMap, container);
        } else {
            for (Pair i : priorityMap) {
                LinkedHashMap<String, String> eachPriorityMap = new LinkedHashMap<>();
                String _sql = "SELECT %s FROM " + firstTableName + " WHERE PRIORITY=" + i.priority;
                eachPriorityMap.put(_sql, firstTableName);
                tableQueue.stream().filter(each -> !each.equals(firstTableName)).forEach(each -> {
                    // where后面=不能有空格，否则解析priority会出错
                    String sql = "SELECT %s FROM " + each + " WHERE PRIORITY=" + i.priority;
                    eachPriorityMap.put(sql, each);
                });
                ConcurrentSkipListSet<String> container;
                container = new ConcurrentSkipListSet<>();
                sqlColumnMap.put(eachPriorityMap, container);
            }
        }
        tableQueue.clear();
        return sqlColumnMap;
    }

    /**
     * 添加sql语句，并开始搜索
     */
    private void startSearch() {
        searchCache();
        searchPriorityFolder();
        searchStartMenu();
        //每个priority用一个线程，每一个后缀名对应一个优先级
        //按照优先级排列，key是sql和表名的对应，value是容器
        LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>>
                nonFormattedSql = getNonFormattedSqlFromTableQueue();
        Bit taskStatus = new Bit(new byte[]{0});
        Bit allTaskStatus = new Bit(new byte[]{0});

        LinkedHashMap<String, ConcurrentSkipListSet<String>> containerMap = new LinkedHashMap<>();
        //任务队列
        ConcurrentHashMap<String, ConcurrentLinkedQueue<Runnable>> taskMap = new ConcurrentHashMap<>();
        //添加搜索任务到队列
        addSearchTasks(nonFormattedSql, taskStatus, allTaskStatus, containerMap, taskMap, isReadFromSharedMemory.get());
        if (isSearchStopped.get()) {
            return;
        }
        taskMap.forEach((disk, taskQueue) -> {
            CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
            for (int i = 0; i < 4; i++) {
                cachedThreadPoolUtil.executeTask(() -> {
                    Runnable runnable;
                    while ((runnable = taskQueue.poll()) != null) {
                        try {
                            searchThreadCount.incrementAndGet();
                            runnable.run();
                            if (isSearchStopped.get()) {
                                break;
                            }
                        } finally {
                            searchThreadCount.decrementAndGet();
                        }
                    }
                });
            }
        });
        waitForTaskAndMergeResults(containerMap, allTaskStatus, taskStatus);
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
            addToCommandSet(new SQLWithTaskId(command, SqlTaskIds.DELETE_FROM_LIST, String.valueOf(path.charAt(0))));
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
            addToCommandSet(new SQLWithTaskId(command, SqlTaskIds.INSERT_TO_LIST, String.valueOf(path.charAt(0))));
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
        int asciiSum = getAscIISum(getFileName(path));
        if (isRemoveFileInDatabase(path)) {
            addDeleteSqlCommandByAscii(asciiSum, path);
            int priorityBySuffix = getPriorityBySuffix(getSuffixByPath(path));
            int asciiGroup = asciiSum / 100;
            asciiGroup = Math.min(asciiGroup, Constants.ALL_TABLE_NUM);
            String tableName = "list" + asciiGroup;
            Cache cache = tableCache.get(path.charAt(0) + "," + tableName + "," + priorityBySuffix);
            if (cache.isCached.get()) {
                if (cache.data.remove(path)) {
                    cacheCount.decrementAndGet();
                }
            }
        }
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
        int asciiSum = getAscIISum(getFileName(path));
        int priorityBySuffix = getPriorityBySuffix(getSuffixByPath(path));
        addAddSqlCommandByAscii(asciiSum, path, priorityBySuffix);
        int asciiGroup = asciiSum / 100;
        asciiGroup = Math.min(asciiGroup, Constants.ALL_TABLE_NUM);
        String tableName = "list" + asciiGroup;
        Cache cache = tableCache.get(path.charAt(0) + "," + tableName + "," + priorityBySuffix);
        if (cache.isCacheValid()) {
            if (cacheCount.get() < MAX_CACHED_RECORD_NUM) {
                if (cache.data.add(path)) {
                    cacheCount.incrementAndGet();
                }
            } else {
                cache.isFileLost.set(true);
            }
        }
    }

    private void addFileToCache(String path) {
        String command = "INSERT OR IGNORE INTO cache(PATH) VALUES(\"" + path + "\");";
        if (isCommandNotRepeat(command)) {
            addToCommandSet(new SQLWithTaskId(command, SqlTaskIds.INSERT_TO_CACHE, "cache"));
            if (IsDebug.isDebug()) {
                System.out.println("添加" + path + "到缓存");
            }
        }
    }

    private void removeFileFromCache(String path) {
        String command = "DELETE from cache where PATH=" + "\"" + path + "\";";
        if (isCommandNotRepeat(command)) {
            addToCommandSet(new SQLWithTaskId(command, SqlTaskIds.DELETE_FROM_CACHE, "cache"));
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
    private void addToCommandSet(SQLWithTaskId sql) {
        if (commandQueue.size() < MAX_SQL_NUM) {
            if (status == Constants.Enums.DatabaseStatus.MANUAL_UPDATE) {
                System.err.println("正在搜索中");
                return;
            }
            commandQueue.add(sql);
        } else {
            if (IsDebug.isDebug()) {
                System.err.println("添加sql语句" + sql + "失败，已达到最大上限");
            }
            //立即处理sql语句
//            executeImmediately();
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
     * @return 装填
     */
    public Constants.Enums.DatabaseStatus getStatus() {
        return status;
    }

    /**
     * 设置数据库状态
     *
     * @param status 状态
     */
    private void setStatus(Constants.Enums.DatabaseStatus status) {
        this.status = status;
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
    private void searchByUSN(String paths, String ignorePath) throws IOException {
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
        String command = "cmd.exe /c " + start + end;
        Runtime.getRuntime().exec(command, null, new File("user"));
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
            //从文件中读取每个数据库的创建时间
            map.forEach((disk, createTime) -> databaseCreateTimeMap.put((String) disk, (String) createTime));
        } catch (Exception e) {
            e.printStackTrace();
        }

        for (String eachDisk : disks) {
            String name = eachDisk.charAt(0) + ".db";
            try {
                Path diskDatabaseFile = Path.of("data/" + name);
                long length = Files.size(diskDatabaseFile);
                if (length > 5L * 1024 * 1024 * 100 || Period.between(LocalDate.parse(databaseCreateTimeMap.get(eachDisk)), now).getDays() > 5 || isDropPrevious) {
                    // 大小超过500M
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

    /**
     * 当软件空闲时将共享内存关闭
     */
    private void closeSharedMemoryOnIdle() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                SearchBar searchBar = SearchBar.getInstance();
                while (isSharedMemoryCreated.get()) {
                    if (isSearchStopped.get()) {
                        if (!searchBar.isVisible()) {
                            isSharedMemoryCreated.set(false);
                            ResultPipe.INSTANCE.closeAllSharedMemory();
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(250);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    private void stopSearch() {
        isSearchStopped.set(true);
//        tableQueue.clear();
        tempResults.clear();
    }

    private void executeAllSQLAndWait(@SuppressWarnings("SameParameterValue") int timeoutMills) {// 等待剩余的sql全部执行完成
        sendExecuteSQLSignal();
        try {
            long time = System.currentTimeMillis();
            // 将在队列中的sql全部执行并等待搜索线程全部完成
            while (searchThreadCount.get() != 0 || !commandQueue.isEmpty()) {
                TimeUnit.MILLISECONDS.sleep(10);
                if (System.currentTimeMillis() - time > timeoutMills) {
                    break;
                }
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void waitForSearchAndSwitchDatabase() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            {
                long start = System.currentTimeMillis();
                final long timeLimit = 10 * 60 * 1000;
                // 阻塞等待程序将共享内存配置完成
                try {
                    while (!ResultPipe.INSTANCE.isComplete() && ProcessUtil.isProcessExist("fileSearcherUSN.exe")) {
                        if (System.currentTimeMillis() - start > timeLimit) {
                            break;
                        }
                        TimeUnit.SECONDS.sleep(1);
                    }
                } catch (InterruptedException | IOException e) {
                    e.printStackTrace();
                }
                isReadFromSharedMemory.set(true);
            }
            // 搜索完成并写入数据库后，重新建立数据库连接
            {
                try {
                    ProcessUtil.waitForProcess("fileSearcherUSN.exe", 1000);
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    try {
                        long startWaitingTime = System.currentTimeMillis();
                        //最多等待5分钟
                        while (searchThreadCount.get() != 0 && System.currentTimeMillis() - startWaitingTime < 5 * 60 * 1000) {
                            TimeUnit.MILLISECONDS.sleep(20);
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    SQLiteUtil.closeAll();
                    invalidateAllCache();
                    SQLiteUtil.initAllConnections();
                    // 可能会出错
                    recreateDatabase();
                    createAllIndex();
                    isDatabaseUpdated.set(true);
                    isReadFromSharedMemory.set(false);
                }
            }
        });
    }

    /**
     * 关闭数据库连接并更新数据库
     *
     * @param ignorePath     忽略文件夹
     * @param isDropPrevious 是否删除之前的记录
     */
    @SneakyThrows
    private boolean updateLists(String ignorePath, boolean isDropPrevious) {
        if (status == Constants.Enums.DatabaseStatus.MANUAL_UPDATE || ProcessUtil.isProcessExist("fileSearcherUSN.exe")) {
            throw new RuntimeException("already searching");
        }
        setStatus(Constants.Enums.DatabaseStatus.MANUAL_UPDATE);
        // 停止搜索
        stopSearch();
        executeAllSQLAndWait(3000);
        recreateDatabase();
        waitForCommandSet(SqlTaskIds.CREATE_TABLE);

        SQLiteUtil.closeAll();
        // 复制数据库到tmp
        SQLiteUtil.copyDatabases("data", "tmp");
        SQLiteUtil.initAllConnections("tmp");

        // 检查数据库文件大小，过大则删除
        checkDbFileSize(isDropPrevious);

        isSharedMemoryCreated.set(true);
        //创建检测系统空闲线程，当检测到系统空闲时关闭共享内存
        closeSharedMemoryOnIdle();
        try {
            // 创建搜索进程并等待
            searchByUSN(AllConfigs.getInstance().getAvailableDisks(), ignorePath.toLowerCase());
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        } finally {
            waitForSearchAndSwitchDatabase();
        }
        return true;
    }

    /**
     * 等待sql任务执行
     *
     * @param taskId 任务id
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
    public int getCacheNum() {
        return cacheNum.get();
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

    private void recreateDatabase() {
        commandQueue.clear();
        //删除所有索引
        String[] disks = RegexUtil.comma.split(AllConfigs.getInstance().getAvailableDisks());
        //创建新表
        String sql = "CREATE TABLE IF NOT EXISTS list";
        for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
            String command = sql + i + " " + "(ASCII INT, PATH text unique, PRIORITY INT)" + ";";
            for (String disk : disks) {
                commandQueue.add(new SQLWithTaskId(command, SqlTaskIds.CREATE_TABLE, String.valueOf(disk.charAt(0))));
            }
        }
        sendExecuteSQLSignal();
    }

    private void checkTimeAndSendExecuteSqlSignalThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            // 时间检测线程
            final long updateTimeLimit = AllConfigs.getInstance().getUpdateTimeLimit();
            final long timeout = Constants.CLOSE_DATABASE_TIMEOUT_MILLS - 30 * 1000;
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.notMainExit()) {
                    if ((status == Constants.Enums.DatabaseStatus.NORMAL && System.currentTimeMillis() - startSearchTimeMills.get() < timeout) ||
                            (status == Constants.Enums.DatabaseStatus.NORMAL && commandQueue.size() > 1000)) {
                        if (isSearchStopped.get()) {
                            sendExecuteSQLSignal();
                        }
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
            PrintStream printStream = new PrintStream(process.getOutputStream(), true);
            Scanner scanner = new Scanner(process.getInputStream());
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
        } catch (IOException e) {
            return false;
        }
    }

    @EventListener(listenClass = SearchBarReadyEvent.class)
    private static void searchBarVisibleListener(Event event) {
        getInstance().sendExecuteSQLSignal();
    }

    @EventRegister(registerClass = InitializeDatabaseEvent.class)
    private static void initAllDatabases(Event event) {
        SQLiteUtil.initAllConnections();
    }

    @EventRegister(registerClass = StartMonitorDiskEvent.class)
    private static void startMonitorDiskEvent(Event event) {
        startMonitorDisk();
    }

    @EventRegister(registerClass = StartSearchEvent.class)
    private static void startSearchEvent(Event event) {
        StartSearchEvent startSearchEvent = (StartSearchEvent) event;
        DatabaseService databaseService = getInstance();
        databaseService.searchText = startSearchEvent.searchText.get();
        databaseService.searchCase = startSearchEvent.searchCase.get();
        databaseService.isIgnoreCase = databaseService.searchCase == null || Arrays.stream(databaseService.searchCase).noneMatch(s -> s.equals("case"));
        String[] _keywords = startSearchEvent.keywords.get();
        databaseService.keywords = new String[_keywords.length];
        databaseService.keywordsLowerCase = new String[_keywords.length];
        databaseService.isKeywordPath = new boolean[_keywords.length];
        // 对keywords进行处理
        for (int i = 0; i < _keywords.length; ++i) {
            String eachKeyword = _keywords[i];
            if (eachKeyword == null || eachKeyword.isEmpty()) {
                databaseService.isKeywordPath[i] = false;
                databaseService.keywords[i] = "";
                databaseService.keywordsLowerCase[i] = "";
                continue;
            }
            final char _firstChar = eachKeyword.charAt(0);
            final boolean isPath = _firstChar == '/' || _firstChar == File.separatorChar;
            if (isPath) {
                // 当关键字为"test;/C:/test"时，分割出来为["test", "/C:/test"]
                if (eachKeyword.contains(":")) {
                    eachKeyword = eachKeyword.substring(1);
                }
                // 将 / 替换为 \，Windows系统使用 \ 来分隔文件夹
                Matcher matcher = RegexUtil.getPattern("/|\\\\", 0).matcher(eachKeyword);
                eachKeyword = matcher.replaceAll(Matcher.quoteReplacement(""));
            }
            databaseService.isKeywordPath[i] = isPath;
            databaseService.keywords[i] = eachKeyword;
            databaseService.keywordsLowerCase[i] = eachKeyword.toLowerCase();
        }
        databaseService.isSearchStopped.set(false);
        databaseService.startSearch();
        databaseService.startSearchTimeMills.set(System.currentTimeMillis());
    }

    @EventRegister(registerClass = StopSearchEvent.class)
    private static void stopSearchEvent(Event event) {
        DatabaseService databaseService = getInstance();
        databaseService.stopSearch();
    }

    @EventListener(listenClass = BootSystemEvent.class)
    private static void init(Event event) {
        DatabaseService databaseService = getInstance();
        databaseService.initCacheNum();
        databaseService.initPriority();
        databaseService.addRecordsToDatabaseThread();
        databaseService.deleteRecordsFromDatabaseThread();
        databaseService.checkTimeAndSendExecuteSqlSignalThread();
        databaseService.executeSqlCommandsThread();
        databaseService.initTableMap();
        databaseService.prepareCache();
        for (String diskPath : RegexUtil.comma.split(AllConfigs.getInstance().getAvailableDisks())) {
            for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
                for (Pair pair : databaseService.priorityMap) {
                    databaseService.tableCache.put(diskPath.charAt(0) + "," + "list" + i + "," + pair.priority, new Cache());
                }
            }
        }
        databaseService.saveTableCacheThread();
    }

    @EventRegister(registerClass = AddToCacheEvent.class)
    private static void addToCacheEvent(Event event) {
        DatabaseService databaseService = getInstance();
        String path = ((AddToCacheEvent) event).path;
        databaseService.cacheSet.add(path);
        if (databaseService.isReadFromSharedMemory.get()) {
            return;
        }
        databaseService.addFileToCache(path);
        databaseService.cacheNum.incrementAndGet();
    }

    @EventRegister(registerClass = DeleteFromCacheEvent.class)
    private static void deleteFromCacheEvent(Event event) {
        DatabaseService databaseService = getInstance();
        String path = ((DeleteFromCacheEvent) event).path;
        databaseService.cacheSet.remove(path);
        if (databaseService.isReadFromSharedMemory.get()) {
            return;
        }
        databaseService.removeFileFromCache(path);
        databaseService.cacheNum.decrementAndGet();
    }

    @EventRegister(registerClass = UpdateDatabaseEvent.class)
    private static void updateDatabaseEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.isReadFromSharedMemory.get()) {
            return;
        }
        UpdateDatabaseEvent updateDatabaseEvent = (UpdateDatabaseEvent) event;
        try {
            if (!databaseService.updateLists(AllConfigs.getInstance().getIgnorePath(), updateDatabaseEvent.isDropPrevious)) {
                throw new RuntimeException("search failed");
            }
        } finally {
            databaseService.setStatus(Constants.Enums.DatabaseStatus.NORMAL);
            if (IsDebug.isDebug()) {
                System.out.println("搜索已经可用");
            }
            try {
                while (!databaseService.isDatabaseUpdated.get()) {
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @EventRegister(registerClass = OptimiseDatabaseEvent.class)
    private static void optimiseDatabaseEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.isReadFromSharedMemory.get()) {
            return;
        }
        databaseService.setStatus(Constants.Enums.DatabaseStatus.VACUUM);
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
        databaseService.setStatus(Constants.Enums.DatabaseStatus.NORMAL);
    }

    @EventRegister(registerClass = AddToSuffixPriorityMapEvent.class)
    private static void addToSuffixPriorityMapEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.isReadFromSharedMemory.get()) {
            return;
        }
        AddToSuffixPriorityMapEvent event1 = (AddToSuffixPriorityMapEvent) event;
        String suffix = event1.suffix;
        int priority = event1.priority;
        databaseService.addToCommandSet(
                new SQLWithTaskId(String.format("INSERT INTO priority VALUES(\"%s\", %d);", suffix, priority), SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = ClearSuffixPriorityMapEvent.class)
    private static void clearSuffixPriorityMapEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.isReadFromSharedMemory.get()) {
            return;
        }
        databaseService.addToCommandSet(new SQLWithTaskId("DELETE FROM priority;", SqlTaskIds.UPDATE_SUFFIX, "cache"));
        databaseService.addToCommandSet(
                new SQLWithTaskId("INSERT INTO priority VALUES(\"defaultPriority\", 0);", SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = DeleteFromSuffixPriorityMapEvent.class)
    private static void deleteFromSuffixPriorityMapEvent(Event event) {
        DeleteFromSuffixPriorityMapEvent delete = (DeleteFromSuffixPriorityMapEvent) event;
        DatabaseService databaseService = getInstance();
        if (databaseService.isReadFromSharedMemory.get()) {
            return;
        }
        databaseService.addToCommandSet(new SQLWithTaskId(String.format("DELETE FROM priority where SUFFIX=\"%s\"", delete.suffix), SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = UpdateSuffixPriorityEvent.class)
    private static void updateSuffixPriorityEvent(Event event) {
        DatabaseService databaseService = DatabaseService.getInstance();
        if (databaseService.isReadFromSharedMemory.get()) {
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
        getInstance().executeAllCommands();
        SQLiteUtil.closeAll();
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
}

