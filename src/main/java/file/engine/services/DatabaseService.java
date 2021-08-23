package file.engine.services;

import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.dllInterface.FileMonitor;
import file.engine.dllInterface.GetAscII;
import file.engine.dllInterface.IsLocalDisk;
import file.engine.dllInterface.ResultPipe;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BootSystemEvent;
import file.engine.event.handler.impl.database.*;
import file.engine.event.handler.impl.monitor.disk.StartMonitorDiskEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.utils.*;
import file.engine.utils.bit.Bit;
import file.engine.utils.system.properties.IsDebug;
import lombok.Data;
import lombok.Getter;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

public class DatabaseService {
    private final ConcurrentLinkedQueue<SQLWithTaskId> commandSet = new ConcurrentLinkedQueue<>();
    private volatile Constants.Enums.DatabaseStatus status = Constants.Enums.DatabaseStatus.NORMAL;
    private final AtomicBoolean isExecuteImmediately = new AtomicBoolean(false);
    private final AtomicInteger cacheNum = new AtomicInteger(0);
    private final HashSet<TableNameWeightInfo> tableSet;    //保存从0-40数据库的表，使用频率和名字对应，使经常使用的表最快被搜索到
    private final ConcurrentLinkedQueue<String> tableQueue;  //保存哪些表需要被查
    private final AtomicBoolean isDatabaseUpdated = new AtomicBoolean(false);
    private final AtomicBoolean isReadSharedMemory = new AtomicBoolean(false);
    private final @Getter
    ConcurrentLinkedQueue<String> tempResults;  //在优先文件夹和数据库cache未搜索完时暂时保存结果，搜索完后会立即被转移到listResults
    private final AtomicBoolean isStop = new AtomicBoolean(false);
    private final ConcurrentLinkedQueue<Pair> priorityMap = new ConcurrentLinkedQueue<>();
    private volatile String[] searchCase;
    private volatile String searchText;
    private volatile String[] keywords;

    private static volatile DatabaseService INSTANCE = null;

    @Data
    private static class Pair {
        private final String suffix;
        private final int priority;
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
        tableSet = new HashSet<>();
        tempResults = new ConcurrentLinkedQueue<>();
        tableQueue = new ConcurrentLinkedQueue<>();
        addRecordsToDatabaseThread();
        deleteRecordsToDatabaseThread();
        checkTimeAndSendExecuteSqlSignalThread();
        executeSqlCommandsThread();
        initCacheNum();
        initPriority();
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
        commandSet.add(new SQLWithTaskId(format, SqlTaskIds.UPDATE_WEIGHT, "weight"));
        if (IsDebug.isDebug()) {
            System.err.println("已更新" + tableName + "权重, 之前为" + origin + "***增加了" + weight);
        }
    }

    /**
     * 获取数据库缓存条目数量，用于判断软件是否还能继续写入缓存
     */
    private void initCacheNum() {
        try (PreparedStatement stmt = SQLiteUtil.getPreparedStatement("SELECT COUNT(PATH) FROM cache;", "cache");
             ResultSet resultSet = stmt.executeQuery()) {
            cacheNum.set(resultSet.getInt(1));
        } catch (Exception throwables) {
            if (IsDebug.isDebug()) {
                throwables.printStackTrace();
            }
        }
    }

    /**
     * 处理所有sql线程
     */
    private void executeSqlCommandsThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            try {
                while (eventManagement.isNotMainExit()) {
                    if (isExecuteImmediately.get()) {
                        isExecuteImmediately.set(false);
                        if (!isReadSharedMemory.get()) {
                            executeAllCommands();
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException ignored) {
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
    private void deleteRecordsToDatabaseThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try (BufferedReader readerRemove =
                         new BufferedReader(new InputStreamReader(
                                 new FileInputStream(new File("tmp").getAbsolutePath() + File.separator + "fileRemoved.txt"),
                                 StandardCharsets.UTF_8))) {
                String tmp;
                int fileCount = 0;
                LinkedHashSet<String> deletePaths = new LinkedHashSet<>();
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.isNotMainExit()) {
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
                        if (!isReadSharedMemory.get()) {
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
    private void startMonitorDisk() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            TranslateUtil translateUtil = TranslateUtil.getInstance();
            String disks = AllConfigs.getInstance().getDisks();
            String[] splitDisks = RegexUtil.comma.split(disks);
            if (isAdmin()) {
                FileMonitor.INSTANCE.set_output(new File("tmp").getAbsolutePath());
                for (String root : splitDisks) {
                    if (IsLocalDisk.INSTANCE.isLocalDisk(root)) {
                        FileMonitor.INSTANCE.monitor(root);
                    }
                }
            } else {
                eventManagement.putEvent(new ShowTaskBarMessageEvent(
                        translateUtil.getTranslation("Warning"),
                        translateUtil.getTranslation("Not administrator, file monitoring function is turned off")));
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
                while (eventManagement.isNotMainExit()) {
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
                        if (!isReadSharedMemory.get()) {
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
     * 从缓存中搜索结果并将匹配的放入listResults
     */
    private void searchCache() {
        try (PreparedStatement statement = SQLiteUtil.getPreparedStatement("SELECT PATH FROM cache;", "cache");
             ResultSet resultSet = statement.executeQuery()) {
            while (resultSet.next()) {
                String eachCache = resultSet.getString("PATH");
                checkIsMatchedAndAddToList(eachCache, null);
            }
        } catch (SQLException throwables) {
            throwables.printStackTrace();
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
        if (PathMatchUtil.check(path, searchCase, searchText, keywords)) {
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
     * 搜索数据酷并加入到tempQueue中
     *
     * @param sql sql
     */
    private int searchAndAddToTempResults(String sql, ConcurrentSkipListSet<String> container, String disk) {
        int count = 0;
        //结果太多则不再进行搜索
        if (isStop.get()) {
            return count;
        }
        try (PreparedStatement stmt = SQLiteUtil.getPreparedStatement(sql, disk);
             ResultSet resultSet = stmt.executeQuery()) {
            String each;
            while (resultSet.next()) {
                //结果太多则不再进行搜索
                //用户重新输入了信息
                if (isStop.get()) {
                    tableQueue.clear();
                    return count;
                }
                each = resultSet.getString("PATH");
                if (checkIsMatchedAndAddToList(each, container)) {
                    count++;
                }
            }
        } catch (SQLException e) {
            System.err.println("error sql : " + sql);
            e.printStackTrace();
        }
        return count;
    }

    private void warmup() {
        initTableMap();
        LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> nonFormattedSql = getNonFormattedSqlFromTableQueue(false);
        nonFormattedSql.forEach((commandsMap, container) -> Arrays.stream(RegexUtil.comma.split(AllConfigs.getInstance().getDisks())).forEach(eachDisk -> {
            Set<String> sqls = commandsMap.keySet();
            String formattedSql;
            for (String each : sqls) {
                formattedSql = String.format(each, "PATH");
                String disk = String.valueOf(eachDisk.charAt(0));
                try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement(formattedSql, disk)) {
                    pStmt.execute();
                } catch (SQLException e) {
                    e.printStackTrace();
                }
            }
        }));
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement("SELECT PATH FROM cache", "cache")) {
            pStmt.execute();
        } catch (SQLException exception) {
            exception.printStackTrace();
        }
    }

    /**
     * 根据优先级将表排序放入tableQueue
     */
    private void initTableQueueByPriority() {
        tableQueue.clear();
        LinkedList<TableNameWeightInfo> tmpCommandList = new LinkedList<>(tableSet);
        //将tableSet通过权重排序
        tmpCommandList.sort((o1, o2) -> Long.compare(o2.weight.get(), o1.weight.get()));
        for (TableNameWeightInfo each : tmpCommandList) {
            if (IsDebug.isDebug()) {
                System.out.println("已添加表" + each.tableName + "----权重" + each.weight.get());
            }
            tableQueue.add(each.tableName);
        }
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
                if (weight > 100000000) {
                    isNeedSubtract = true;
                }
                if (isNeedSubtract) {
                    weight = weight / 2;
                }
                tableSet.add(new TableNameWeightInfo("list" + i, weight));
            }
        } else {
            for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
                tableSet.add(new TableNameWeightInfo("list" + i, 0));
            }
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
            Bit emptyContainerTask = new Bit(new byte[]{0});
            try {
                // 每当有一个容器全部包含后被清空，emptyContainerTask中对应的位会被设为1，当所有容器都为空时，取not运算将会得到1，但开始时取反也会得1，所以需要判断长度
                while (!isStop.get() && !(emptyContainerTask.not().equals(one) && emptyContainerTask.length() > 1)) {
                    Bit tmpTaskStatus = new Bit(new byte[]{0});
                    //线程状态的记录从第二个位开始，所以初始值为1 0
                    Bit start = new Bit(new byte[]{1, 0});
                    //循环次数，也是下方与运算结束后需要向右偏移的位数
                    int loopCount = 1;
                    //由于任务的耗时不同，如果阻塞时间过长，则跳过该任务，在下次循环中重新拿取结果
                    long waitTime = 0;
                    ConcurrentSkipListSet<String> results;
                    while (start.length() <= allTaskStatus.length() || taskStatus.or(zero).equals(zero)) {
                        if (isStop.get()) {
                            //用户重新输入，结束当前任务
                            break;
                        }
                        results = containerMap.get(start.toString());
                        if (results != null) {
                            for (String result : results) {
                                if (!tempResults.contains(result)) {
                                    tempResults.add(result);
                                    results.remove(result);
                                }
                            }
                        }
                        //当线程完成，taskStatus中的位会被设置为1
                        //这时，将taskStatus和start做与运算，然后移到第一位，如果为1，则线程已完成搜索
                        Bit and = taskStatus.and(start);
                        boolean isFailed = System.currentTimeMillis() - waitTime > 300 && waitTime != 0;
                        if (((and).shiftRight(loopCount)).equals(one) || isFailed) {
                            // failCount过大，阻塞时间过长则跳过
                            waitTime = 0;
                            results = containerMap.get(start.toString());
                            if (results != null) {
                                for (String result : results) {
                                    if (!tempResults.contains(result)) {
                                        tempResults.add(result);
                                    }
                                }
                                if (!isFailed) {
                                    results.clear();
                                    emptyContainerTask = emptyContainerTask.or(start);
                                }
                            }
                            tmpTaskStatus = tmpTaskStatus.or(start);
                            //将start左移，代表当前任务结束，继续拿下一个任务的结果
                            start.shiftLeft(1);
                            loopCount++;
                        } else {
                            if (waitTime == 0) {
                                waitTime = System.currentTimeMillis();
                            }
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(1);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    private void addSearchTasksForSharedMemory(LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> nonFormattedSql,
                                               Bit taskStatus,
                                               Bit allTaskStatus,
                                               LinkedHashMap<String, ConcurrentSkipListSet<String>> containerMap,
                                               ConcurrentLinkedQueue<Runnable> taskQueue) {
        Bit number = new Bit(new byte[]{1});
        nonFormattedSql.forEach((commandsMap, container) -> Arrays.stream(RegexUtil.comma.split(AllConfigs.getInstance().getDisks())).forEach(eachDisk -> {
            //为每个任务分配的位，不断左移以不断进行分配
            number.shiftLeft(1);
            Bit currentTaskNum = new Bit(number);
            allTaskStatus.set(allTaskStatus.or(currentTaskNum));
            containerMap.put(currentTaskNum.toString(), container);
            taskQueue.add(() -> {
                try {
                    Set<String> sqls = commandsMap.keySet();
                    for (String each : sqls) {
                        if (isStop.get()) {
                            return;
                        }
                        HashMap<String, String> parseSql = parseSql(each);
                        String listName = parseSql.get("list");
                        int priority = Integer.parseInt(parseSql.get("priority"));
                        String result;
                        for (int count = 0;
                             !isStop.get() && ((result = ResultPipe.INSTANCE.getResult(eachDisk.charAt(0), listName, priority, count)) != null);
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
        }));
    }

    /**
     * 解析出sql中的list和priority
     *
     * @param sql sql
     * @return map
     */
    private HashMap<String, String> parseSql(String sql) {
        String[] split = RegexUtil.blank.split(sql);
        HashMap<String, String> parsedVal = new HashMap<>();
        Arrays.stream(split).filter(each -> !each.isBlank()).forEach(each -> {
            int val;
            //noinspection IndexOfReplaceableByContains
            if (each.indexOf("list") != -1) {
                parsedVal.put("list", each);
            } else if ((val = each.indexOf("=")) != -1) {
                parsedVal.put("priority", each.substring(val + 1));
            }
        });
        return parsedVal;
    }

    /**
     * 添加搜索任务到队列
     *
     * @param nonFormattedSql sql未被格式化的所有任务
     * @param taskStatus      用于保存任务信息，这是一个通用变量，从第二个位开始，每一个位代表一个任务，当任务完成，该位将被置为1，否则为0，
     *                        例如第一个和第三个任务完成，第二个未完成，则为1010
     * @param allTaskStatus   所有的任务信息，从第二位开始，只要有任务被创建，该为就为1，例如三个任务被创建，则为1110
     * @param containerMap    每个任务搜索出来的结果都会被放到一个属于它自己的一个容器中，该容器保存任务与容器的映射关系
     * @param taskQueue       任务栈
     */
    private void addSearchTasks(LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> nonFormattedSql,
                                Bit taskStatus,
                                Bit allTaskStatus,
                                LinkedHashMap<String, ConcurrentSkipListSet<String>> containerMap,
                                ConcurrentLinkedQueue<Runnable> taskQueue) {
        Bit number = new Bit(new byte[]{1});
        nonFormattedSql.forEach((commandsMap, container) -> Arrays.stream(RegexUtil.comma.split(AllConfigs.getInstance().getDisks())).forEach(eachDisk -> {
            //为每个任务分配的位，不断左移以不断进行分配
            number.shiftLeft(1);
            Bit currentTaskNum = new Bit(number);
            allTaskStatus.set(allTaskStatus.or(currentTaskNum));
            containerMap.put(currentTaskNum.toString(), container);
            taskQueue.add(() -> {
                try {
                    Set<String> sqls = commandsMap.keySet();
                    for (String each : sqls) {
                        if (isStop.get()) {
                            return;
                        }
                        String formattedSql;
                        //格式化是为了以后的拓展性
                        formattedSql = String.format(each, "PATH");
                        //当前数据库表中有多少个结果匹配成功
                        int matchedNum = searchAndAddToTempResults(
                                formattedSql,
                                container,
                                String.valueOf(eachDisk.charAt(0)));
                        long weight = Math.min(matchedNum, 5);
                        if (weight != 0L) {
                            //更新表的权重，每次搜索将会按照各个表的权重排序
                            updateTableWeight(commandsMap.get(each), weight);
                        }
                        if (isStop.get()) {
                            break;
                        }
                    }
                } finally {
                    //执行完后将对应的线程flag设为1
                    taskStatus.set(taskStatus.or(currentTaskNum));
                }
            });
        }));
    }

    /**
     * 生成未格式化的sql
     * 第一个map中key保存未格式化的sql，value保存表名称，第二个map为搜索结果的暂时存储容器
     *
     * @return map
     */
    private LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> getNonFormattedSqlFromTableQueue(boolean isNeedContainer) {
        if (isDatabaseUpdated.get()) {
            isDatabaseUpdated.set(false);
            initPriority();
        }
        LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> sqlColumnMap = new LinkedHashMap<>();
        if (priorityMap.isEmpty()) {
            return sqlColumnMap;
        }
        initTableQueueByPriority();
        priorityMap.forEach(i -> {
            LinkedHashMap<String, String> eachPriorityMap = new LinkedHashMap<>();
            tableQueue.forEach(each -> {
                // where后面=不能有空格，否则解析priority会出错
                String sql = "SELECT %s FROM " + each + " WHERE priority=" + i.priority;
                eachPriorityMap.put(sql, each);
            });
            ConcurrentSkipListSet<String> container = null;
            if (isNeedContainer) {
                container = new ConcurrentSkipListSet<>();
            }
            sqlColumnMap.put(eachPriorityMap, container);
        });
        tableQueue.clear();
        return sqlColumnMap;
    }

    /**
     * 添加sql语句，并开始搜索
     */
    private void addSqlCommands() {
        if (!isReadSharedMemory.get()) {
            searchCache();
        }
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        //每个priority用一个线程，每一个后缀名对应一个优先级
        //按照优先级排列，key是sql和表名的对应，value是容器
        LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>>
                nonFormattedSql = getNonFormattedSqlFromTableQueue(true);
        Bit taskStatus = new Bit(new byte[]{0});
        Bit allTaskStatus = new Bit(new byte[]{0});

        LinkedHashMap<String, ConcurrentSkipListSet<String>> containerMap = new LinkedHashMap<>();
        //任务队列
        ConcurrentLinkedQueue<Runnable> taskQueue = new ConcurrentLinkedQueue<>();
        //添加搜索任务到队列
        if (isReadSharedMemory.get()) {
            addSearchTasksForSharedMemory(nonFormattedSql, taskStatus, allTaskStatus, containerMap, taskQueue);
        } else {
            addSearchTasks(nonFormattedSql, taskStatus, allTaskStatus, containerMap, taskQueue);
        }

        //添加消费者线程，接受任务进行处理，最高4线程
        int processors = Runtime.getRuntime().availableProcessors();
        processors = Math.min(processors, 4);
        for (int i = 0; i < processors; i++) {
            cachedThreadPoolUtil.executeTask(() -> {
                Runnable todo;
                while ((todo = taskQueue.poll()) != null) {
                    todo.run();
                }
            });
        }
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
        asciiGroup = Math.min(asciiGroup, 40);
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
        asciiGroup = Math.min(asciiGroup, 40);
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
        for (SQLWithTaskId each : commandSet) {
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
        }
    }

    /**
     * 初始化优先级表
     */
    private void initPriority() {
        priorityMap.clear();
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement("SELECT * FROM priority order by PRIORITY desc;", "cache")) {
            ResultSet resultSet = pStmt.executeQuery();
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
    private int getPriorityBySuffix(String suffix) {
        List<Pair> result = priorityMap.stream().filter(each -> each.suffix.equals(suffix)).collect(Collectors.toList());
        if (result.isEmpty()) {
            if (!"defaultPriority".equals(suffix)) {
                return getPriorityBySuffix("defaultPriority");
            }
        } else {
            return result.get(0).priority;
        }
        return 0;
    }

    /**
     * 获取文件后缀
     *
     * @param path 文件路径
     * @return 后缀名
     */
    private String getSuffixByPath(String path) {
        String name = getFileName(path);
        return name.substring(name.lastIndexOf('.') + 1).toLowerCase();
    }

    private void addFileToDatabase(String path) {
        int asciiSum = getAscIISum(getFileName(path));
        addAddSqlCommandByAscii(asciiSum, path, getPriorityBySuffix(getSuffixByPath(path)));
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
    private void executeImmediately() {
        isExecuteImmediately.set(true);
    }

    /**
     * 执行sql
     */
    @SuppressWarnings("SqlNoDataSourceInspection")
    private void executeAllCommands() {
        if (!commandSet.isEmpty()) {
            LinkedHashSet<SQLWithTaskId> tempCommandSet = new LinkedHashSet<>(commandSet);
            HashMap<String, LinkedList<String>> commandMap = new HashMap<>();
            tempCommandSet.forEach(sqlWithTaskId -> {
                if (commandMap.containsKey(sqlWithTaskId.key)) {
                    commandMap.get(sqlWithTaskId.key).add(sqlWithTaskId.sql);
                } else {
                    LinkedList<String> sqls = new LinkedList<>();
                    sqls.add(sqlWithTaskId.sql);
                    commandMap.put(sqlWithTaskId.key, sqls);
                }
            });
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
            commandSet.removeAll(tempCommandSet);
        }
    }

    /**
     * 添加任务到任务列表
     *
     * @param sql 任务
     */
    private void addToCommandSet(SQLWithTaskId sql) {
        if (commandSet.size() < Constants.MAX_SQL_NUM) {
            if (status == Constants.Enums.DatabaseStatus.MANUAL_UPDATE) {
                System.err.println("正在搜索中");
                return;
            }
            commandSet.add(sql);
        } else {
            if (IsDebug.isDebug()) {
                System.err.println("添加sql语句" + sql + "失败，已达到最大上限");
            }
            //立即处理sql语句
            executeImmediately();
        }
    }

    /**
     * 检查任务是否重复
     *
     * @param sql 任务
     * @return boolean
     */
    private boolean isCommandNotRepeat(String sql) {
        for (SQLWithTaskId each : commandSet) {
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
     * 更新文件索引
     *
     * @param disks      磁盘
     * @param ignorePath 忽略文件夹
     */
    private void searchFile(String disks, String ignorePath) throws IOException {
        searchByUSN(disks, ignorePath.toLowerCase());
    }

    /**
     * 创建索引
     */
    private void createAllIndex() {
        commandSet.add(new SQLWithTaskId("CREATE INDEX IF NOT EXISTS cache_index ON cache(PATH);", SqlTaskIds.CREATE_INDEX, "cache"));
        for (String each : RegexUtil.comma.split(AllConfigs.getInstance().getDisks())) {
            for (int i = 0; i <= Constants.ALL_TABLE_NUM; ++i) {
                String createIndex = "CREATE INDEX IF NOT EXISTS list" + i + "_index ON list" + i + "(PRIORITY);";
                commandSet.add(new SQLWithTaskId(createIndex, SqlTaskIds.CREATE_INDEX, String.valueOf(each.charAt(0))));
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
     * 进程是否存在
     *
     * @param procName   进程名
     * @return boolean
     * @throws IOException          失败
     * @throws InterruptedException 失败
     */
    @SuppressWarnings("IndexOfReplaceableByContains")
    private boolean isProcessExist(String procName) throws IOException, InterruptedException {
        StringBuilder strBuilder = new StringBuilder();
        if (!procName.isEmpty()) {
            Process p = Runtime.getRuntime().exec("tasklist /FI \"IMAGENAME eq " + procName + "\"");
            p.waitFor();
            String eachLine;
            try (BufferedReader buffr = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                while ((eachLine = buffr.readLine()) != null) {
                    strBuilder.append(eachLine);
                }
            }
            return strBuilder.toString().indexOf(procName) != -1;
        }
        return false;
    }

    /**
     * 添加等待进程并当进程完成后执行回调
     *
     * @param procName 进程名
     * @param callback 回调
     */
    private void waitForProcessAsync(@SuppressWarnings("SameParameterValue") String procName, Runnable callback) {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                waitForProcess(procName);
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            } finally {
                callback.run();
            }
        });
    }

    /**
     * 等待进程
     *
     * @param procName 进程名
     * @throws IOException          失败
     * @throws InterruptedException 失败
     */
    private void waitForProcess(@SuppressWarnings("SameParameterValue") String procName) throws IOException, InterruptedException {
        long start = System.currentTimeMillis();
        long timeLimit = 3 * 60 * 1000;
        if (IsDebug.isDebug()) {
            timeLimit = Long.MAX_VALUE;
        }
        while (isProcessExist(procName)) {
            TimeUnit.MILLISECONDS.sleep(10);
            if (System.currentTimeMillis() - start > timeLimit) {
                System.err.printf("等待进程%s超时\n", procName);
                String command = String.format("taskkill /im %s /f", procName);
                Process exec = Runtime.getRuntime().exec(command);
                exec.waitFor();
                break;
            }
        }
    }

    /**
     * 检查索引数据库大小
     */
    private void checkFileSize() throws IOException {
        SQLiteUtil.closeAll();
        for (String eachDisk : RegexUtil.comma.split(AllConfigs.getInstance().getDisks())) {
            String name = eachDisk.charAt(0) + ".db";
            long length = Files.size(Path.of("data/" + name));
            if (length > 6L * 1024 * 1024 * 100) {
                // 大小超过600M
                if (IsDebug.isDebug()) {
                    System.out.println("当前文件" + name + "大小超过600M，已删除");
                }
                Files.delete(Path.of("data/" + name));
            }
        }
        SQLiteUtil.initAllConnections();
    }

    /**
     * 关闭数据库连接并更新数据库
     *
     * @param ignorePath 忽略文件夹
     */
    private void updateLists(String ignorePath) throws IOException {
        checkFileSize();
        recreateDatabase();
        waitForCommandSet(SqlTaskIds.CREATE_TABLE);
        SQLiteUtil.closeAll();
        searchFile(AllConfigs.getInstance().getDisks(), ignorePath);
        try {
            waitForProcessAsync("fileSearcherUSN.exe", () -> {
                SQLiteUtil.initAllConnections();
                // 可能会出错
                recreateDatabase();
                createAllIndex();
                ResultPipe.INSTANCE.closeAllSharedMemory();
                isDatabaseUpdated.set(true);
                isReadSharedMemory.set(false);
            });
            long start = System.currentTimeMillis();
            isReadSharedMemory.set(true);
            while (!ResultPipe.INSTANCE.isComplete() && isProcessExist("fileSearcherUSN.exe")) {
                if (System.currentTimeMillis() - start > 60000) {
                    break;
                }
                TimeUnit.MILLISECONDS.sleep(10);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
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
            while (eventManagement.isNotMainExit()) {
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
        } catch (InterruptedException ignored) {
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
        for (SQLWithTaskId tasks : commandSet) {
            if (tasks.taskId == taskId) {
                return true;
            }
        }
        return false;
    }

    private HashMap<String, Integer> queryAllWeights() {
        HashMap<String, Integer> stringIntegerHashMap = new HashMap<>();
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement("SELECT TABLE_NAME, TABLE_WEIGHT FROM weight", "weight")) {
            ResultSet resultSet = pStmt.executeQuery();
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
        commandSet.clear();
        //删除所有索引
//        for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
//            commandSet.add(new SQLWithTaskId(SqlTaskIds.DROP_INDEX, "DROP INDEX IF EXISTS list" + i + "_index;"));
//        }
        //创建新表
        String[] disks = RegexUtil.comma.split(AllConfigs.getInstance().getDisks());
        String sql = "CREATE TABLE IF NOT EXISTS list";
        for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
            String command = sql + i + " " + "(ASCII INT, PATH text unique, PRIORITY INT)" + ";";
            for (String disk : disks) {
                commandSet.add(new SQLWithTaskId(command, SqlTaskIds.CREATE_TABLE, String.valueOf(disk.charAt(0))));
            }
        }
        executeImmediately();
    }

    private void checkTimeAndSendExecuteSqlSignalThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            // 时间检测线程
            final long updateTimeLimit = AllConfigs.getInstance().getUpdateTimeLimit();
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.isNotMainExit()) {
                    if (status == Constants.Enums.DatabaseStatus.NORMAL) {
                        executeImmediately();
                    }
                    TimeUnit.SECONDS.sleep(updateTimeLimit);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private boolean isAdmin() {
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


    @EventRegister(registerClass = StartMonitorDiskEvent.class)
    private static void startMonitorDiskEvent(Event event) {
        getInstance().startMonitorDisk();
    }

    @EventRegister(registerClass = StartSearchEvent.class)
    private static void startSearchEvent(Event event) {
        StartSearchEvent startSearchEvent = (StartSearchEvent) event;
        DatabaseService databaseService = getInstance();
        databaseService.searchText = startSearchEvent.searchText;
        databaseService.searchCase = startSearchEvent.searchCase;
        databaseService.keywords = startSearchEvent.keywords;
        databaseService.isStop.set(false);
        databaseService.addSqlCommands();
    }

    @EventRegister(registerClass = StopSearchEvent.class)
    private static void stopSearchEvent(Event event) {
        DatabaseService databaseService = getInstance();
        databaseService.isStop.set(true);
        databaseService.tableQueue.clear();
        databaseService.tempResults.clear();
    }

    @EventListener(listenClass = BootSystemEvent.class)
    private static void warmupDatabase(Event event) {
        getInstance().warmup();
    }

    @EventRegister(registerClass = AddToCacheEvent.class)
    private static void addToCacheEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.isReadSharedMemory.get()) {
            return;
        }
        databaseService.addFileToCache(((AddToCacheEvent) event).path);
        databaseService.cacheNum.incrementAndGet();
    }

    @EventRegister(registerClass = DeleteFromCacheEvent.class)
    private static void deleteFromCacheEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.isReadSharedMemory.get()) {
            return;
        }
        databaseService.removeFileFromCache(((DeleteFromCacheEvent) event).path);
        databaseService.cacheNum.decrementAndGet();
    }

    @EventRegister(registerClass = UpdateDatabaseEvent.class)
    private static void updateDatabaseEvent(Event event) throws IOException {
        DatabaseService databaseService = getInstance();
        if (databaseService.isReadSharedMemory.get()) {
            return;
        }
        databaseService.setStatus(Constants.Enums.DatabaseStatus.MANUAL_UPDATE);
        databaseService.updateLists(AllConfigs.getInstance().getIgnorePath());
        databaseService.setStatus(Constants.Enums.DatabaseStatus.NORMAL);
    }

    @EventRegister(registerClass = OptimiseDatabaseEvent.class)
    private static void optimiseDatabaseEvent(Event event) {
        DatabaseService databaseService = getInstance();
        if (databaseService.isReadSharedMemory.get()) {
            return;
        }
        databaseService.setStatus(Constants.Enums.DatabaseStatus.VACUUM);
        //执行VACUUM命令
        for (String eachDisk : RegexUtil.comma.split(AllConfigs.getInstance().getDisks())) {
            try (PreparedStatement stmt = SQLiteUtil.getPreparedStatement("VACUUM;", String.valueOf(eachDisk.charAt(0)))) {
                stmt.execute();
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
        if (databaseService.isReadSharedMemory.get()) {
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
        if (databaseService.isReadSharedMemory.get()) {
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
        if (databaseService.isReadSharedMemory.get()) {
            return;
        }
        databaseService.addToCommandSet(new SQLWithTaskId(String.format("DELETE FROM priority where SUFFIX=\"%s\"", delete.suffix), SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = UpdateSuffixPriorityEvent.class)
    private static void updateSuffixPriorityEvent(Event event) {
        DatabaseService databaseService = DatabaseService.getInstance();
        if (databaseService.isReadSharedMemory.get()) {
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

