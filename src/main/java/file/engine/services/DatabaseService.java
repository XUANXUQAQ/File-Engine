package file.engine.services;

import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.Enums;
import file.engine.configs.Constants;
import file.engine.dllInterface.GetAscII;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.database.*;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.utils.RegexUtil;
import file.engine.utils.SQLiteUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.Data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class DatabaseService {
    private final ConcurrentLinkedQueue<SQLWithTaskId> commandSet = new ConcurrentLinkedQueue<>();
    private volatile Enums.DatabaseStatus status = Enums.DatabaseStatus.NORMAL;
    private final AtomicBoolean isExecuteImmediately = new AtomicBoolean(false);
    private final AtomicInteger cacheNum = new AtomicInteger(0);

    private static volatile DatabaseService INSTANCE = null;

    private DatabaseService() {
        addRecordsToDatabaseThread();
        deleteRecordsToDatabaseThread();
        checkTimeAndSendExecuteSqlSignalThread();
        executeSqlCommandsThread();
        initCacheNum();
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
                        executeAllCommands();
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
                    if (status == Enums.DatabaseStatus.NORMAL) {
                        while ((tmp = readerRemove.readLine()) != null) {
                            fileCount++;
                            deletePaths.add(tmp);
                            if (fileCount > 3000) {
                                break;
                            }
                        }
                    }
                    if (status == Enums.DatabaseStatus.NORMAL && !deletePaths.isEmpty()) {
                        eventManagement.putEvent(new DeleteFromDatabaseEvent(deletePaths));
                        deletePaths = new LinkedHashSet<>();
                        fileCount = 0;
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException ignored) {
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
                    if (status == Enums.DatabaseStatus.NORMAL) {
                        while ((tmp = readerAdd.readLine()) != null) {
                            fileCount++;
                            addPaths.add(tmp);
                            if (fileCount > 3000) {
                                break;
                            }
                        }
                    }
                    if (status == Enums.DatabaseStatus.NORMAL && !addPaths.isEmpty()) {
                        eventManagement.putEvent(new AddToDatabaseEvent(addPaths));
                        addPaths = new LinkedHashSet<>();
                        fileCount = 0;
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException ignored) {
            }
        });
    }

    /**
     * 生成删除记录sql
     * @param asciiSum ascii
     * @param path 文件路径
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
     * @param asciiSum ascii
     * @param path 文件路径
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
     * @param path 文件路径
     */
    private void removeFileFromDatabase(String path) {
        int asciiSum = getAscIISum(getFileName(path));
        if (isRemoveFileInDatabase(path)) {
            addDeleteSqlCommandByAscii(asciiSum, path);
        }
    }

    /**
     * 根据文件后缀获取优先级信息
     * @param suffix 文件后缀名
     * @return 优先级
     */
    private int getPriorityBySuffix(String suffix) {
        String sqlTemplate = "select PRIORITY from priority where suffix=\"%s\"";
        String sql = String.format(sqlTemplate, suffix);
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement(sql, "cache")) {
            ResultSet resultSet = pStmt.executeQuery();
            if (resultSet.next()) {
                String priority = resultSet.getString("PRIORITY");
                return Integer.parseInt(priority);
            }
        } catch (Exception throwables) {
            throwables.printStackTrace();
        }
        if (!"defaultPriority".equals(suffix)) {
            return getPriorityBySuffix("defaultPriority");
        }
        return 0;
    }

    /**
     * 获取文件后缀
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
     * @param sql 任务
     */
    private void addToCommandSet(SQLWithTaskId sql) {
        if (commandSet.size() < Constants.MAX_SQL_NUM) {
            if (status == Enums.DatabaseStatus.MANUAL_UPDATE) {
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
     * @return 装填
     */
    public Enums.DatabaseStatus getStatus() {
        return status;
    }

    /**
     * 设置数据库状态
     * @param status 状态
     */
    private void setStatus(Enums.DatabaseStatus status) {
        this.status = status;
    }

    /**
     * 更新文件索引
     * @param disks 磁盘
     * @param ignorePath 忽略文件夹
     */
    private void searchFile(String disks, String ignorePath) throws IOException, InterruptedException {
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
     * @param paths 磁盘信息
     * @param ignorePath 忽略文件夹
     * @throws IOException exception
     * @throws InterruptedException exception
     */
    private void searchByUSN(String paths, String ignorePath) throws IOException, InterruptedException {
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
        waitForProcess("fileSearcherUSN.exe");
    }

    /**
     * 进程是否存在
     * @param procName 进程名
     * @param strBuilder 用于获取cmd执行tasklist后返回的信息
     * @return boolean
     * @throws IOException 失败
     * @throws InterruptedException 失败
     */
    private boolean isTaskExist(String procName, StringBuilder strBuilder) throws IOException, InterruptedException {
        if (!procName.isEmpty()) {
            Process p = Runtime.getRuntime().exec("tasklist /FI \"IMAGENAME eq " + procName + "\"");
            p.waitFor();
            String eachLine;
            try (BufferedReader buffr = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                while ((eachLine = buffr.readLine()) != null) {
                    strBuilder.append(eachLine);
                }
            }
            return strBuilder.toString().contains(procName);
        }
        return false;
    }

    /**
     * 等待进程
     * @param procName 进程名
     * @throws IOException 失败
     * @throws InterruptedException 失败
     */
    private void waitForProcess(@SuppressWarnings("SameParameterValue") String procName) throws IOException, InterruptedException {
        StringBuilder strBuilder = new StringBuilder();
        long start = System.currentTimeMillis();
        while (isTaskExist(procName, strBuilder)) {
            TimeUnit.MILLISECONDS.sleep(10);
            if (System.currentTimeMillis() - start > 3 * 60 * 1000) {
                System.err.printf("等待进程%s超时\n", procName);
                String command = String.format("taskkill /im %s /f", procName);
                Process exec = Runtime.getRuntime().exec(command);
                exec.waitFor();
                break;
            }
            strBuilder.delete(0, strBuilder.length());
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
     * @param ignorePath 忽略文件夹
     */
    private void updateLists(String ignorePath) throws IOException, InterruptedException {
        checkFileSize();
        recreateDatabase();
        waitForCommandSet(SqlTaskIds.CREATE_TABLE);
        SQLiteUtil.closeAll();
        try {
            searchFile(AllConfigs.getInstance().getDisks(), ignorePath);
        } finally {
            SQLiteUtil.initAllConnections();
            // 可能会出错
            recreateDatabase();
            createAllIndex();
        }
    }

    /**
     * 等待sql任务执行
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

    /**
     * 更新表权重
     * @param tableName 表名
     * @param weight 权重
     */
    private void updateTableWeight(String tableName, long weight) {
        String format = String.format("UPDATE weight SET TABLE_WEIGHT=%d WHERE TABLE_NAME=\"%s\"", weight, tableName);
        commandSet.add(new SQLWithTaskId(format, SqlTaskIds.UPDATE_WEIGHT, "weight"));
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
                    if (status == Enums.DatabaseStatus.NORMAL) {
                        eventManagement.putEvent(new ExecuteSQLEvent());
                    }
                    TimeUnit.SECONDS.sleep(updateTimeLimit);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    @EventRegister(registerClass = QueryAllWeightsEvent.class)
    private static void queryAllWeights(Event event) {
        HashMap<String, Integer> weightsMap = getInstance().queryAllWeights();
        event.setReturnValue(weightsMap);
    }

    @EventRegister(registerClass = UpdateTableWeightEvent.class)
    private static void updateWeight(Event event) {
        UpdateTableWeightEvent updateTableWeightEvent = (UpdateTableWeightEvent) event;
        getInstance().updateTableWeight(updateTableWeightEvent.tableName, updateTableWeightEvent.tableWeight);
    }

    @EventRegister(registerClass = AddToCacheEvent.class)
    private static void addToCacheEvent(Event event) {
        DatabaseService databaseService = getInstance();
        databaseService.addFileToCache(((AddToCacheEvent) event).path);
        databaseService.cacheNum.incrementAndGet();
    }

    @EventRegister(registerClass = DeleteFromCacheEvent.class)
    private static void deleteFromCacheEvent(Event event) {
        DatabaseService databaseService = getInstance();
        databaseService.removeFileFromCache(((DeleteFromCacheEvent) event).path);
        databaseService.cacheNum.decrementAndGet();
    }

    @EventRegister(registerClass = AddToDatabaseEvent.class)
    private static void addToDatabaseEvent(Event event) {
        DatabaseService databaseService = getInstance();
        for (Object each : ((AddToDatabaseEvent) event).getPaths()) {
            databaseService.addFileToDatabase((String) each);
        }
    }

    @EventRegister(registerClass = DeleteFromDatabaseEvent.class)
    private static void deleteFromDatabaseEvent(Event event) {
        DeleteFromDatabaseEvent deleteFromDatabaseEvent = ((DeleteFromDatabaseEvent) event);
        DatabaseService databaseService = getInstance();
        for (Object each : deleteFromDatabaseEvent.getPaths()) {
            databaseService.removeFileFromDatabase((String) each);
        }
    }

    @EventRegister(registerClass = UpdateDatabaseEvent.class)
    private static void updateDatabaseEvent(Event event) throws IOException, InterruptedException {
        DatabaseService databaseService = getInstance();
        databaseService.setStatus(Enums.DatabaseStatus.MANUAL_UPDATE);
        databaseService.updateLists(AllConfigs.getInstance().getIgnorePath());
        databaseService.setStatus(Enums.DatabaseStatus.NORMAL);
    }

    @EventRegister(registerClass = ExecuteSQLEvent.class)
    private static void executeSQLEvent(Event event) {
        getInstance().executeImmediately();
    }

    @EventRegister(registerClass = OptimiseDatabaseEvent.class)
    private static void optimiseDatabaseEvent(Event event) {
        DatabaseService databaseService = getInstance();
        databaseService.setStatus(Enums.DatabaseStatus.VACUUM);
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
        databaseService.setStatus(Enums.DatabaseStatus.NORMAL);
    }

    @EventRegister(registerClass = AddToSuffixPriorityMapEvent.class)
    private static void addToSuffixPriorityMapEvent(Event event) {
        AddToSuffixPriorityMapEvent event1 = (AddToSuffixPriorityMapEvent) event;
        String suffix = event1.suffix;
        int priority = event1.priority;
        DatabaseService databaseService = getInstance();
        databaseService.addToCommandSet(
                new SQLWithTaskId(String.format("INSERT INTO priority VALUES(\"%s\", %d);", suffix, priority), SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = ClearSuffixPriorityMapEvent.class)
    private static void clearSuffixPriorityMapEvent(Event event) {
        DatabaseService databaseService = getInstance();
        databaseService.addToCommandSet(new SQLWithTaskId("DELETE FROM priority;", SqlTaskIds.UPDATE_SUFFIX, "cache"));
        databaseService.addToCommandSet(
                new SQLWithTaskId("INSERT INTO priority VALUES(\"defaultPriority\", 0);", SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = DeleteFromSuffixPriorityMapEvent.class)
    private static void deleteFromSuffixPriorityMapEvent(Event event) {
        DeleteFromSuffixPriorityMapEvent delete = (DeleteFromSuffixPriorityMapEvent) event;
        DatabaseService databaseService = getInstance();
        databaseService.addToCommandSet(new SQLWithTaskId(String.format("DELETE FROM priority where SUFFIX=\"%s\"", delete.suffix), SqlTaskIds.UPDATE_SUFFIX, "cache"));
    }

    @EventRegister(registerClass = UpdateSuffixPriorityEvent.class)
    private static void updateSuffixPriorityEvent(Event event) {
        EventManagement eventManagement = EventManagement.getInstance();
        UpdateSuffixPriorityEvent update = (UpdateSuffixPriorityEvent) event;
        String origin = update.originSuffix;
        String newSuffix = update.newSuffix;
        int newNum = update.newPriority;
        eventManagement.putEvent(new DeleteFromSuffixPriorityMapEvent(origin));
        eventManagement.putEvent(new AddToSuffixPriorityMapEvent(newSuffix, newNum));
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void restartEvent() {
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
        CREATE_INDEX, CREATE_TABLE, DROP_TABLE, DROP_INDEX, UPDATE_SUFFIX,UPDATE_WEIGHT
    }
}

