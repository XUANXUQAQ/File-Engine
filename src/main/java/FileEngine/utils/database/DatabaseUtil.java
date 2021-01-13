package FileEngine.utils.database;

import FileEngine.IsDebug;
import FileEngine.configs.AllConfigs;
import FileEngine.configs.Enums;
import FileEngine.dllInterface.GetAscII;
import FileEngine.dllInterface.IsLocalDisk;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventHandler;
import FileEngine.eventHandler.impl.database.*;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarMessageEvent;
import FileEngine.utils.CachedThreadPoolUtil;
import FileEngine.utils.EventUtil;
import FileEngine.utils.TranslateUtil;

import javax.swing.filechooser.FileSystemView;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class DatabaseUtil {
    private final ConcurrentLinkedQueue<SQLWithTaskId> commandSet = new ConcurrentLinkedQueue<>();
    private volatile Enums.DatabaseStatus status = Enums.DatabaseStatus.NORMAL;
    private final AtomicBoolean isExecuteImmediately = new AtomicBoolean(false);

    private static final int MAX_SQL_NUM = 5000;

    private static volatile DatabaseUtil INSTANCE = null;

    private DatabaseUtil() {
        addRecordsToDatabaseThread();
        deleteRecordsToDatabaseThread();
        checkTimeAndSendExecuteSqlSignalThread();
        executeSqlCommandsThread();
    }

    private void executeSqlCommandsThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventUtil eventUtil = EventUtil.getInstance();
            try (Statement statement = SQLiteUtil.getStatement()) {
                while (eventUtil.isNotMainExit()) {
                    if (isExecuteImmediately.get()) {
                        isExecuteImmediately.set(false);
                        executeAllCommands(statement);
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException ignored) {
            } catch (Exception throwables) {
                throwables.printStackTrace();
            }
        });
    }

    public static DatabaseUtil getInstance() {
        if (INSTANCE == null) {
            synchronized (DatabaseUtil.class) {
                if (INSTANCE == null) {
                    INSTANCE = new DatabaseUtil();
                }
            }
        }
        return INSTANCE;
    }

    private void deleteRecordsToDatabaseThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try (BufferedReader readerRemove =
                         new BufferedReader(new InputStreamReader(
                                 new FileInputStream(new File("tmp").getAbsolutePath() + File.separator + "fileRemoved.txt"),
                                         StandardCharsets.UTF_8))) {
                String tmp;
                int fileCount = 0;
                LinkedHashSet<String> deletePaths = new LinkedHashSet<>();
                EventUtil eventUtil = EventUtil.getInstance();
                while (eventUtil.isNotMainExit()) {
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
                        eventUtil.putEvent(new DeleteFromDatabaseEvent(deletePaths));
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
                EventUtil eventUtil = EventUtil.getInstance();
                while (eventUtil.isNotMainExit()) {
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
                        eventUtil.putEvent(new AddToDatabaseEvent(addPaths));
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

    private void addDeleteSqlCommandByAscii(int asciiSum, String path) {
        String command;
        int asciiGroup = asciiSum / 100;

        switch (asciiGroup) {
            case 0:
                command = "DELETE from list0 where PATH=" + "\"" + path + "\";";
                break;
            case 1:
                command = "DELETE from list1 where PATH=" + "\"" + path + "\";";
                break;
            case 2:
                command = "DELETE from list2 where PATH=" + "\"" + path + "\";";
                break;
            case 3:
                command = "DELETE from list3 where PATH=" + "\"" + path + "\";";
                break;
            case 4:
                command = "DELETE from list4 where PATH=" + "\"" + path + "\";";
                break;
            case 5:
                command = "DELETE from list5 where PATH=" + "\"" + path + "\";";
                break;
            case 6:
                command = "DELETE from list6 where PATH=" + "\"" + path + "\";";
                break;
            case 7:
                command = "DELETE from list7 where PATH=" + "\"" + path + "\";";
                break;
            case 8:
                command = "DELETE from list8 where PATH=" + "\"" + path + "\";";
                break;
            case 9:
                command = "DELETE from list9 where PATH=" + "\"" + path + "\";";
                break;
            case 10:
                command = "DELETE from list10 where PATH=" + "\"" + path + "\";";
                break;
            case 11:
                command = "DELETE from list11 where PATH=" + "\"" + path + "\";";
                break;
            case 12:
                command = "DELETE from list12 where PATH=" + "\"" + path + "\";";
                break;
            case 13:
                command = "DELETE from list13 where PATH=" + "\"" + path + "\";";
                break;
            case 14:
                command = "DELETE from list14 where PATH=" + "\"" + path + "\";";
                break;
            case 15:
                command = "DELETE from list15 where PATH=" + "\"" + path + "\";";
                break;
            case 16:
                command = "DELETE from list16 where PATH=" + "\"" + path + "\";";
                break;
            case 17:
                command = "DELETE from list17 where PATH=" + "\"" + path + "\";";
                break;
            case 18:
                command = "DELETE from list18 where PATH=" + "\"" + path + "\";";
                break;
            case 19:
                command = "DELETE from list19 where PATH=" + "\"" + path + "\";";
                break;
            case 20:
                command = "DELETE from list20 where PATH=" + "\"" + path + "\";";
                break;
            case 21:
                command = "DELETE from list21 where PATH=" + "\"" + path + "\";";
                break;
            case 22:
                command = "DELETE from list22 where PATH=" + "\"" + path + "\";";
                break;
            case 23:
                command = "DELETE from list23 where PATH=" + "\"" + path + "\";";
                break;
            case 24:
                command = "DELETE from list24 where PATH=" + "\"" + path + "\";";
                break;
            case 25:
                command = "DELETE from list25 where PATH=" + "\"" + path + "\";";
                break;
            case 26:
                command = "DELETE from list26 where PATH=" + "\"" + path + "\";";
                break;
            case 27:
                command = "DELETE from list27 where PATH=" + "\"" + path + "\";";
                break;
            case 28:
                command = "DELETE from list28 where PATH=" + "\"" + path + "\";";
                break;
            case 29:
                command = "DELETE from list29 where PATH=" + "\"" + path + "\";";
                break;
            case 30:
                command = "DELETE from list30 where PATH=" + "\"" + path + "\";";
                break;
            case 31:
                command = "DELETE from list31 where PATH=" + "\"" + path + "\";";
                break;
            case 32:
                command = "DELETE from list32 where PATH=" + "\"" + path + "\";";
                break;
            case 33:
                command = "DELETE from list33 where PATH=" + "\"" + path + "\";";
                break;
            case 34:
                command = "DELETE from list34 where PATH=" + "\"" + path + "\";";
                break;
            case 35:
                command = "DELETE from list35 where PATH=" + "\"" + path + "\";";
                break;
            case 36:
                command = "DELETE from list36 where PATH=" + "\"" + path + "\";";
                break;
            case 37:
                command = "DELETE from list37 where PATH=" + "\"" + path + "\";";
                break;
            case 38:
                command = "DELETE from list38 where PATH=" + "\"" + path + "\";";
                break;
            case 39:
                command = "DELETE from list39 where PATH=" + "\"" + path + "\";";
                break;
            case 40:
                command = "DELETE from list40 where PATH=" + "\"" + path + "\";";
                break;
            default:
                command = null;
                break;
        }
        if (command != null && isCommandNotRepeat(command)) {
            addToCommandSet(new SQLWithTaskId(SqlTaskIds.DELETE_FROM_LIST, command));
        }
    }

    private void addAddSqlCommandByAscii(int asciiSum, String path, int priority) {
        String commandTemplate = "INSERT OR IGNORE INTO %s VALUES(%d, \"%s\", %d)";
        int asciiGroup = asciiSum / 100;
        if (asciiGroup <= 40) {
            String columnName = "list" + asciiGroup;
            String command = String.format(commandTemplate, columnName, asciiSum, path, priority);
            if (command != null && isCommandNotRepeat(command)) {
                addToCommandSet(new SQLWithTaskId(SqlTaskIds.INSERT_TO_LIST, command));
            }
        }
    }

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

    private void removeFileFromDatabase(String path) {
        int asciiSum = getAscIISum(getFileName(path));
        if (isRemoveFileInDatabase(path)) {
            addDeleteSqlCommandByAscii(asciiSum, path);
            if (IsDebug.isDebug()) {
                System.out.println("删除" + path + "," + "asciiSum为" + asciiSum);
            }
        }
    }

    private int getPriorityBySuffix(String suffix) {
        String sqlTemplate = "select PRIORITY from priority where suffix=\"%s\"";
        String sql = String.format(sqlTemplate, suffix);
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement(sql)) {
            ResultSet resultSet = pStmt.executeQuery();
            if (resultSet.next()) {
                String priority = resultSet.getString("PRIORITY");
                return Integer.parseInt(priority);
            }
        } catch (Exception throwables) {
            throwables.printStackTrace();
        }
        return getPriorityBySuffix("defaultPriority");
    }

    private String getSuffixByPath(String path) {
        String name = getFileName(path);
        return name.substring(name.lastIndexOf('.') + 1).toLowerCase();
    }

    private void addFileToDatabase(String path) {
        int asciiSum = getAscIISum(getFileName(path));
        addAddSqlCommandByAscii(asciiSum, path, getPriorityBySuffix(getSuffixByPath(path)));
        if (IsDebug.isDebug()) {
            System.out.println("添加" + path + "," + "asciiSum为" + asciiSum);
        }
    }

    private void addFileToCache(String path) {
        String command = "INSERT OR IGNORE INTO cache(PATH) VALUES(\"" + path + "\");";
        if (isCommandNotRepeat(command)) {
            addToCommandSet(new SQLWithTaskId(SqlTaskIds.INSERT_TO_CACHE, command));
            if (IsDebug.isDebug()) {
                System.out.println("添加" + path + "到缓存");
            }
        }
    }

    private void removeFileFromCache(String path) {
        String command = "DELETE from cache where PATH=" + "\"" + path + "\";";
        if (isCommandNotRepeat(command)) {
            addToCommandSet(new SQLWithTaskId(SqlTaskIds.DELETE_FROM_CACHE, command));
            if (IsDebug.isDebug()) {
                System.out.println("删除" + path + "到缓存");
            }
        }
    }

    private void executeImmediately() {
        isExecuteImmediately.set(true);
    }

    private void executeAllCommands(Statement stmt) {
        if (!commandSet.isEmpty()) {
            LinkedHashSet<SQLWithTaskId> tempCommandSet = new LinkedHashSet<>(commandSet);
            try {
                if (IsDebug.isDebug()) {
                    System.out.println("----------------------------------------------");
                    System.out.println("执行SQL命令");
                    System.out.println("----------------------------------------------");
                }
                stmt.execute("BEGIN;");
                for (SQLWithTaskId each : tempCommandSet) {
                    stmt.execute(each.sql);
                    commandSet.remove(each);
                    if (IsDebug.isDebug()) {
                        System.err.println("当前执行SQL---" + each.sql + "----------------任务组：" + each.taskId);
                    }
                }
            } catch (SQLException e) {
                if (IsDebug.isDebug()) {
                    e.printStackTrace();
                    for (SQLWithTaskId each : tempCommandSet) {
                        System.err.println("执行失败：" + each.sql + "----------------任务组：" + each.taskId);
                    }
                }
            } finally {
                try {
                    stmt.execute("COMMIT;");
                } catch (SQLException throwables) {
                    throwables.printStackTrace();
                }
            }
        }
    }

    private void addToCommandSet(SQLWithTaskId sql) {
        if (commandSet.size() < MAX_SQL_NUM) {
            commandSet.add(sql);
        } else {
            if (IsDebug.isDebug()) {
                System.err.println("添加sql语句" + sql + "失败，已达到最大上限");
            }
            //立即处理sql语句
            executeImmediately();
        }
    }

    private boolean isCommandNotRepeat(String sql) {
        for (SQLWithTaskId each : commandSet) {
            if (each.sql.equals(sql)) {
                return false;
            }
        }
        return true;
    }

    public Enums.DatabaseStatus getStatus() {
        return status;
    }

    private void setStatus(Enums.DatabaseStatus status) {
        this.status = status;
    }

    private void searchFile(String ignorePath, int searchDepth) {
        boolean canSearchByUsn = false;
        File[] roots = File.listRoots();
        StringBuilder ntfsDisk = new StringBuilder(50);
        StringBuilder nonNtfsDisk = new StringBuilder(50);
        for (File root : roots) {
            if (IsLocalDisk.INSTANCE.isLocalDisk(root.getAbsolutePath())) {
                if (IsLocalDisk.INSTANCE.isDiskNTFS(root.getAbsolutePath())) {
                    canSearchByUsn = true;
                    ntfsDisk.append(root.getAbsolutePath()).append(",");
                } else {
                    nonNtfsDisk.append(root.getAbsolutePath()).append(",");
                }
            }
        }
        if (canSearchByUsn) {
            try {
                searchByUSN(ntfsDisk.toString(), ignorePath.toLowerCase());
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        } else {
            nonNtfsDisk.append(ntfsDisk);
        }

        String[] paths = nonNtfsDisk.toString().split(",");
        for (String path : paths) {
            if (!path.isEmpty()) {
                path = path.substring(0, 2);
                try {
                    searchFile(path, searchDepth, ignorePath);
                } catch (InterruptedException | IOException e) {
                    e.printStackTrace();
                }
            }
        }

        String[] desktops = new String[]{getDesktop(), "C:\\Users\\Public\\Desktop"};
        for (String eachDesktop : desktops) {
            File[] desktopFiles = new File(eachDesktop).listFiles();
            if (desktopFiles != null) {
                if (desktopFiles.length != 0) {
                    searchFileIgnoreSearchDepth(eachDesktop, ignorePath);
                }
            }
        }
        createAllIndex();
        waitForCommandSet(SqlTaskIds.CREATE_INDEX);
        EventUtil.getInstance().putEvent(new ShowTaskBarMessageEvent(
                TranslateUtil.getInstance().getTranslation("Info"),
                TranslateUtil.getInstance().getTranslation("Search Done")));
    }

    private void createAllIndex() {
        commandSet.add(new SQLWithTaskId(SqlTaskIds.CREATE_INDEX, "CREATE INDEX IF NOT EXISTS cache_index ON cache(PATH);"));
        for (int i = 0; i <= 40; ++i) {
            String createIndex = "CREATE INDEX IF NOT EXISTS list" + i + "_index ON list" + i + "(PRIORITY);";
            commandSet.add(new SQLWithTaskId(SqlTaskIds.CREATE_INDEX, createIndex));
        }
        executeImmediately();
    }

    private void searchByUSN(String paths, String ignorePath) throws IOException, InterruptedException {
        File usnSearcher = new File("user/fileSearcherUSN.exe");
        String absPath = usnSearcher.getAbsolutePath();
        String start = absPath.substring(0, 2);
        String end = "\"" + absPath.substring(2) + "\"";
        File database = new File("data.db");
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
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/MFTSearchInfo.dat"), StandardCharsets.UTF_8))) {
            buffW.write("");
        }
    }

    private String getDesktop() {
        FileSystemView fsv = FileSystemView.getFileSystemView();
        return fsv.getHomeDirectory().getAbsolutePath();
    }

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

    private void waitForProcess(String procName) throws IOException, InterruptedException {
        StringBuilder strBuilder = new StringBuilder();
        while (isTaskExist(procName, strBuilder)) {
            TimeUnit.MILLISECONDS.sleep(10);
            strBuilder.delete(0, strBuilder.length());
        }
    }

    private void searchFileIgnoreSearchDepth(String path, String ignorePath) {
        File fileSearcher = new File("user/fileSearcher.exe");
        String absPath = fileSearcher.getAbsolutePath();
        String start = absPath.substring(0, 2);
        String end = "\"" + absPath.substring(2) + "\"";
        File database = new File("data.db");
        String command = "cmd.exe /c " + start + end + " \"" + path + "\"" + " \"1\" " + "\"" + ignorePath + "\" " + "\"" + database.getAbsolutePath() + "\" " + "\"" + "1" + "\"";
        try {
            Runtime.getRuntime().exec(command, null, new File("user"));
            waitForProcess("fileSearcher.exe");
        } catch (IOException | InterruptedException e) {
            if (!(e instanceof InterruptedException) && IsDebug.isDebug()) {
                e.printStackTrace();
            }
        }
    }

    private void searchFile(String path, int searchDepth, String ignorePath) throws InterruptedException, IOException {
        File fileSearcher = new File("user/fileSearcher.exe");
        String absPath = fileSearcher.getAbsolutePath();
        String start = absPath.substring(0, 2);
        String end = "\"" + absPath.substring(2) + "\"";
        File database = new File("data.db");
        String command = "cmd.exe /c " + start + end + " \"" + path + "\"" + " \"" + searchDepth + "\" " + "\"" + ignorePath + "\" " + "\"" + database.getAbsolutePath() + "\" " + "\"" + "0" + "\"";
        Runtime.getRuntime().exec(command, null, new File("user"));
        waitForProcess("fileSearcher.exe");
    }

    private void updateLists(String ignorePath, int searchDepth) {
        recreateDatabase();
        waitForCommandSet(SqlTaskIds.CREATE_TABLE);
        searchFile(ignorePath, searchDepth);
    }

    private void waitForCommandSet(SqlTaskIds taskId) {
        try {
            int count = 0;
            EventUtil eventUtil = EventUtil.getInstance();
            while (eventUtil.isNotMainExit()) {
                count++;
                //等待10s
                if (count > 1000) {
                    System.err.println("等待SQL语句任务" + taskId + "处理超时");
                    break;
                }
                //判断commandSet中是否还有taskId存在
                if (!isTaskExistInCommandSet(taskId)) {
                    break;
                }
                TimeUnit.MILLISECONDS.sleep(10);
            }
        }catch (InterruptedException ignored) {
        }
    }

    private boolean isTaskExistInCommandSet(SqlTaskIds taskId) {
        for (SQLWithTaskId tasks : commandSet) {
            if (tasks.taskId == taskId) {
                return true;
            }
        }
        return false;
    }

    private boolean isTableExist(ArrayList<String> tableNames) {
        try (Statement stmt = SQLiteUtil.getStatement()) {
            for (String tableName : tableNames) {
                String sql= String.format("SELECT ASCII, PATH, PRIORITY FROM %s;", tableName);
                stmt.executeQuery(sql);
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private boolean isDatabaseDamaged() {
        ArrayList<String> list = new ArrayList<>();
        for (int i = 0; i <= 40; i++) {
            list.add("list" + i);
        }
        return !isTableExist(list);
    }

    private void recreateDatabase() {
        commandSet.clear();
        //删除所有表和索引

        boolean isDataDamaged = isDatabaseDamaged();

        for (int i = 0; i <= 40; i++) {
            if (isDataDamaged) {
                commandSet.add(new SQLWithTaskId(SqlTaskIds.DROP_TABLE, "DROP TABLE IF EXISTS list" + i + ";"));
            } else {
                commandSet.add(new SQLWithTaskId(SqlTaskIds.DELETE_FROM_LIST, "DELETE FROM list" + i + ";"));
            }
            commandSet.add(new SQLWithTaskId(SqlTaskIds.DROP_INDEX, "DROP INDEX IF EXISTS list" + i + "_index;"));
        }
        //创建新表
        String sql = "CREATE TABLE IF NOT EXISTS list";
        for (int i = 0; i <= 40; i++) {
            String command = sql + i + " " + "(ASCII INT, PATH text unique, PRIORITY INT)" + ";";
            commandSet.add(new SQLWithTaskId(SqlTaskIds.CREATE_TABLE, command));
        }
        executeImmediately();
    }

    private void checkTimeAndSendExecuteSqlSignalThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            // 时间检测线程
            final long updateTimeLimit = AllConfigs.getInstance().getUpdateTimeLimit();
            try {
                EventUtil eventUtil = EventUtil.getInstance();
                while (eventUtil.isNotMainExit()) {
                    if (status == Enums.DatabaseStatus.NORMAL) {
                        eventUtil.putEvent(new ExecuteSQLEvent());
                    }
                    TimeUnit.SECONDS.sleep(updateTimeLimit);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    public static void registerEventHandler() {
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.register(AddToCacheEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().addFileToCache(((AddToCacheEvent) event).path);
            }
        });

        eventUtil.register(DeleteFromCacheEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().removeFileFromCache(((DeleteFromCacheEvent) event).path);
            }
        });

        eventUtil.register(AddToDatabaseEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                for (Object each : ((AddToDatabaseEvent) event).getPaths()) {
                    getInstance().addFileToDatabase((String) each);
                }
            }
        });

        eventUtil.register(DeleteFromDatabaseEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                DeleteFromDatabaseEvent deleteFromDatabaseEvent = ((DeleteFromDatabaseEvent) event);
                for (Object each : deleteFromDatabaseEvent.getPaths()) {
                    getInstance().removeFileFromDatabase((String) each);
                }
            }
        });

        eventUtil.register(UpdateDatabaseEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().setStatus(Enums.DatabaseStatus.MANUAL_UPDATE);
                getInstance().updateLists(AllConfigs.getInstance().getIgnorePath(), AllConfigs.getInstance().getSearchDepth());
                getInstance().setStatus(Enums.DatabaseStatus.NORMAL);
            }
        });

        eventUtil.register(ExecuteSQLEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().executeImmediately();
            }
        });

        eventUtil.register(OptimiseDatabaseEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().setStatus(Enums.DatabaseStatus.VACUUM);
                //执行VACUUM命令
                try (PreparedStatement stmt = SQLiteUtil.getPreparedStatement("VACUUM;")) {
                    stmt.execute();
                } catch (Exception ex) {
                    ex.printStackTrace();
                } finally {
                    if (IsDebug.isDebug()) {
                        System.out.println("结束优化");
                    }
                    getInstance().setStatus(Enums.DatabaseStatus.NORMAL);
                }
            }
        });

        eventUtil.register(AddToSuffixPriorityMapEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                AddToSuffixPriorityMapEvent event1 = (AddToSuffixPriorityMapEvent) event;
                String suffix = event1.suffix;
                int priority = event1.priority;
                DatabaseUtil databaseUtil = getInstance();
                databaseUtil.addToCommandSet(
                        new SQLWithTaskId(SqlTaskIds.UPDATE_SUFFIX,
                                String.format("INSERT INTO priority VALUES(\"%s\", %d);", suffix, priority)));
            }
        });

        eventUtil.register(ClearSuffixPriorityMapEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                DatabaseUtil databaseUtil = getInstance();
                databaseUtil.addToCommandSet(new SQLWithTaskId(SqlTaskIds.UPDATE_SUFFIX, "DELETE FROM priority;"));
                databaseUtil.addToCommandSet(
                        new SQLWithTaskId(SqlTaskIds.UPDATE_SUFFIX, "INSERT INTO priority VALUES(\"defaultPriority\", 0);"));
            }
        });

        eventUtil.register(DeleteFromSuffixPriorityMapEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                DeleteFromSuffixPriorityMapEvent delete = (DeleteFromSuffixPriorityMapEvent) event;
                DatabaseUtil databaseUtil = getInstance();
                databaseUtil.addToCommandSet(new SQLWithTaskId(SqlTaskIds.UPDATE_SUFFIX,
                        String.format("DELETE FROM priority where SUFFIX=\"%s\"", delete.suffix)));
            }
        });

        eventUtil.register(UpdateSuffixPriorityEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                UpdateSuffixPriorityEvent update = (UpdateSuffixPriorityEvent) event;
                String origin = update.originSuffix;
                String newSuffix = update.newSuffix;
                int newNum = update.newPriority;
                eventUtil.putEvent(new DeleteFromSuffixPriorityMapEvent(origin));
                eventUtil.putEvent(new AddToSuffixPriorityMapEvent(newSuffix, newNum));
            }
        });
    }

    private static class SQLWithTaskId {
        private final String sql;
        private final SqlTaskIds taskId;

        private SQLWithTaskId(SqlTaskIds taskId, String sql) {
            this.sql = sql;
            this.taskId = taskId;
        }
    }

    private enum SqlTaskIds {
        DELETE_FROM_LIST, DELETE_FROM_CACHE, INSERT_TO_LIST, INSERT_TO_CACHE,
        CREATE_INDEX, CREATE_TABLE, DROP_TABLE, DROP_INDEX, UPDATE_SUFFIX
    }
}

