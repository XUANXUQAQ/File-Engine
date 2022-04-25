package file.engine.utils;

import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.utils.file.FileUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.SneakyThrows;
import org.sqlite.SQLiteConfig;
import org.sqlite.SQLiteOpenMode;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/**
 * @author XUANXU
 */
public class SQLiteUtil {
    private static final SQLiteConfig sqLiteConfig = new SQLiteConfig();
    private static final ConcurrentHashMap<String, ConnectionWrapper> connectionPool = new ConcurrentHashMap<>();
    private static String currentDatabaseDir = "data";

    static {
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        cachedThreadPoolUtil.executeTask(() -> {
            try {
                while (!cachedThreadPoolUtil.isShutdown()) {
                    for (ConnectionWrapper conn : connectionPool.values()) {
                        long currentTimeMillis = System.currentTimeMillis();
                        try {
                            if (currentTimeMillis - conn.usingTimeMills > Constants.CLOSE_DATABASE_TIMEOUT_MILLS && !conn.connection.isClosed()) {
                                synchronized (SQLiteUtil.class) {
                                    if (currentTimeMillis - conn.usingTimeMills > Constants.CLOSE_DATABASE_TIMEOUT_MILLS && !conn.connection.isClosed()) {
                                        System.out.println("长时间未使用 " + conn.url + "  已关闭连接");
                                        conn.connection.close();
                                    }
                                }
                            }
                        } catch (SQLException e) {
                            e.printStackTrace();
                        }
                    }
                    TimeUnit.SECONDS.sleep(1);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    private static class ConnectionWrapper {
        private final String url;
        private Connection connection;
        private volatile long usingTimeMills;

        private ConnectionWrapper(String url) {
            this.url = url;
            try {
                this.connection = DriverManager.getConnection(url, sqLiteConfig.toProperties());
            } catch (SQLException e) {
                e.printStackTrace();
            }
            this.usingTimeMills = System.currentTimeMillis();
        }
    }

    private static void initSqliteConfig() {
        sqLiteConfig.setTempStore(SQLiteConfig.TempStore.FILE);
        sqLiteConfig.setJournalMode(SQLiteConfig.JournalMode.WAL);
        sqLiteConfig.setOpenMode(SQLiteOpenMode.NOMUTEX);
        sqLiteConfig.setSynchronous(SQLiteConfig.SynchronousMode.OFF);
        sqLiteConfig.setLockingMode(SQLiteConfig.LockingMode.NORMAL);
    }

    @SneakyThrows
    private static Connection getFromConnectionPool(String key) {
        ConnectionWrapper connectionWrapper = connectionPool.get(key);
        if (connectionWrapper == null) {
            throw new RuntimeException("no connection named " + key);
        }
        synchronized (SQLiteUtil.class) {
            if (connectionWrapper.connection.isClosed()) {
                connectionWrapper.connection = DriverManager.getConnection(connectionWrapper.url, sqLiteConfig.toProperties());
                System.out.println("已恢复连接 " + connectionWrapper.url);
            }
            connectionWrapper.usingTimeMills = System.currentTimeMillis();
        }
        return connectionWrapper.connection;
    }

    /**
     * 仅用于select语句，以及需要及时生效的SQL语句
     *
     * @param sql select语句
     * @return 已编译的PreparedStatement
     * @throws SQLException 失败
     */
    public static PreparedStatement getPreparedStatement(String sql, String key) throws SQLException {
        if (isConnectionNotInitialized(key)) {
            File data = new File(currentDatabaseDir, key + ".db");
            initConnection("jdbc:sqlite:" + data.getAbsolutePath(), key);
        }
        return getFromConnectionPool(key).prepareStatement(sql);
    }

    /**
     * 用于需要重复运行多次指令的地方
     *
     * @return Statement
     * @throws SQLException 失败
     */
    public static Statement getStatement(String key) throws SQLException {
        if (isConnectionNotInitialized(key)) {
            File data = new File(currentDatabaseDir, key + ".db");
            initConnection("jdbc:sqlite:" + data.getAbsolutePath(), key);
        }
        return getFromConnectionPool(key).createStatement();
    }

    private static boolean isConnectionNotInitialized(String key) {
        if (connectionPool.isEmpty()) {
            throw new RuntimeException("The connection must be initialized first, call initConnection(String url)");
        }
        return !connectionPool.containsKey(key);
    }

    public static void initConnection(String url, String key) throws SQLException {
        initSqliteConfig();
        ConnectionWrapper connectionWrapper = new ConnectionWrapper(url);
        connectionPool.put(key, connectionWrapper);
    }

    /**
     * 关闭所有数据库连接
     */
    public static void closeAll() {
        if (IsDebug.isDebug()) {
            System.err.println("正在关闭数据库连接");
        }
        connectionPool.forEach((k, v) -> {
            try {
                v.connection.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        });
        connectionPool.clear();
    }

    private static void deleteMalFormedFile() {
        if (Files.exists(Path.of("user/malformedDB"))) {
            String line;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream("user/malformedDB"), StandardCharsets.UTF_8))) {
                while ((line = reader.readLine()) != null) {
                    if (line.isEmpty() || line.isBlank()) {
                        continue;
                    }
                    Files.delete(Path.of(line));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                Files.delete(Path.of("user/malformedDB"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static String initialize() {
        if (!Files.exists(Path.of("data"))) {
            try {
                Files.createDirectories(Path.of("data"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        String disks = AllConfigs.getInstance().getAvailableDisks();
        if (disks == null || disks.isEmpty() || disks.isBlank()) {
            throw new RuntimeException("initialize failed");
        }
        return disks;
    }

    /**
     * 复制数据库文件到另一个文件夹
     *
     * @param fromDir 源路径
     * @param toDir   目标路径
     */
    public static void copyDatabases(String fromDir, String toDir) {
        File cache = new File(fromDir, "cache.db");
        FileUtil.copyFile(cache, new File(toDir, "cache.db"));
        File weight = new File(fromDir, "weight.db");
        FileUtil.copyFile(weight, new File(toDir, "weight.db"));
        String[] split = RegexUtil.comma.split(initialize());
        for (String eachDisk : split) {
            String dbName = eachDisk.charAt(0) + ".db";
            File data = new File(fromDir, dbName);
            FileUtil.copyFile(data, new File(toDir, dbName));
        }
    }

    /**
     * 检查数据库中表是否为空
     *
     * @param tableNames 所有待检测的表名
     * @return true如果超过10个表结果都不超过10条
     */
    private static boolean isDatabaseEmpty(ArrayList<String> tableNames) throws SQLException {
        int emptyNum = 0;
        for (String each : RegexUtil.comma.split(AllConfigs.getInstance().getAvailableDisks())) {
            try (Statement stmt = SQLiteUtil.getStatement(String.valueOf(each.charAt(0)))) {
                for (String tableName : tableNames) {
                    String sql = String.format("SELECT ASCII FROM %s LIMIT 10;", tableName);
                    try (ResultSet resultSet = stmt.executeQuery(sql)) {
                        int count = 0;
                        while (resultSet.next()) {
                            count++;
                        }
                        if (count < 10) {
                            emptyNum++;
                        }
                    }
                }
            }
        }
        return emptyNum > 10;
    }

    /**
     * 检查数据库是否损坏
     *
     * @return boolean
     */
    public static boolean isDatabaseDamaged() {
        try {
            ArrayList<String> list = new ArrayList<>();
            for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
                list.add("list" + i);
            }
            return isDatabaseEmpty(list);
        } catch (Exception e) {
            e.printStackTrace();
            return true;
        }
    }

    public static void initAllConnections() {
        initAllConnections("data");
    }

    public static void initAllConnections(String dir) {
        currentDatabaseDir = dir;
        deleteMalFormedFile();
        String[] split = RegexUtil.comma.split(initialize());
        ArrayList<File> malformedFiles = new ArrayList<>();
        for (String eachDisk : split) {
            File data = new File(dir, eachDisk.charAt(0) + ".db");
            try {
                initConnection("jdbc:sqlite:" + data.getAbsolutePath(), String.valueOf(eachDisk.charAt(0)));
                initTables(String.valueOf(eachDisk.charAt(0)));
            } catch (Exception e) {
                malformedFiles.add(data);
            }
        }

        File cache = new File(dir, "cache.db");
        try {
            initConnection("jdbc:sqlite:" + cache.getAbsolutePath(), "cache");
            createCacheTable();
            createPriorityTable();
        } catch (SQLException e) {
            malformedFiles.add(cache);
        }
        File weight = new File(dir, "weight.db");
        try {
            initConnection("jdbc:sqlite:" + weight.getAbsolutePath(), "weight");
            createWeightTable();
        } catch (SQLException exception) {
            malformedFiles.add(weight);
        }
        if (!malformedFiles.isEmpty()) {
            try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/malformedDB"), StandardCharsets.UTF_8))) {
                for (File file : malformedFiles) {
                    writer.write(file.getAbsolutePath());
                    writer.newLine();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            EventManagement.getInstance().putEvent(new RestartEvent());
        }
    }

    /**
     * 检查表是否存在
     *
     * @return true or false
     */
    @SuppressWarnings("SameParameterValue")
    private static boolean isTableExist(String tableName, String key) {
        try (Statement p = getStatement(key)) {
            p.execute(String.format("SELECT * FROM %s", tableName));
            return true;
        } catch (SQLException exception) {
            return false;
        }
    }

    private static void createWeightTable() throws SQLException {
        try (PreparedStatement pStmt = getPreparedStatement("CREATE TABLE IF NOT EXISTS weight(TABLE_NAME text unique, TABLE_WEIGHT INT)", "weight")) {
            pStmt.executeUpdate();
        }
        try (Statement stmt = getStatement("weight")) {
            for (int i = 0; i < 41; i++) {
                String tableName = "list" + i;
                String format = String.format("INSERT OR IGNORE INTO weight values(\"%s\", %d)", tableName, 0);
                stmt.executeUpdate(format);
            }
        }
    }

    /**
     * 初始化表
     *
     * @param disk disk
     */
    private static void initTables(String disk) {
        try (Statement stmt = getStatement(disk)) {
            for (int i = 0; i < 41; i++) {
                stmt.executeUpdate("CREATE TABLE IF NOT EXISTS list" + i + "(ASCII INT, PATH TEXT, PRIORITY INT)");
            }
        } catch (SQLException exception) {
            exception.printStackTrace();
        }
    }

    private static void createPriorityTable() throws SQLException {
        if (isTableExist("priority", "cache")) {
            return;
        }
        try (Statement statement = getStatement("cache")) {
            int row = statement.executeUpdate("CREATE TABLE IF NOT EXISTS priority(SUFFIX text unique, PRIORITY INT)");
            if (row == 0) {
                int count = 10;
                HashMap<String, Integer> map = new HashMap<>();
                map.put("lnk", count--);
                map.put("exe", count--);
                map.put("bat", count--);
                map.put("cmd", count--);
                map.put("txt", count--);
                map.put("docx", count--);
                map.put("zip", count--);
                map.put("rar", count--);
                map.put("7z", count--);
                map.put("html", count);
                map.put("defaultPriority", 0);
                insertAllSuffixPriority(map, statement);
            }
        }
    }

    private static void createCacheTable() throws SQLException {
        try (PreparedStatement pStmt = getPreparedStatement("CREATE TABLE IF NOT EXISTS cache(PATH text unique);", "cache")) {
            pStmt.executeUpdate();
        }
    }

    private static void insertAllSuffixPriority(HashMap<String, Integer> suffixMap, Statement statement) {
        try {
            statement.execute("BEGIN;");
            suffixMap.forEach((suffix, priority) -> {
                String generateFormattedSql = generateFormattedSql(suffix, priority);
                try {
                    statement.execute(generateFormattedSql);
                } catch (SQLException e) {
                    e.printStackTrace();
                }
            });
        } catch (SQLException throwable) {
            throwable.printStackTrace();
        } finally {
            try {
                statement.execute("COMMIT;");
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    private static String generateFormattedSql(String suffix, int priority) {
        return String.format("INSERT OR IGNORE INTO priority VALUES(\"%s\", %d)", suffix, priority);
    }
}
