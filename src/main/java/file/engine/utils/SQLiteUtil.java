package file.engine.utils;

import file.engine.utils.system.properties.IsDebug;
import file.engine.configs.AllConfigs;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.stop.RestartEvent;
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

/**
 * @author XUANXU
 */
public class SQLiteUtil {
    private static final SQLiteConfig sqLiteConfig = new SQLiteConfig();
    private static final ConcurrentHashMap<String, Connection> connectionPool = new ConcurrentHashMap<>();

    private static void initSqliteConfig() {
        sqLiteConfig.setTempStore(SQLiteConfig.TempStore.MEMORY);
        sqLiteConfig.setJournalMode(SQLiteConfig.JournalMode.WAL);
        sqLiteConfig.setPageSize(65535);
        sqLiteConfig.setOpenMode(SQLiteOpenMode.NOMUTEX);
        sqLiteConfig.setDefaultCacheSize(256 * 1024);
        sqLiteConfig.setSynchronous(SQLiteConfig.SynchronousMode.OFF);
        sqLiteConfig.setLockingMode(SQLiteConfig.LockingMode.NORMAL);
    }

    /**
     * 仅用于select语句，以及需要及时生效的SQL语句
     *
     * @param sql select语句
     * @return 已编译的PreparedStatement
     * @throws SQLException 失败
     */
    public static PreparedStatement getPreparedStatement(String sql, String key) throws SQLException {
        checkEmpty(key);
        return connectionPool.get(key).prepareStatement(sql);
    }

    /**
     * 用于需要重复运行多次指令的地方
     *
     * @return Statement
     * @throws SQLException 失败
     */
    public static Statement getStatement(String key) throws SQLException {
        checkEmpty(key);
        return connectionPool.get(key).createStatement();
    }

    private static void checkEmpty(String key) {
        if (connectionPool.isEmpty()) {
            throw new RuntimeException("The connection must be initialized first, call initConnection(String url)");
        }
        if (!connectionPool.containsKey(key)) {
            throw new RuntimeException("key doesn't exist");
        }
    }

    public static void initConnection(String url, String key) throws SQLException {
        initSqliteConfig();
        Connection conn = DriverManager.getConnection(url, sqLiteConfig.toProperties());
        connectionPool.put(key, conn);
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
                v.close();
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
        String disks = AllConfigs.getInstance().getDisks();
        if (disks == null || disks.isEmpty() || disks.isBlank()) {
            throw new RuntimeException("initialize failed");
        }
        return disks;
    }

    public static void initAllConnections() {
        deleteMalFormedFile();
        String[] split = RegexUtil.comma.split(initialize());
        ArrayList<File> malformedFiles = new ArrayList<>();
        for (String eachDisk : split) {
            File data = new File("data", eachDisk.charAt(0) + ".db");
            try {
                initConnection("jdbc:sqlite:" + data.getAbsolutePath(), String.valueOf(eachDisk.charAt(0)));
            } catch (Exception e) {
                malformedFiles.add(data);
            }
        }

        File cache = new File("data", "cache.db");
        try {
            initConnection("jdbc:sqlite:" + cache.getAbsolutePath(), "cache");
            createCacheTable();
            createPriorityTable();
        } catch (SQLException e) {
            malformedFiles.add(cache);
        }
        File weight = new File("data", "weight.db");
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
