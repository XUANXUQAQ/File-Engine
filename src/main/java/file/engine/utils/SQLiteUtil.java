package file.engine.utils;

import file.engine.IsDebug;
import file.engine.configs.AllConfigs;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.stop.RestartEvent;
import org.sqlite.SQLiteConfig;

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
        if (!connectionPool.containsKey(key)) {
            throw new RuntimeException("key doesn't exist");
        }
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
        if (!connectionPool.containsKey(key)) {
            throw new RuntimeException("key doesn't exist");
        }
        return connectionPool.get(key).createStatement();
    }

    private static void checkEmpty(String key) {
        if (connectionPool.isEmpty() || !connectionPool.containsKey(key)) {
            throw new RuntimeException("The connection must be initialized first, call initConnection(String url)");
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

    private static void createPriorityTable() throws SQLException {
        try (Statement statement = SQLiteUtil.getStatement("cache")) {
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
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement("CREATE TABLE IF NOT EXISTS cache(PATH text unique);", "cache")) {
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
