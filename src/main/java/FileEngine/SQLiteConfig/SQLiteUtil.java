package FileEngine.SQLiteConfig;

import org.sqlite.SQLiteConfig;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class SQLiteUtil {
    private static SQLiteConfig sqLiteConfig;
    private static volatile boolean isInitialized = false;
    private static volatile Connection conn;

    private static void init() {
        sqLiteConfig = new SQLiteConfig();
        sqLiteConfig.setTempStore(SQLiteConfig.TempStore.MEMORY);
        sqLiteConfig.setJournalMode(SQLiteConfig.JournalMode.OFF);
        sqLiteConfig.setPageSize(16384);
        sqLiteConfig.setDefaultCacheSize(50000);
        sqLiteConfig.setSynchronous(SQLiteConfig.SynchronousMode.OFF);
        sqLiteConfig.setLockingMode(SQLiteConfig.LockingMode.NORMAL);
    }

    public static Statement getStatement() throws Exception {
        if (conn == null) {
            throw new Exception("The connection must be initialized first, call initConnection(String url)");
        }
        return conn.createStatement();
    }

    public static void initConnection(String url) throws SQLException {
        if (!isInitialized) {
            init();
            isInitialized = true;
        }
        conn = DriverManager.getConnection(url, sqLiteConfig.toProperties());
    }

    public static Connection getConnection() {
        return conn;
    }

    public static void createAllTables() throws Exception {
        String sql = "CREATE TABLE list";
        try (Statement stmt = getStatement()) {
            stmt.execute("BEGIN;");
            for (int i = 0; i <= 40; i++) {
                String command = sql + i + " " + "(PATH text unique)" + ";";
                stmt.executeUpdate(command);
            }
            stmt.execute("COMMIT;");
        }
    }
}
