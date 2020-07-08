package SQLiteConfig;

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
        sqLiteConfig.setPageSize(8192);
        sqLiteConfig.setCacheSize(50000);
        sqLiteConfig.setSynchronous(SQLiteConfig.SynchronousMode.OFF);
        sqLiteConfig.setLockingMode(SQLiteConfig.LockingMode.NORMAL);
    }

    public static Statement getStatement(String url) throws SQLException {
        if (conn == null) {
            synchronized (SQLiteUtil.class) {
                if (conn == null) {
                    conn = getConnection(url);
                }
            }
        }
        return conn.createStatement();
    }

    private static Connection getConnection(String url) throws SQLException {
        if (!isInitialized) {
            init();
            isInitialized = true;
        }
        return DriverManager.getConnection(url, sqLiteConfig.toProperties());
    }
}
