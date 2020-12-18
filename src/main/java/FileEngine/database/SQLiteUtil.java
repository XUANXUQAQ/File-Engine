package FileEngine.database;

import org.sqlite.SQLiteConfig;

import java.sql.*;

/**
 * @author XUANXU
 */
public class SQLiteUtil {
    private static final SQLiteConfig sqLiteConfig = new SQLiteConfig();
    private static Connection conn;

    private static void initSqliteConfig() {
        sqLiteConfig.setTempStore(SQLiteConfig.TempStore.MEMORY);
        sqLiteConfig.setJournalMode(SQLiteConfig.JournalMode.OFF);
        sqLiteConfig.setPageSize(65535);
        sqLiteConfig.setDefaultCacheSize(256*1024);
        sqLiteConfig.setSynchronous(SQLiteConfig.SynchronousMode.OFF);
        sqLiteConfig.setLockingMode(SQLiteConfig.LockingMode.NORMAL);
    }

    /**
     * 仅用于select语句，以及需要及时生效的SQL语句
     * @param sql select语句
     * @return 已编译的PreparedStatement
     * @throws SQLException 失败
     */
    public static PreparedStatement getPreparedStatement(String sql) throws SQLException {
        if (conn == null) {
            throw new SQLException("The connection must be initialized first, call initConnection(String url)");
        }
        return conn.prepareStatement(sql);
    }

    /**
     * 用于需要重复运行多次指令的地方
     * @return Statement
     * @throws SQLException 失败
     */
    public static Statement getStatement() throws SQLException {
        if (conn == null) {
            throw new SQLException("The connection must be initialized first, call initConnection(String url)");
        }
        return conn.createStatement();
    }

    public static void initConnection(String url) throws SQLException {
        initSqliteConfig();
        conn = DriverManager.getConnection(url, sqLiteConfig.toProperties());
    }

    public static void closeAll() {
        try {
            conn.close();
        } catch (SQLException throwables) {
            throwables.printStackTrace();
        }
    }
}
