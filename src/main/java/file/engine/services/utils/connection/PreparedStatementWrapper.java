package file.engine.services.utils.connection;

import org.sqlite.SQLiteConnection;
import org.sqlite.jdbc4.JDBC4PreparedStatement;

import java.sql.SQLException;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 通过复写AutoCloseable接口的close方法实现引用计数，确保在关闭数据库时没有被使用
 * 必须使用 try-with-source语法
 */
class PreparedStatementWrapper extends JDBC4PreparedStatement {
    private final AtomicInteger connectionUsingCounter;

    public PreparedStatementWrapper(SQLiteConnection conn, String sql, AtomicInteger connectionUsingCounter) throws SQLException {
        super(conn, sql);
        this.connectionUsingCounter = connectionUsingCounter;
        this.connectionUsingCounter.incrementAndGet();
    }

    @Override
    public void close() throws SQLException {
        super.close();
        connectionUsingCounter.decrementAndGet();
    }
}
