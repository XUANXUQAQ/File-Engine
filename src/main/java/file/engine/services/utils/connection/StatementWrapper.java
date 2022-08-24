package file.engine.services.utils.connection;

import org.sqlite.SQLiteConnection;
import org.sqlite.jdbc4.JDBC4Statement;

import java.sql.SQLException;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 通过复写AutoCloseable接口的close方法实现引用计数，确保在关闭数据库时没有被使用
 * 必须使用 try-with-source语法
 */
class StatementWrapper extends JDBC4Statement {
    private final AtomicInteger connectionUsingCounter;

    public StatementWrapper(SQLiteConnection conn, AtomicInteger connectionUsingCounter) {
        super(conn);
        this.connectionUsingCounter = connectionUsingCounter;
        this.connectionUsingCounter.incrementAndGet();
    }

    @Override
    public void close() throws SQLException {
        super.close();
        connectionUsingCounter.decrementAndGet();
    }
}
