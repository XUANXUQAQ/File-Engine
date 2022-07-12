package file.engine.utils.connection;

import org.sqlite.SQLiteConnection;
import org.sqlite.jdbc4.JDBC4PreparedStatement;

import java.sql.SQLException;
import java.util.concurrent.atomic.AtomicInteger;

public class PreparedStatementWrapper extends JDBC4PreparedStatement {
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
