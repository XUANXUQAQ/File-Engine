package file.engine.utils.connection;

import org.sqlite.SQLiteConnection;
import org.sqlite.jdbc4.JDBC4Statement;

import java.sql.SQLException;
import java.util.concurrent.atomic.AtomicInteger;

public class StatementWrapper extends JDBC4Statement {
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
