package file.engine.event.handler.impl.database;

public class ExecuteSQLEvent extends DatabaseEvent {
    public ExecuteSQLEvent() {
        super(null);
        this.setBlock();
    }
}
