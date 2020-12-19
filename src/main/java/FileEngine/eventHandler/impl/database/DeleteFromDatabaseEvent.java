package FileEngine.eventHandler.impl.database;

public class DeleteFromDatabaseEvent extends DatabaseEvent {

    public DeleteFromDatabaseEvent(String path) {
        super(path);
    }
}
