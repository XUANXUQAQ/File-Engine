package FileEngine.eventHandler.impl.database;

public class AddToDatabaseEvent extends DatabaseEvent {

    public AddToDatabaseEvent(String path) {
        super(path);
    }
}
