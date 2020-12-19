package FileEngine.eventHandler.impl.database;

public class AddToCacheEvent extends DatabaseEvent {

    public AddToCacheEvent(String path) {
        super(path);
    }
}
