package file.engine.event.handler.impl.database;

public class AddToCacheEvent extends DatabaseEvent {

    public AddToCacheEvent(String path) {
        super(path);
        this.setBlock();
    }
}
