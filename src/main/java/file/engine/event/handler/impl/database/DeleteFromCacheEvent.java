package file.engine.event.handler.impl.database;

public class DeleteFromCacheEvent extends DatabaseEvent {

    public DeleteFromCacheEvent(String path) {
        super(path);
        this.setBlock();
    }
}
