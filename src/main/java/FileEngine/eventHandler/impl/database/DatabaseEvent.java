package FileEngine.eventHandler.impl.database;

import FileEngine.eventHandler.Event;

public class DatabaseEvent extends Event {
    public final String path;

    protected DatabaseEvent(String path) {
        super();
        this.path = path;
        this.setBlock();
    }
}
