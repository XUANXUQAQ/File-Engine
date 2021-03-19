package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

public class DatabaseEvent extends Event {
    public final String path;

    protected DatabaseEvent(String path) {
        super();
        this.path = path;
        this.setBlock();
    }
}
