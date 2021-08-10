package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

public class AddToCacheEvent extends Event {
    public final String path;

    public AddToCacheEvent(String path) {
        this.path = path;
    }
}
