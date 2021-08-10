package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

public class DeleteFromCacheEvent extends Event {
    public final String path;

    public DeleteFromCacheEvent(String path) {
        this.path = path;
    }
}
