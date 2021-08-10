package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

import java.util.LinkedHashSet;

public class DeleteFromDatabaseEvent extends Event {
    private final LinkedHashSet<String> paths;

    public Object[] getPaths() {
        return paths.toArray();
    }

    public DeleteFromDatabaseEvent(LinkedHashSet<String> paths) {
        this.paths = paths;
    }
}
