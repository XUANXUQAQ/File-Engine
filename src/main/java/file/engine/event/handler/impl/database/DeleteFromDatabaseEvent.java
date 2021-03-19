package file.engine.event.handler.impl.database;

import java.util.LinkedHashSet;

public class DeleteFromDatabaseEvent extends DatabaseEvent {
    private final LinkedHashSet<String> paths;

    public Object[] getPaths() {
        return paths.toArray();
    }

    public DeleteFromDatabaseEvent(LinkedHashSet<String> paths) {
        super(null);
        this.paths = paths;
        this.setBlock();
    }
}
