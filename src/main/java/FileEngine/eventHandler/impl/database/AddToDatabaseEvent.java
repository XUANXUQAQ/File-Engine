package FileEngine.eventHandler.impl.database;

import java.util.LinkedHashSet;

public class AddToDatabaseEvent extends DatabaseEvent {
    private final LinkedHashSet<String> paths;

    public Object[] getPaths() {
        return paths.toArray();
    }

    public AddToDatabaseEvent(LinkedHashSet<String> paths) {
        super(null);
        this.paths = paths;
        this.setBlock();
    }
}
