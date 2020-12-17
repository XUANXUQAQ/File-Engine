package FileEngine.taskHandler.impl.database;

import FileEngine.taskHandler.Task;

public class DatabaseTask extends Task {
    public final String path;

    protected DatabaseTask(String path) {
        super();
        this.path = path;
    }
}
