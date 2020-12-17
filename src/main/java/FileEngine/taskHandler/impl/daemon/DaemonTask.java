package FileEngine.taskHandler.impl.daemon;

import FileEngine.taskHandler.Task;

public class DaemonTask extends Task {
    public final String currentWorkingDir;

    protected DaemonTask(String currentWorkingDir) {
        super();
        this.currentWorkingDir = currentWorkingDir;
    }
}
