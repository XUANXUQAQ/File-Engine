package FileEngine.eventHandler.impl.daemon;

import FileEngine.eventHandler.Event;

public class DaemonEvent extends Event {
    public final String currentWorkingDir;

    protected DaemonEvent(String currentWorkingDir) {
        super();
        this.currentWorkingDir = currentWorkingDir;
    }
}
