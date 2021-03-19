package file.engine.event.handler.impl.daemon;

import file.engine.event.handler.Event;

public class DaemonEvent extends Event {
    public final String currentWorkingDir;

    protected DaemonEvent(String currentWorkingDir) {
        super();
        this.currentWorkingDir = currentWorkingDir;
    }
}
