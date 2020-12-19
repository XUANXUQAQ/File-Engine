package FileEngine.eventHandler.impl.daemon;

public class StartDaemonEvent extends DaemonEvent {

    public StartDaemonEvent(String currentWorkingDir) {
        super(currentWorkingDir);
    }
}
