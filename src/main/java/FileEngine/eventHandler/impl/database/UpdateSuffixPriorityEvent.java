package FileEngine.eventHandler.impl.database;

import FileEngine.eventHandler.Event;

public class UpdateSuffixPriorityEvent extends Event {
    public final String originSuffix;
    public final String newSuffix;
    public final int newPriority;

    public UpdateSuffixPriorityEvent(String originSuffix, String newSuffix, int newPriority) {
        this.originSuffix = originSuffix;
        this.newPriority = newPriority;
        this.newSuffix = newSuffix;
    }
}
