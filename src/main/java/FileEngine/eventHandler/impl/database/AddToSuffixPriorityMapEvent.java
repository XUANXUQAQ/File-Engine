package FileEngine.eventHandler.impl.database;

import FileEngine.eventHandler.Event;

public class AddToSuffixPriorityMapEvent extends Event {
    public final String suffix;
    public final int priority;

    public AddToSuffixPriorityMapEvent(String suffix, int priority) {
        this.suffix = suffix;
        this.priority = priority;
    }
}
