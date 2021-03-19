package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

public class AddToSuffixPriorityMapEvent extends Event {
    public final String suffix;
    public final int priority;

    public AddToSuffixPriorityMapEvent(String suffix, int priority) {
        this.suffix = suffix;
        this.priority = priority;
    }
}
