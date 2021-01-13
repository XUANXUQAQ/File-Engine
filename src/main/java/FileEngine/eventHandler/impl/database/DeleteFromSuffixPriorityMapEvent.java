package FileEngine.eventHandler.impl.database;

import FileEngine.eventHandler.Event;

public class DeleteFromSuffixPriorityMapEvent extends Event {
    public final String suffix;

    public DeleteFromSuffixPriorityMapEvent(String suffix) {
        this.suffix = suffix;
    }
}
