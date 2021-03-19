package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

public class DeleteFromSuffixPriorityMapEvent extends Event {
    public final String suffix;

    public DeleteFromSuffixPriorityMapEvent(String suffix) {
        this.suffix = suffix;
    }
}
