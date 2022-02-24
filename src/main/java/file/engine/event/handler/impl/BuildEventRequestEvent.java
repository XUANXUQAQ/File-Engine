package file.engine.event.handler.impl;

import file.engine.event.handler.Event;

public class BuildEventRequestEvent extends Event {
    public final Object[] eventInfo;

    public BuildEventRequestEvent(Object[] eventInfo) {
        this.eventInfo = eventInfo;
    }
}
