package file.engine.event.handler.impl.plugin;

import file.engine.event.handler.Event;

public final class EventProcessedBroadcastEvent extends Event {

    public final Class<?> c;
    public final Object eventInstance;

    public EventProcessedBroadcastEvent(Class<?> c, Object eventInstance) {
        super();
        this.c = c;
        this.eventInstance = eventInstance;
    }
}
