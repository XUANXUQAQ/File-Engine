package file.engine.event.handler.impl.frame.settingsFrame;

import file.engine.event.handler.Event;

public class AddCacheEvent extends Event {
    public final String cache;

    public AddCacheEvent(String cache) {
        this.cache = cache;
    }
}
