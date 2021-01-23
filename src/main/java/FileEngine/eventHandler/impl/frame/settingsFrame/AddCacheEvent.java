package FileEngine.eventHandler.impl.frame.settingsFrame;

import FileEngine.eventHandler.Event;

public class AddCacheEvent extends Event {
    public final String cache;

    public AddCacheEvent(String cache) {
        this.cache = cache;
    }
}
