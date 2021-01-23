package FileEngine.eventHandler.impl.frame.settingsFrame;

import FileEngine.eventHandler.Event;

public class IsCacheExistEvent extends Event {
    public final String cache;

    public IsCacheExistEvent(String cache) {
        this.cache = cache;
    }
}
