package file.engine.event.handler.impl.frame.settingsFrame;

import file.engine.event.handler.Event;

public class IsCacheExistEvent extends Event {
    public final String cache;

    public IsCacheExistEvent(String cache) {
        this.cache = cache;
    }
}
