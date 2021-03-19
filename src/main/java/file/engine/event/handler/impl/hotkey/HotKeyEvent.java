package file.engine.event.handler.impl.hotkey;

import file.engine.event.handler.Event;

public class HotKeyEvent extends Event {
    public final String hotkey;

    protected HotKeyEvent(String hotkey) {
        super();
        this.hotkey = hotkey;
    }
}
