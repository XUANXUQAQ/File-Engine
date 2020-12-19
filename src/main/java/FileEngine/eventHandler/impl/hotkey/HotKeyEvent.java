package FileEngine.eventHandler.impl.hotkey;

import FileEngine.eventHandler.Event;

public class HotKeyEvent extends Event {
    public final String hotkey;

    protected HotKeyEvent(String hotkey) {
        super();
        this.hotkey = hotkey;
    }
}
