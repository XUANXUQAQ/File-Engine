package file.engine.event.handler.impl.hotkey;

import file.engine.event.handler.Event;

public class CheckHotKeyAvailableEvent extends Event {
    public final String hotkey;

    public CheckHotKeyAvailableEvent(String hotkey) {
        this.hotkey = hotkey;
    }
}
