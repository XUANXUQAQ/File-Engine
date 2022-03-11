package file.engine.event.handler.impl.hotkey;

public class CheckHotKeyAvailableEvent extends HotKeyEvent {
    public CheckHotKeyAvailableEvent(String hotkey) {
        super(hotkey);
    }
}
