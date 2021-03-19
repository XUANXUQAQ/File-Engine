package file.engine.event.handler.impl.hotkey;

public class ResponseCtrlEvent extends HotKeyEvent{
    public final boolean isResponse;
    public ResponseCtrlEvent(boolean b) {
        super(null);
        this.isResponse = b;
    }
}
