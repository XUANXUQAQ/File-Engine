package FileEngine.eventHandler.impl.taskbar;

import FileEngine.eventHandler.Event;

public class ShowTaskBarMessageEvent extends Event {
    public final String caption;
    public final String message;

    public ShowTaskBarMessageEvent(String caption, String message) {
        super();
        this.caption = caption;
        this.message = message;
    }
}
