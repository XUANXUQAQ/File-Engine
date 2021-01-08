package FileEngine.eventHandler.impl.taskbar;

import FileEngine.eventHandler.Event;

public class ShowTaskBarMessageEvent extends Event {
    public final String caption;
    public final String message;
    public final Event event;

    public ShowTaskBarMessageEvent(String caption, String message) {
        this.caption = caption;
        this.message = message;
        this.event = null;
    }

    public ShowTaskBarMessageEvent(String caption, String message, Event event) {
        super();
        this.caption = caption;
        this.message = message;
        this.event = event;
    }
}
