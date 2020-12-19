package FileEngine.eventHandler.impl;

import FileEngine.eventHandler.Event;

public class SetSwingLaf extends Event {
    public final String theme;

    public SetSwingLaf(String theme) {
        super();
        this.theme = theme;
    }
}
