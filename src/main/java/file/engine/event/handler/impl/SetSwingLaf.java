package file.engine.event.handler.impl;

import file.engine.event.handler.Event;

public class SetSwingLaf extends Event {
    public final String theme;

    public SetSwingLaf(String theme) {
        super();
        this.theme = theme;
    }
}
