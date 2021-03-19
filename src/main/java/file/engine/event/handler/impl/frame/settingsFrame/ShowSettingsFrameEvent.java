package file.engine.event.handler.impl.frame.settingsFrame;

import file.engine.event.handler.Event;

public class ShowSettingsFrameEvent extends Event {
    public String showTabName = null;
    public ShowSettingsFrameEvent() {
        super();
    }

    public ShowSettingsFrameEvent(String showTabName) {
        this.showTabName = showTabName;
    }
}
