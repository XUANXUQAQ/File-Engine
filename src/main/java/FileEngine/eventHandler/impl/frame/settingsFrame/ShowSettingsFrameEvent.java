package FileEngine.eventHandler.impl.frame.settingsFrame;

import FileEngine.eventHandler.Event;

public class ShowSettingsFrameEvent extends Event {
    public String showTabName = null;
    public ShowSettingsFrameEvent() {
        super();
    }

    public ShowSettingsFrameEvent(String showTabName) {
        this.showTabName = showTabName;
    }
}
