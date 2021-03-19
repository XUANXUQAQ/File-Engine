package file.engine.event.handler.impl.frame.searchBar;

import file.engine.event.handler.Event;

public class ShowSearchBarEvent extends Event {
    public final boolean isGrabFocus;

    public ShowSearchBarEvent(boolean isGrabFocus) {
        super();
        this.isGrabFocus = isGrabFocus;
    }
}
