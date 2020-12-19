package FileEngine.eventHandler.impl.frame.searchBar;

import FileEngine.eventHandler.Event;

public class ShowSearchBarEvent extends Event {
    public final boolean isGrabFocus;

    public ShowSearchBarEvent(boolean isGrabFocus) {
        super();
        this.isGrabFocus = isGrabFocus;
    }
}
