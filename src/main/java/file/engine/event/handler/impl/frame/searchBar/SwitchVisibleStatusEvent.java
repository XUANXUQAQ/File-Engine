package file.engine.event.handler.impl.frame.searchBar;

public class SwitchVisibleStatusEvent extends ShowSearchBarEvent {

    public SwitchVisibleStatusEvent(boolean isGrabFocus) {
        super(isGrabFocus);
    }
}
