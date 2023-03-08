package file.engine.event.handler.impl.frame.searchBar;

public class SwitchVisibleStatusEvent extends ShowSearchBarEvent {
    public SwitchVisibleStatusEvent(boolean isGrabFocus, boolean isSwitchToNormal) {
        super(isGrabFocus, isSwitchToNormal);
    }

    public SwitchVisibleStatusEvent(boolean isGrabFocus) {
        super(isGrabFocus);
    }
}
