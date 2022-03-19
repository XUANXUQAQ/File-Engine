package file.engine.event.handler.impl.frame.searchBar;

import file.engine.event.handler.Event;

/**
 * 该类用于显示调用searchBar，若要监听显示searchBar事件，请使用@EventListener监听SearchBarReadyEvent
 * @see SearchBarReadyEvent
 */
public class ShowSearchBarEvent extends Event {
    public final boolean isGrabFocus;
    public final boolean isSwitchToNormal;

    public ShowSearchBarEvent(boolean isGrabFocus, boolean isSwitchToNormal) {
        super();
        this.isGrabFocus = isGrabFocus;
        this.isSwitchToNormal = isSwitchToNormal;
    }

    public ShowSearchBarEvent(boolean isGrabFocus) {
        super();
        this.isGrabFocus = isGrabFocus;
        this.isSwitchToNormal = false;
    }
}
