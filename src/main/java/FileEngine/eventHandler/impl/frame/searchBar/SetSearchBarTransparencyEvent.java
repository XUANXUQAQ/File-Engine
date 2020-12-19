package FileEngine.eventHandler.impl.frame.searchBar;

import FileEngine.eventHandler.Event;

public class SetSearchBarTransparencyEvent extends Event {
    public final float trans;

    public SetSearchBarTransparencyEvent(float trans) {
        super();
        this.trans = trans;
    }
}
