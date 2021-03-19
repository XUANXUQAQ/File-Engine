package file.engine.event.handler.impl.frame.searchBar;

import file.engine.event.handler.Event;

public class SetSearchBarTransparencyEvent extends Event {
    public final float trans;

    public SetSearchBarTransparencyEvent(float trans) {
        super();
        this.trans = trans;
    }
}
