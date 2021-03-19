package file.engine.event.handler.impl.frame.searchBar;

import file.engine.event.handler.Event;

public class SearchBarColorEvent extends Event {
    public final int color;

    protected SearchBarColorEvent(int color) {
        super();
        this.color = color;
    }
}
