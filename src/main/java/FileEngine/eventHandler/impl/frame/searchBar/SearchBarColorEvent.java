package FileEngine.eventHandler.impl.frame.searchBar;

import FileEngine.eventHandler.Event;

public class SearchBarColorEvent extends Event {
    public final int color;

    protected SearchBarColorEvent(int color) {
        super();
        this.color = color;
    }
}
