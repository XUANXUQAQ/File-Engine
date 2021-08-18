package file.engine.event.handler.impl.frame.searchBar;

import file.engine.event.handler.Event;

public class SearchBarReadyEvent extends Event {

    public final String showingType;

    public SearchBarReadyEvent(String showingType) {
        this.showingType = showingType;
    }
}
