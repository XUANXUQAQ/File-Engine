package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

public class StartSearchEvent extends Event {

    public final String[] searchCase;
    public final String searchText;
    public final String[] keywords;

    public StartSearchEvent(String searchText, String[] searchCase, String[] keywords) {
        this.searchCase = searchCase;
        this.searchText = searchText;
        this.keywords = keywords;
    }
}
