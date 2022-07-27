package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;
import lombok.NonNull;

public class StartSearchEvent extends Event {

    public final String[] searchCase;
    public final String searchText;
    public final String[] keywords;

    public StartSearchEvent(@NonNull String searchText, @NonNull String[] searchCase, @NonNull String[] keywords) {
        this.searchCase = searchCase;
        this.searchText = searchText;
        this.keywords = keywords;
    }
}
