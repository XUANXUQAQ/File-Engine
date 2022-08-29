package file.engine.event.handler.impl.database;

import lombok.NonNull;

import java.util.function.Supplier;

public class PrepareCudaSearchEvent extends StartSearchEvent {
    public PrepareCudaSearchEvent(@NonNull Supplier<String> searchText, @NonNull Supplier<String[]> searchCase, @NonNull Supplier<String[]> keywords) {
        super(searchText, searchCase, keywords);
    }
}
