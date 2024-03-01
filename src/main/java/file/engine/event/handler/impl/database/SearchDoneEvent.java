package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.concurrent.ConcurrentLinkedQueue;

@EqualsAndHashCode(callSuper = true)
@Data
public class SearchDoneEvent extends Event {

    public final ConcurrentLinkedQueue<String> searchResults;
}
