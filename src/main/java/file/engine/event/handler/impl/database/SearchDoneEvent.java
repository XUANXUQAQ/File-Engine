package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

import java.util.concurrent.ConcurrentLinkedQueue;

public class SearchDoneEvent extends Event {

    @SuppressWarnings({"FieldCanBeLocal", "unused"})
    //为插件获取搜索结果提供支持
    public final ConcurrentLinkedQueue<String> searchResults;

    public SearchDoneEvent(ConcurrentLinkedQueue<String> searchResults) {
        this.searchResults = searchResults;
    }
}
