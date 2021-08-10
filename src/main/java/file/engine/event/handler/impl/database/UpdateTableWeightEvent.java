package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

public class UpdateTableWeightEvent extends Event {
    public final String tableName;
    public final long tableWeight;

    public UpdateTableWeightEvent(String tableName, long tableWeight) {
        this.tableName = tableName;
        this.tableWeight = tableWeight;
    }
}
