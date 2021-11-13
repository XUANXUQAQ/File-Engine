package file.engine.event.handler.impl.database;

import file.engine.event.handler.Event;

public class UpdateDatabaseEvent extends Event {

    public final boolean isDropPrevious;

    public UpdateDatabaseEvent(boolean isDropPrevious) {
        this.setBlock();
        this.isDropPrevious = isDropPrevious;
        setMaxRetryTimes(1);
    }
}
