package file.engine.event.handler.impl.open.file;

import file.engine.event.handler.Event;

public class OpenFileEvent extends Event {

    public final OpenStatus openStatus;
    public final String path;

    public enum OpenStatus {
        NORMAL_OPEN, WITH_ADMIN, LAST_DIR
    }

    public OpenFileEvent(OpenStatus openStatus, String path) {
        this.openStatus = openStatus;
        this.path = path;
    }
}
