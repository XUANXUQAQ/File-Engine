package file.engine.event.handler.impl.database.gpu;

public class GPURemoveRecordEvent extends GPUBaseEvent {
    public final String key;
    public final String record;

    public GPURemoveRecordEvent(String key, String record) {
        super();
        this.key = key;
        this.record = record;
    }
}
