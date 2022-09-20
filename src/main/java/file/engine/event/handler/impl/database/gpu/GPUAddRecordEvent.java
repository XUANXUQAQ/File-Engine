package file.engine.event.handler.impl.database.gpu;

public class GPUAddRecordEvent extends GPUBaseEvent {

    public final String key;
    public final String record;

    public GPUAddRecordEvent(String key, String record) {
        super();
        this.key = key;
        this.record = record;
    }
}
