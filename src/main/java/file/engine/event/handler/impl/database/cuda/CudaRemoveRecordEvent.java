package file.engine.event.handler.impl.database.cuda;

public class CudaRemoveRecordEvent extends CudaBaseEvent {
    public final String key;
    public final String record;

    public CudaRemoveRecordEvent(String key, String record) {
        super();
        this.key = key;
        this.record = record;
    }
}
