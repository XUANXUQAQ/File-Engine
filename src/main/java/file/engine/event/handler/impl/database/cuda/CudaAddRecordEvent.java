package file.engine.event.handler.impl.database.cuda;

public class CudaAddRecordEvent extends CudaBaseEvent {

    public final String key;
    public final String record;

    public CudaAddRecordEvent(String key, String record) {
        super();
        this.key = key;
        this.record = record;
    }
}
