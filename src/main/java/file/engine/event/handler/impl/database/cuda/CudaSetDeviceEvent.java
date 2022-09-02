package file.engine.event.handler.impl.database.cuda;

public class CudaSetDeviceEvent extends CudaBaseEvent {
    public final int deviceNum;

    public CudaSetDeviceEvent(int deviceNum) {
        this.deviceNum = deviceNum;
    }
}
