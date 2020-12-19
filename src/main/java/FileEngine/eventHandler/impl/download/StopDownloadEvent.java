package FileEngine.eventHandler.impl.download;

public class StopDownloadEvent extends DownloadEvent {

    public StopDownloadEvent(String fileName) {
        super(null, fileName, null);
    }
}
