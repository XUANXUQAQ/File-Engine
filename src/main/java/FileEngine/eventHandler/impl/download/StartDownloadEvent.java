package FileEngine.eventHandler.impl.download;

public class StartDownloadEvent extends DownloadEvent {

    public StartDownloadEvent(String url, String fileName, String savePath) {
        super(url, fileName, savePath);
    }
}
