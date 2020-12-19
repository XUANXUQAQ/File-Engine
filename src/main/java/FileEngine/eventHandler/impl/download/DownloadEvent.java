package FileEngine.eventHandler.impl.download;

import FileEngine.eventHandler.Event;

public class DownloadEvent extends Event {
    public final String url;
    public final String fileName;
    public final String savePath;

    protected DownloadEvent(String url, String fileName, String savePath) {
        super();
        this.url = url;
        this.fileName = fileName;
        this.savePath = savePath;
    }
}
