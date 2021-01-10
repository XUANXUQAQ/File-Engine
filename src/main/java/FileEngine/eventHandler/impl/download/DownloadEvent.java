package FileEngine.eventHandler.impl.download;

import FileEngine.eventHandler.Event;
import FileEngine.utils.download.DownloadManager;

public class DownloadEvent extends Event {
    public final DownloadManager downloadManager;

    protected DownloadEvent(DownloadManager downloadManager) {
        super();
        this.downloadManager = downloadManager;
    }
}
