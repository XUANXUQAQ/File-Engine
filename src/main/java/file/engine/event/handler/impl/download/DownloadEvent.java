package file.engine.event.handler.impl.download;

import file.engine.event.handler.Event;
import file.engine.services.download.DownloadManager;

public class DownloadEvent extends Event {
    public final DownloadManager downloadManager;

    protected DownloadEvent(DownloadManager downloadManager) {
        super();
        this.downloadManager = downloadManager;
    }
}
