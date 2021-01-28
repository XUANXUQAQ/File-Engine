package FileEngine.eventHandler.impl.download;

import FileEngine.services.download.DownloadManager;

public class StopDownloadEvent extends DownloadEvent {

    public StopDownloadEvent(DownloadManager downloadManager) {
        super(downloadManager);
    }
}
