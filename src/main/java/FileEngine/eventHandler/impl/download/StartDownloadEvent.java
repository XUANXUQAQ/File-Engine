package FileEngine.eventHandler.impl.download;

import FileEngine.services.download.DownloadManager;

public class StartDownloadEvent extends DownloadEvent {

    public StartDownloadEvent(DownloadManager downloadManager) {
        super(downloadManager);
    }
}
