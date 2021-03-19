package file.engine.event.handler.impl.download;

import file.engine.services.download.DownloadManager;

public class StartDownloadEvent extends DownloadEvent {

    public StartDownloadEvent(DownloadManager downloadManager) {
        super(downloadManager);
    }
}
