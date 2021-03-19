package file.engine.event.handler.impl.download;

import file.engine.services.download.DownloadManager;

public class StopDownloadEvent extends DownloadEvent {

    public StopDownloadEvent(DownloadManager downloadManager) {
        super(downloadManager);
    }
}
