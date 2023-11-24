package file.engine.event.handler.impl.download;

import file.engine.services.download.DownloadManager;

public class DownloadDoneEvent extends DownloadEvent {
    public DownloadDoneEvent(DownloadManager downloadManager) {
        super(downloadManager);
    }
}
