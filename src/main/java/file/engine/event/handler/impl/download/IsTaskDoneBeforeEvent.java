package file.engine.event.handler.impl.download;

import file.engine.services.download.DownloadManager;

public class IsTaskDoneBeforeEvent extends DownloadEvent{
    public IsTaskDoneBeforeEvent(DownloadManager downloadManager) {
        super(downloadManager);
    }
}
