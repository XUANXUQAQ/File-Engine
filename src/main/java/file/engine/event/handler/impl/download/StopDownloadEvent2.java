package file.engine.event.handler.impl.download;

import file.engine.services.download.DownloadManager;

public class StopDownloadEvent2 extends DownloadEvent {

    public StopDownloadEvent2(String url, String fileName, String savePath) {
        super(new DownloadManager(url, fileName, savePath));
    }
}
