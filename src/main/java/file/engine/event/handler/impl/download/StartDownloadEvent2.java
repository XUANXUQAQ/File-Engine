package file.engine.event.handler.impl.download;

import file.engine.services.download.DownloadManager;

public class StartDownloadEvent2 extends DownloadEvent {

    public StartDownloadEvent2(String url, String fileName, String savePath) {
        super(new DownloadManager(url, fileName, savePath));
    }
}
