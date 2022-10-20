package file.engine.services.download;

import file.engine.annotation.EventRegister;
import file.engine.event.handler.Event;
import file.engine.event.handler.impl.download.StartDownloadEvent;
import file.engine.event.handler.impl.download.StopDownloadEvent;
import file.engine.utils.CachedThreadPoolUtil;

public class DownloadService {
    private static volatile DownloadService INSTANCE = null;

    public static DownloadService getInstance() {
        if (INSTANCE == null) {
            synchronized (DownloadService.class) {
                if (INSTANCE == null) {
                    INSTANCE = new DownloadService();
                }
            }
        }
        return INSTANCE;
    }

    private DownloadService() {
    }

    @EventRegister(registerClass = StartDownloadEvent.class)
    private static void startDownloadEvent(Event event) {
        StartDownloadEvent startDownloadTask = (StartDownloadEvent) event;
        getInstance().downLoadFromUrl(startDownloadTask.downloadManager);
    }

    @EventRegister(registerClass = StopDownloadEvent.class)
    private static void stopDownloadEvent(Event event) {
        StopDownloadEvent stopDownloadTask = (StopDownloadEvent) event;
        getInstance().cancelDownload(stopDownloadTask.downloadManager);
    }

    /**
     * 从网络Url中下载文件
     */
    private void downLoadFromUrl(DownloadManager downloadManager) {
        CachedThreadPoolUtil.getInstance().executeTask(downloadManager::download);
    }

    /**
     * 取消下载任务
     */
    private void cancelDownload(DownloadManager downloadManager) {
        downloadManager.setInterrupt();
    }
}
