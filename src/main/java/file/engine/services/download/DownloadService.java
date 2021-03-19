package file.engine.services.download;

import file.engine.annotation.EventRegister;
import file.engine.configs.Enums;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.download.StartDownloadEvent;
import file.engine.event.handler.impl.download.StopDownloadEvent;
import file.engine.utils.CachedThreadPoolUtil;

import java.io.IOException;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

public class DownloadService {
    private final Set<DownloadManager> downloadManagerSet = ConcurrentHashMap.newKeySet();
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

    private DownloadService() {}

    @EventRegister
    @SuppressWarnings("unused")
    public static void registerEventHandler() {
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.register(StartDownloadEvent.class, event -> {
            StartDownloadEvent startDownloadTask = (StartDownloadEvent) event;
            getInstance().downLoadFromUrl(startDownloadTask.downloadManager);
        });

        eventManagement.register(StopDownloadEvent.class, event -> {
            StopDownloadEvent stopDownloadTask = (StopDownloadEvent) event;
            getInstance().cancelDownload(stopDownloadTask.downloadManager);
        });
    }

    /**
     * 从网络Url中下载文件
     */
    private void downLoadFromUrl(DownloadManager downloadManager) {
        if (getFromSet(downloadManager).getDownloadStatus() == Enums.DownloadStatus.DOWNLOAD_DONE) {
            downloadManager.setDownloadDone();
            return;
        }
        CachedThreadPoolUtil.getInstance().executeTask(downloadManager::download);
        downloadManagerSet.add(downloadManager);
    }

    private DownloadManager getFromSet(DownloadManager downloadManager) {
        for (DownloadManager each : downloadManagerSet) {
            if (
                    each.fileName.equals(downloadManager.fileName) &&
                    each.savePath.equals(downloadManager.savePath)
            ) {
                return each;
            }
        }
        return downloadManager;
    }

    /**
     * 根据下载文件名获取当前下载进度
     */
    public double getDownloadProgress(DownloadManager downloadManager) {
        return downloadManager.getDownloadProgress();
    }

    public void waitForDownloadTask(DownloadManager downloadManager, int maxWaitingMills) throws IOException {
        try {
            long startTime = System.currentTimeMillis();
            final int sleepMills = 10;
            while (true) {
                if (System.currentTimeMillis() - startTime > maxWaitingMills) {
                    throw new IOException("download failed");
                }
                Enums.DownloadStatus downloadStatus = downloadManager.getDownloadStatus();
                if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_DONE) {
                    return;
                } else if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_ERROR) {
                    throw new IOException("download failed");
                } else if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_INTERRUPTED) {
                    return;
                }
                TimeUnit.MILLISECONDS.sleep(sleepMills);
            }
        } catch (InterruptedException ignored) {
        }
    }

    public boolean isTaskDone(DownloadManager downloadManager) {
        return getFromSet(downloadManager).getDownloadStatus() == Enums.DownloadStatus.DOWNLOAD_DONE;
    }

    /**
     * 取消下载任务
     */
    private void cancelDownload(DownloadManager downloadManager) {
        downloadManager.setInterrupt();
    }

    /**
     * 获取当前任务的下载状态， 已完成 无任务 下载错误 已取消
     */
    public Enums.DownloadStatus getDownloadStatus(DownloadManager downloadManager) {
        return downloadManager.getDownloadStatus();
    }
}
