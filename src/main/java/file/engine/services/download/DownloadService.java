package file.engine.services.download;

import file.engine.annotation.EventRegister;
import file.engine.configs.Constants;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.download.*;
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

    @EventRegister(registerClass = IsTaskDoneBeforeEvent.class)
    private static void IsTaskDoneBeforeEvent(Event event) {
        IsTaskDoneBeforeEvent isTaskDoneBeforeEvent = (IsTaskDoneBeforeEvent) event;
        isTaskDoneBeforeEvent.setReturnValue(getInstance().isTaskDoneBefore(isTaskDoneBeforeEvent.downloadManager));
    }

    /**
     * 从网络Url中下载文件
     */
    private void downLoadFromUrl(DownloadManager downloadManager) {
        if (getFromSet(downloadManager).getDownloadStatus() == Constants.Enums.DownloadStatus.DOWNLOAD_DONE) {
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

    public boolean waitForDownloadTask(DownloadManager downloadManager, int maxWaitingMills) throws IOException {
        try {
            long startTime = System.currentTimeMillis();
            final int sleepMills = 10;
            EventManagement instance = EventManagement.getInstance();
            while (instance.notMainExit()) {
                if (System.currentTimeMillis() - startTime > maxWaitingMills) {
                    throw new IOException("download failed");
                }
                Constants.Enums.DownloadStatus downloadStatus = downloadManager.getDownloadStatus();
                if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_DONE) {
                    return true;
                } else if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_ERROR) {
                    throw new IOException("download failed");
                } else if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_INTERRUPTED) {
                    return false;
                }
                TimeUnit.MILLISECONDS.sleep(sleepMills);
            }
        } catch (InterruptedException ignored) {
        }
        return false;
    }

    private boolean isTaskDoneBefore(DownloadManager downloadManager) {
        return getFromSet(downloadManager).getDownloadStatus() == Constants.Enums.DownloadStatus.DOWNLOAD_DONE;
    }

    /**
     * 取消下载任务
     */
    private void cancelDownload(DownloadManager downloadManager) {
        downloadManager.setInterrupt();
    }
}
