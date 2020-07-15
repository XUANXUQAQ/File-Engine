package FileEngine.Download;

import FileEngine.Frames.SettingsFrame;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DownloadUtil {
    private static final ExecutorService downloadThreadPool = Executors.newCachedThreadPool();
    private static final ConcurrentHashMap<String, DownloadManager> DOWNLOAD_MAP = new ConcurrentHashMap<>();

    private static class DownloadUpdateBuilder {
        private static final DownloadUtil INSTANCE = new DownloadUtil();
    }

    public static DownloadUtil getInstance() {
        return DownloadUpdateBuilder.INSTANCE;
    }

    private DownloadUtil() {
        downloadThreadPool.execute(() -> {
            try {
                while (SettingsFrame.isNotMainExit()) {
                    for (DownloadManager each : DOWNLOAD_MAP.values()) {
                        int status = each.getDownloadStatus();
                        if (status == DownloadManager.DOWNLOAD_INTERRUPTED || status == DownloadManager.DOWNLOAD_ERROR) {
                            deleteTask(each.getFileName());
                        }
                    }
                    Thread.sleep(5000);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    /**
     * 从网络Url中下载文件
     *
     * @param urlStr   地址
     * @param savePath 保存位置
     */
    public void downLoadFromUrl(String urlStr, String fileName, String savePath) {
        DownloadManager downloadManager = new DownloadManager(urlStr, fileName, savePath);
        downloadThreadPool.execute(downloadManager::download);
        DOWNLOAD_MAP.put(fileName, downloadManager);
    }

    public double getDownloadProgress(String fileName) {
        if (isFileNameNotContainsSuffix(fileName)) {
            System.err.println("Warning:" + fileName + " doesn't have suffix");
        }
        if (hasTask(fileName)) {
            return DOWNLOAD_MAP.get(fileName).getDownloadProgress();
        }
        return 0.0;
    }

    public void cancelDownload(String fileName) {
        if (isFileNameNotContainsSuffix(fileName)) {
            System.err.println("Warning:" + fileName + " doesn't have suffix");
        }
        if (SettingsFrame.isDebug()) {
            System.out.println("cancel downloading " + fileName);
        }
        if (hasTask(fileName)) {
            DOWNLOAD_MAP.get(fileName).setInterrupt();
        }
    }

    public boolean hasTask(String fileName) {
        if (isFileNameNotContainsSuffix(fileName)) {
            System.err.println("Warning:" + fileName + " doesn't have suffix");
        }
        return DOWNLOAD_MAP.containsKey(fileName);
    }

    public int getDownloadStatus(String fileName) {
        if (isFileNameNotContainsSuffix(fileName)) {
            System.err.println("Warning:" + fileName + " doesn't have suffix");
        }
        if (hasTask(fileName)) {
            return DOWNLOAD_MAP.get(fileName).getDownloadStatus();
        }
        return DownloadManager.DOWNLOAD_ERROR;
    }

    private void deleteTask(String fileName) {
        DOWNLOAD_MAP.remove(fileName);
    }

    private boolean isFileNameNotContainsSuffix(String fileName) {
        if (fileName == null) {
            return false;
        }
        if (SettingsFrame.isDebug()) {
            return fileName.lastIndexOf(".") == -1;
        } else {
            return false;
        }
    }
}
