package FileEngine.download;

import FileEngine.enums.Enums;
import FileEngine.frames.SettingsFrame;
import FileEngine.threadPool.CachedThreadPool;

import java.io.*;
import java.math.BigDecimal;
import java.net.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

public class DownloadUtil {
    private final ConcurrentHashMap<String, DownloadManager> DOWNLOAD_MAP = new ConcurrentHashMap<>();

    private static class DownloadUpdateBuilder {
        private static final DownloadUtil INSTANCE = new DownloadUtil();
    }

    public static DownloadUtil getInstance() {
        return DownloadUpdateBuilder.INSTANCE;
    }

    private DownloadUtil() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (SettingsFrame.isNotMainExit()) {
                    for (DownloadManager each : DOWNLOAD_MAP.values()) {
                        int status = each.getDownloadStatus();
                        if (status == Enums.DownloadStatus.DOWNLOAD_INTERRUPTED || status == Enums.DownloadStatus.DOWNLOAD_ERROR) {
                            deleteTask(each.getFileName());
                        }
                    }
                    TimeUnit.SECONDS.sleep(5);
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
        DownloadManager downloadManager = new DownloadManager(urlStr, fileName, savePath, SettingsFrame.getProxy());
        CachedThreadPool.getInstance().executeTask(downloadManager::download);
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

    private boolean hasTask(String fileName) {
        if (isFileNameNotContainsSuffix(fileName)) {
            System.err.println("Warning:" + fileName + " doesn't have suffix");
        }
        return DOWNLOAD_MAP.containsKey(fileName);
    }

    public int getDownloadStatus(String fileName) {
        if (hasTask(fileName)) {
            return DOWNLOAD_MAP.get(fileName).getDownloadStatus();
        }
        return Enums.DownloadStatus.DOWNLOAD_NO_TASK;
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

    private static class DownloadManager {
        private final String url;
        private final String localPath;
        private final String fileName;
        private volatile double progress = 0.0;
        private volatile boolean isUserInterrupted = false;
        private volatile int downloadStatus;
        private Proxy proxy = null;
        private Authenticator authenticator = null;

        public DownloadManager(String url, String fileName, String savePath, SettingsFrame.ProxyInfo proxyInfo) {
            this.url = url;
            this.fileName = fileName;
            this.localPath = savePath;
            this.downloadStatus = Enums.DownloadStatus.DOWNLOAD_DOWNLOADING;
            setProxy(proxyInfo.type, proxyInfo.address, proxyInfo.port, proxyInfo.userName, proxyInfo.password);
        }

        public String getFileName() {
            return fileName;
        }

        public void download() {
            try {
                System.setProperty("http.keepAlive", "false");
                URL urlAddress = new URL(url);
                HttpURLConnection con;
                if (proxy.equals(Proxy.NO_PROXY)) {
                    con = (HttpURLConnection) urlAddress.openConnection();
                    Authenticator.setDefault(null);
                } else {
                    con = (HttpURLConnection) urlAddress.openConnection(proxy);
                    Authenticator.setDefault(authenticator);
                }
                //设置超时为3秒
                con.setConnectTimeout(3000);
                //防止屏蔽程序抓取而返回403错误
                con.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36 Edg/80.0.361.57");
                InputStream in = con.getInputStream();
                byte[] buffer = new byte[1];
                int currentProgress = 0;
                int len;
                //文件保存位置
                File saveDir = new File(localPath);
                if (!saveDir.exists()) {
                    saveDir.mkdirs();
                }
                BufferedOutputStream bfos = new BufferedOutputStream(new FileOutputStream(new File(saveDir + File.separator + fileName)));

                int fileLength = con.getContentLength();
                while ((len = in.read(buffer)) != -1) {
                    if (isUserInterrupted) {
                        break;
                    }
                    bfos.write(buffer, 0, len);
                    currentProgress += len;
                    progress = div(currentProgress, fileLength);
                }
                bfos.close();
                in.close();
                con.disconnect();
                if (isUserInterrupted) {
                    throw new IOException("User Interrupted");
                }
                downloadStatus = Enums.DownloadStatus.DOWNLOAD_DONE;
            } catch (IOException e) {
                if (SettingsFrame.isDebug()) {
                    e.printStackTrace();
                }
                if ("User Interrupted".equals(e.getMessage())) {
                    downloadStatus = Enums.DownloadStatus.DOWNLOAD_INTERRUPTED;
                } else {
                    downloadStatus = Enums.DownloadStatus.DOWNLOAD_ERROR;
                }
            }
        }

        private double div(double v1, double v2) {
            BigDecimal b1 = new BigDecimal(Double.toString(v1));
            BigDecimal b2 = new BigDecimal(Double.toString(v2));
            return b1.divide(b2, 2, BigDecimal.ROUND_HALF_UP).doubleValue();
        }

        public void setInterrupt() {
            isUserInterrupted = true;
        }

        public double getDownloadProgress() {
            return progress;
        }

        public int getDownloadStatus() {
            return downloadStatus;
        }

        private void setProxy(Proxy.Type proxyType, String address, int port, String userName, String password) {
            SocketAddress sa = new InetSocketAddress(address, port);
            authenticator = new BasicAuthenticator(userName, password);
            if (proxyType == Proxy.Type.DIRECT) {
                proxy = Proxy.NO_PROXY;
            } else {
                proxy = new Proxy(proxyType, sa);
            }
        }

    }
}
