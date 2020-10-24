package FileEngine.download;

import FileEngine.configs.AllConfigs;
import FileEngine.threadPool.CachedThreadPool;

import javax.net.ssl.*;
import java.io.*;
import java.math.BigDecimal;
import java.net.*;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.Security;
import java.security.cert.X509Certificate;
import java.util.concurrent.ConcurrentHashMap;

public class DownloadUtil {
    private final ConcurrentHashMap<String, DownloadManager> DOWNLOAD_MAP = new ConcurrentHashMap<>();

    private static class DownloadUpdateBuilder {
        private static final DownloadUtil INSTANCE = new DownloadUtil();
    }

    public static DownloadUtil getInstance() {
        return DownloadUpdateBuilder.INSTANCE;
    }

    private DownloadUtil() {
    }

    /**
     * 从网络Url中下载文件
     *
     * @param urlStr   地址
     * @param savePath 保存位置
     */
    public void downLoadFromUrl(String urlStr, String fileName, String savePath) {
        DownloadManager downloadManager = new DownloadManager(urlStr, fileName, savePath, AllConfigs.getProxy());
        CachedThreadPool.getInstance().executeTask(downloadManager::download);
        DOWNLOAD_MAP.put(fileName, downloadManager);
    }

    /**
     * 根据下载文件名获取当前下载进度
     *
     * @param fileName 下载文件名
     * @return 进度
     */
    public double getDownloadProgress(String fileName) {
        if (isFileNameNotContainsSuffix(fileName)) {
            System.err.println("Warning:" + fileName + " doesn't have suffix");
        }
        if (hasTask(fileName)) {
            return DOWNLOAD_MAP.get(fileName).getDownloadProgress();
        }
        return 0.0;
    }

    /**
     * 取消下载任务
     *
     * @param fileName 下载文件名
     */
    public void cancelDownload(String fileName) {
        if (isFileNameNotContainsSuffix(fileName)) {
            System.err.println("Warning:" + fileName + " doesn't have suffix");
        }
        if (AllConfigs.isDebug()) {
            System.out.println("cancel downloading " + fileName);
        }
        if (hasTask(fileName)) {
            DOWNLOAD_MAP.get(fileName).setInterrupt();
        }
    }

    /**
     * 判断是否有当前下载任务
     *
     * @param fileName 下载文件名
     * @return 任务是否存在
     */
    private boolean hasTask(String fileName) {
        if (isFileNameNotContainsSuffix(fileName)) {
            System.err.println("Warning:" + fileName + " doesn't have suffix");
        }
        return DOWNLOAD_MAP.containsKey(fileName);
    }

    /**
     * 获取当前任务的下载状态， 已完成 无任务 下载错误 已取消
     *
     * @param fileName 下载文件名
     * @return 当前任务状态
     */
    public AllConfigs.DownloadStatus getDownloadStatus(String fileName) {
        if (hasTask(fileName)) {
            return DOWNLOAD_MAP.get(fileName).getDownloadStatus();
        }
        return AllConfigs.DownloadStatus.DOWNLOAD_NO_TASK;
    }

    /**
     * 判断当前下载文件是否拥有文件后缀名，debug使用
     *
     * @param fileName 下载文件名
     * @return true和false
     */
    private boolean isFileNameNotContainsSuffix(String fileName) {
        if (fileName == null) {
            return false;
        }
        if (AllConfigs.isDebug()) {
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
        private volatile AllConfigs.DownloadStatus downloadStatus;
        private Proxy proxy = null;
        private Authenticator authenticator = null;

        private DownloadManager(String url, String fileName, String savePath, AllConfigs.ProxyInfo proxyInfo) {
            this.url = url;
            this.fileName = fileName;
            this.localPath = savePath;
            this.downloadStatus = AllConfigs.DownloadStatus.DOWNLOAD_DOWNLOADING;
            setProxy(proxyInfo.type, proxyInfo.address, proxyInfo.port, proxyInfo.userName, proxyInfo.password);
        }

        // trusting all certificate
        private void doTrustToCertificates() throws Exception {
            Security.addProvider(new com.sun.net.ssl.internal.ssl.Provider());
            TrustManager[] trustAllCerts = new TrustManager[]{
                    new X509TrustManager() {
                        public X509Certificate[] getAcceptedIssuers() {
                            return null;
                        }

                        public void checkServerTrusted(X509Certificate[] certs, String authType) {
                        }

                        public void checkClientTrusted(X509Certificate[] certs, String authType) {
                        }
                    }
            };

            SSLContext sc = SSLContext.getInstance("SSL");
            sc.init(null, trustAllCerts, new SecureRandom());
            HttpsURLConnection.setDefaultSSLSocketFactory(sc.getSocketFactory());
            HostnameVerifier hv = (urlHostName, session) -> {
                if (!urlHostName.equalsIgnoreCase(session.getPeerHost())) {
                    System.out.println("Warning: URL host '" + urlHostName + "' is different to SSLSession host '" + session.getPeerHost() + "'.");
                }
                return true;
            };
            HttpsURLConnection.setDefaultHostnameVerifier(hv);
            System.setProperty("https.protocols", "TLSv1.2,TLSv1.1,SSLv3");
        }

        private void download() {
            try {
                System.setProperty("http.keepAlive", "false");
                URL urlAddress = new URL(url);
                HttpURLConnection con;
                doTrustToCertificates();
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
                con.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44");
                InputStream in = con.getInputStream();
                byte[] buffer = new byte[1];
                int currentProgress = 0;
                int len;
                //文件保存位置
                File saveDir = new File(localPath);

                if (!saveDir.exists()) {
                    saveDir.mkdirs();
                }
                if (!saveDir.exists()) {
                    throw new IOException("Create dirs failed");
                }
                File fileFullPath = new File(saveDir + File.separator + fileName);
                BufferedOutputStream bfos = new BufferedOutputStream(new FileOutputStream(fileFullPath));

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
                    //删除文件
                    fileFullPath.delete();
                    throw new IOException("User Interrupted");
                }
                downloadStatus = AllConfigs.DownloadStatus.DOWNLOAD_DONE;
            } catch (IOException | NoSuchAlgorithmException | KeyManagementException e) {
                e.printStackTrace();
                if (!"User Interrupted".equals(e.getMessage())) {
                    downloadStatus = AllConfigs.DownloadStatus.DOWNLOAD_ERROR;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private double div(double v1, double v2) {
            BigDecimal b1 = new BigDecimal(Double.toString(v1));
            BigDecimal b2 = new BigDecimal(Double.toString(v2));
            return b1.divide(b2, 2, BigDecimal.ROUND_HALF_UP).doubleValue();
        }

        private void setInterrupt() {
            isUserInterrupted = true;
            downloadStatus = AllConfigs.DownloadStatus.DOWNLOAD_INTERRUPTED;
        }

        private double getDownloadProgress() {
            return progress;
        }

        private AllConfigs.DownloadStatus getDownloadStatus() {
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
