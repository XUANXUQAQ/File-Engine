package file.engine.services.download;

import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.configs.ProxyInfo;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.download.DownloadDoneEvent;
import lombok.extern.slf4j.Slf4j;

import javax.net.ssl.*;
import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.*;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;
import java.util.concurrent.TimeUnit;

@Slf4j
public class DownloadManager {
    public final String url;
    public final String savePath;
    public final String fileName;
    private volatile double progress = 0.0;
    private volatile boolean isUserInterrupted = false;
    private volatile Constants.Enums.DownloadStatus downloadStatus;
    private Proxy proxy = null;
    private Authenticator authenticator = null;

    public DownloadManager(String url, String fileName, String savePath) {
        this.url = url;
        this.fileName = fileName;
        this.savePath = savePath;
        this.downloadStatus = Constants.Enums.DownloadStatus.DOWNLOAD_NO_TASK;
        ProxyInfo proxyInfo = AllConfigs.getInstance().getProxy();
        setProxy(proxyInfo.type, proxyInfo.address, proxyInfo.port, proxyInfo.userName, proxyInfo.password);
    }

    // trusting all certificate
    private void doTrustToCertificates() throws Exception {
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
                log.info("Warning: URL host '" + urlHostName + "' is different to SSLSession host '" + session.getPeerHost() + "'.");
            }
            return true;
        };
        HttpsURLConnection.setDefaultHostnameVerifier(hv);
        System.setProperty("https.protocols", "TLSv1.2,TLSv1.1,SSLv3");
    }

    public boolean waitFor(int maxWaitingMills) throws IOException {
        long startTime = System.currentTimeMillis();
        final int sleepMills = 10;
        EventManagement instance = EventManagement.getInstance();
        while (instance.notMainExit()) {
            if (System.currentTimeMillis() - startTime > maxWaitingMills) {
                throw new IOException("download failed");
            }
            Constants.Enums.DownloadStatus downloadStatus = this.getDownloadStatus();
            if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_DONE) {
                return true;
            } else if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_ERROR) {
                throw new IOException("download failed");
            } else if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_INTERRUPTED) {
                return false;
            }
            try {
                TimeUnit.MILLISECONDS.sleep(sleepMills);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        return false;
    }

    protected void download() {
        try {
            this.downloadStatus = Constants.Enums.DownloadStatus.DOWNLOAD_DOWNLOADING;
            System.setProperty("http.keepAlive", "false");
            URI uriAddress = new URI(url);
            var urlAddress = uriAddress.toURL();
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
            con.setRequestProperty("User-Agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.47");
            byte[] buffer = new byte[1];
            int currentProgress = 0;
            int len;
            //文件保存位置
            File saveDir = new File(savePath);
            if (!saveDir.exists()) {
                if (!saveDir.mkdirs()) {
                    throw new IOException("Create dirs failed");
                }
            }
            File fileFullPath = new File(saveDir, fileName);
            try (InputStream in = con.getInputStream();
                 BufferedOutputStream bfos = new BufferedOutputStream(new FileOutputStream(fileFullPath))) {
                int fileLength = con.getContentLength();
                while ((len = in.read(buffer)) != -1) {
                    if (isUserInterrupted) {
                        break;
                    }
                    bfos.write(buffer, 0, len);
                    currentProgress += len;
                    progress = div(currentProgress, fileLength);
                }
            }
            con.disconnect();
            if (isUserInterrupted) {
                //删除文件
                if (!fileFullPath.delete()) {
                    throw new RuntimeException(fileName + "下载残余文件删除失败");
                }
                throw new IOException("User Interrupted");
            }
            downloadStatus = Constants.Enums.DownloadStatus.DOWNLOAD_DONE;
            EventManagement.getInstance().putEvent(new DownloadDoneEvent(this));
        } catch (IOException | NoSuchAlgorithmException | KeyManagementException e) {
            log.error("error: {}", e.getMessage(), e);
            if (!"User Interrupted".equals(e.getMessage())) {
                downloadStatus = Constants.Enums.DownloadStatus.DOWNLOAD_ERROR;
            }
        } catch (Exception e) {
            log.error("error: {}", e.getMessage(), e);
        }
    }

    private double div(double v1, double v2) {
        BigDecimal b1 = new BigDecimal(Double.toString(v1));
        BigDecimal b2 = new BigDecimal(Double.toString(v2));
        return b1.divide(b2, 2, RoundingMode.HALF_UP).doubleValue();
    }

    protected void setInterrupt() {
        downloadStatus = Constants.Enums.DownloadStatus.DOWNLOAD_INTERRUPTED;
        isUserInterrupted = true;
    }

    public double getDownloadProgress() {
        return progress;
    }

    public Constants.Enums.DownloadStatus getDownloadStatus() {
        if (downloadStatus != Constants.Enums.DownloadStatus.DOWNLOAD_DONE) {
            return downloadStatus;
        } else {
            if (!new File(savePath, fileName).exists()) {
                return Constants.Enums.DownloadStatus.DOWNLOAD_NO_TASK;
            } else {
                return downloadStatus;
            }
        }
    }

    protected void setProxy(Proxy.Type proxyType, String address, int port, String userName, String password) {
        SocketAddress sa = new InetSocketAddress(address, port);
        authenticator = new BasicAuthenticator(userName, password);
        if (proxyType == Proxy.Type.DIRECT) {
            proxy = Proxy.NO_PROXY;
        } else {
            proxy = new Proxy(proxyType, sa);
        }
    }
}
