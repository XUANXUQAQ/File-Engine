package FileEngine.services.download;

import FileEngine.IsDebug;
import FileEngine.configs.AllConfigs;
import FileEngine.configs.Enums;
import FileEngine.configs.ProxyInfo;

import javax.net.ssl.*;
import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.net.*;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;

public class DownloadManager {
    public final String url;
    public final String savePath;
    public final String fileName;
    private volatile double progress = 0.0;
    private volatile boolean isUserInterrupted = false;
    private volatile Enums.DownloadStatus downloadStatus;
    private Proxy proxy = null;
    private Authenticator authenticator = null;

    public DownloadManager(String url, String fileName, String savePath) {
        this.url = url;
        this.fileName = fileName;
        this.savePath = savePath;
        this.downloadStatus = Enums.DownloadStatus.DOWNLOAD_NO_TASK;
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
                System.out.println("Warning: URL host '" + urlHostName + "' is different to SSLSession host '" + session.getPeerHost() + "'.");
            }
            return true;
        };
        HttpsURLConnection.setDefaultHostnameVerifier(hv);
        System.setProperty("https.protocols", "TLSv1.2,TLSv1.1,SSLv3");
    }

    protected void download() {
        try {
            this.downloadStatus = Enums.DownloadStatus.DOWNLOAD_DOWNLOADING;
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
                    System.err.println(fileName + "下载残余文件删除失败");
                }
                throw new IOException("User Interrupted");
            }
            downloadStatus = Enums.DownloadStatus.DOWNLOAD_DONE;
        } catch (IOException | NoSuchAlgorithmException | KeyManagementException e) {
            e.printStackTrace();
            if (!"User Interrupted".equals(e.getMessage())) {
                downloadStatus = Enums.DownloadStatus.DOWNLOAD_ERROR;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private double div(double v1, double v2) {
        BigDecimal b1 = new BigDecimal(Double.toString(v1));
        BigDecimal b2 = new BigDecimal(Double.toString(v2));
        return b1.divide(b2, 2, RoundingMode.HALF_UP).doubleValue();
    }

    protected void setDownloadDone() {
        this.downloadStatus = Enums.DownloadStatus.DOWNLOAD_DONE;
    }

    protected void setInterrupt() {
        downloadStatus = Enums.DownloadStatus.DOWNLOAD_INTERRUPTED;
        isUserInterrupted = true;
    }

    protected double getDownloadProgress() {
        return progress;
    }

    protected Enums.DownloadStatus getDownloadStatus() {
        if (downloadStatus != Enums.DownloadStatus.DOWNLOAD_DONE) {
            return downloadStatus;
        } else {
            if (!new File(savePath, fileName).exists()) {
                if (IsDebug.isDebug()) {
                    System.err.println("文件不存在，重新下载");
                }
                return Enums.DownloadStatus.DOWNLOAD_NO_TASK;
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
