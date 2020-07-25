package FileEngine.download;

import FileEngine.frames.SettingsFrame;

import java.io.*;
import java.math.BigDecimal;
import java.net.*;

public class DownloadManager {
    private final String url;
    private final String localPath;
    private final String fileName;
    private volatile double progress = 0.0;
    private volatile boolean isUserInterrupted = false;
    private volatile int downloadStatus;
    private Proxy proxy = null;
    private Authenticator authenticator = null;

    public static final int DOWNLOAD_DONE = 0;
    public static final int DOWNLOAD_ERROR = 1;
    public static final int DOWNLOAD_DOWNLOADING = 2;
    public static final int DOWNLOAD_INTERRUPTED = 3;
    public static final int DOWNLOAD_NO_TASK = 4;

    public DownloadManager(String url, String fileName, String savePath, SettingsFrame.ProxyInfo proxyInfo) {
        this.url = url;
        this.fileName = fileName;
        this.localPath = savePath;
        this.downloadStatus = DOWNLOAD_DOWNLOADING;
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
            downloadStatus = DOWNLOAD_DONE;
        } catch (IOException e) {
            if (SettingsFrame.isDebug()) {
                e.printStackTrace();
            }
            if ("User Interrupted".equals(e.getMessage())) {
                downloadStatus = DOWNLOAD_INTERRUPTED;
            } else {
                downloadStatus = DOWNLOAD_ERROR;
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
