package FileEngine.download;

import FileEngine.frames.SettingsFrame;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.net.HttpURLConnection;
import java.net.URL;

public class DownloadManager {
    private final String url;
    private final String localPath;
    private final String fileName;
    private volatile double progress = 0.0;
    private volatile boolean isUserInterrupted = false;
    private volatile int downloadStatus;

    public static final int DOWNLOAD_DONE = 0;
    public static final int DOWNLOAD_ERROR = 1;
    public static final int DOWNLOAD_DOWNLOADING = 2;
    public static final int DOWNLOAD_INTERRUPTED = 3;

    public DownloadManager(String _url, String _fileName, String _savePath) {
        this.url = _url;
        this.fileName = _fileName;
        this.localPath = _savePath;
        downloadStatus = DOWNLOAD_DOWNLOADING;
    }

    public String getFileName() {
        return fileName;
    }

    public void download() {
        try {
            System.setProperty("http.keepAlive", "false");
            URL urlAddress = new URL(url);
            HttpURLConnection con = (HttpURLConnection) urlAddress.openConnection();
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
                saveDir.mkdir();
            }
            File file = new File(saveDir + File.separator + fileName);
            FileOutputStream fos = new FileOutputStream(file);

            int fileLength = con.getContentLength();
            while ((len = in.read(buffer)) != -1) {
                if (isUserInterrupted) {
                    break;
                }
                fos.write(buffer, 0, len);
                currentProgress += len;
                progress = div(currentProgress, fileLength, 2);
            }
            fos.close();
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

    private double div(double v1, double v2, int scale) {
        if (scale < 0) {
            throw new IllegalArgumentException(
                    "The scale must be a positive integer or zero");
        }
        BigDecimal b1 = new BigDecimal(Double.toString(v1));
        BigDecimal b2 = new BigDecimal(Double.toString(v2));
        return b1.divide(b2, scale, BigDecimal.ROUND_HALF_UP).doubleValue();
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
}
