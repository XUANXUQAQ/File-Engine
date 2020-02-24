package download;

import frame.SettingsFrame;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.net.URL;
import java.net.URLConnection;

public class DownloadUpdate {
    private boolean isUserInterruptDownload = false;
    private JFrame frame = new JFrame();
    private JProgressBar progressBar = new JProgressBar();
    private static DownloadUpdate downloadupdate = new DownloadUpdate();

    public static DownloadUpdate getInstance() {
        return downloadupdate;
    }

    private DownloadUpdate() {
        JPanel panel = new JPanel();
        JButton buttonCancel = new JButton();
        buttonCancel.addActionListener(e -> {
            isUserInterruptDownload = true;
            frame.setVisible(false);
        });
        buttonCancel.setText("取消");
        panel.setLayout(new BorderLayout());
        panel.add(progressBar, BorderLayout.CENTER);
        panel.add(buttonCancel, BorderLayout.SOUTH);
        panel.setOpaque(true);
        frame.add(panel);

        URL frameIcon = SettingsFrame.class.getResource("/icons/frame.png");
        frame.setIconImage(new ImageIcon(frameIcon).getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取当前分辨率
        int width = screenSize.width;
        int height = screenSize.height;
        frame.setSize(width / 3, height / 4);
        frame.setLocation(width / 2 - width / 4, height / 2 - height / 4);
    }

    private byte[] readInputStream(InputStream inputStream, int maxLength) throws IOException {
        byte[] buffer = new byte[1024];
        int progress = 0;
        int len;
        progressBar.setMinimum(0);
        progressBar.setMaximum(maxLength);
        progressBar.setValue(0);
        progressBar.setStringPainted(true);
        progressBar.setBackground(Color.pink);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        while ((len = inputStream.read(buffer)) != -1) {
            bos.write(buffer, 0, len);
            progress += len;
            progressBar.setValue(progress);
            progressBar.setString("已下载：" + (int) (progressBar.getPercentComplete() * 100) + "%");
            if (isUserInterruptDownload) {
                bos.close();
                return null;
            }
        }
        bos.close();
        return bos.toByteArray();
    }



    /**
     * 从网络Url中下载文件
     *
     * @param urlStr   地址
     * @param savePath 保存位置
     */
    public void downLoadFromUrl(String urlStr, String fileName, String savePath) throws Exception {
        System.setProperty("http.keepAlive", "false"); // must be set
        frame.setVisible(true);
        isUserInterruptDownload = false;
        URL url = new URL(urlStr);
        URLConnection con = url.openConnection();
        //设置超时间为3秒
        con.setConnectTimeout(3 * 1000);
        //防止屏蔽程序抓取而返回403错误
        con.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36 Edg/80.0.361.57");
        InputStream in = con.getInputStream();
        byte[] getData = readInputStream(in, con.getContentLength());
        if (getData == null) {
            throw new Exception("用户中断下载");
        }
        //文件保存位置
        File saveDir = new File(savePath);
        if (!saveDir.exists()) {
            saveDir.mkdir();
        }
        File file = new File(saveDir + File.separator + fileName);
        FileOutputStream fos = new FileOutputStream(file);
        fos.write(getData);
        fos.close();
        frame.setVisible(false);
    }

    public void hideFrame() {
        frame.setVisible(false);
    }
}
