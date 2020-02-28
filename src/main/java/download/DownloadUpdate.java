package download;

import frame.SettingsFrame;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;

public class DownloadUpdate {
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
            frame.setVisible(false);
        });
        buttonCancel.setText("ȡ��");
        panel.setLayout(new BorderLayout());
        panel.add(progressBar, BorderLayout.CENTER);
        panel.add(buttonCancel, BorderLayout.SOUTH);
        panel.setOpaque(true);
        frame.add(panel);

        URL frameIcon = SettingsFrame.class.getResource("/icons/frame.png");
        frame.setIconImage(new ImageIcon(frameIcon).getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // ��ȡ��ǰ�ֱ���
        int width = screenSize.width;
        int height = screenSize.height;
        frame.setSize(width / 3, height / 4);
        frame.setLocation(width / 2 - width / 4, height / 2 - height / 4);
    }




    /**
     * ������Url�������ļ�
     *
     * @param urlStr   ��ַ
     * @param savePath ����λ��
     */
    public void downLoadFromUrl(String urlStr, String fileName, String savePath) throws Exception {
        System.setProperty("http.keepAlive", "false"); // must be set
        frame.setVisible(true);
        URL url = new URL(urlStr);
        URLConnection con = url.openConnection();
        //���ó�ʱΪ3��
        con.setConnectTimeout(3 * 1000);
        //��ֹ���γ���ץȡ������403����
        con.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36 Edg/80.0.361.57");
        InputStream in = con.getInputStream();
        byte[] buffer = new byte[1024];
        int progress = 0;
        int len;
        //�ļ�����λ��
        File saveDir = new File(savePath);
        if (!saveDir.exists()) {
            saveDir.mkdir();
        }
        File file = new File(saveDir + File.separator + fileName);
        FileOutputStream fos = new FileOutputStream(file);
        progressBar.setMinimum(0);
        progressBar.setMaximum(con.getContentLength());
        progressBar.setValue(0);
        progressBar.setStringPainted(true);
        progressBar.setBackground(Color.pink);
        while ((len = in.read(buffer)) != -1) {
            fos.write(buffer, 0, len);
            progress += len;
            progressBar.setValue(progress);
            progressBar.setString("�����أ�" + (int) (progressBar.getPercentComplete() * 100) + "%");
            if (!frame.isVisible()) {
                fos.close();
                throw new Exception("�û��ж�����");
            }
        }
        fos.close();
        frame.setVisible(false);
    }

    public void hideFrame() {
        frame.setVisible(false);
    }
}
