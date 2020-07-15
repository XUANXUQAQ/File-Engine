package FileEngine.Frames;

import FileEngine.Download.DownloadManager;
import FileEngine.Download.DownloadUtil;
import FileEngine.PluginSystem.PluginUtil;
import com.alibaba.fastjson.JSONObject;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static FileEngine.Frames.SettingsFrame.getTranslation;

public class PluginMarket {
    private static class PluginMarketBuilder {
        private static final PluginMarket INSTANCE = new PluginMarket();
    }

    private JList<Object> listPlugins;
    private JTextArea textAreaPluginDescription;
    private JScrollPane scrollPluginDescription;
    private JTextField textFieldSearchPlugin;
    private JLabel labelIcon;
    private JLabel labelPluginName;
    private JButton buttonInstall;
    private JLabel labelOfficialSite;
    private JLabel labelAuthor;
    private JLabel labelVersion;
    private JPanel panel;
    private JLabel labelProgress;
    private final JFrame frame = new JFrame("Plugin Market");
    private volatile String searchKeywords;
    private final HashMap<String, String> NAME_PLUGIN_INFO_URL_MAP = new HashMap<>();
    private volatile boolean isStartSearch = false;
    private final ExecutorService threadPool = Executors.newCachedThreadPool();

    private PluginMarket() {
        addSelectPluginOnListListener();
        addSearchPluginListener();
        addButtonInstallListener();
        addOpenPluginOfficialSiteListener();
        threadPool.execute(() -> {
            try {
                String fileName;
                String originString = buttonInstall.getText();
                while (SettingsFrame.isNotMainExit()) {
                    fileName = (String) listPlugins.getSelectedValue();
                    checkDownloadTask(labelProgress, buttonInstall, fileName + ".jar", originString);
                }
            } catch (InterruptedException ignored) {
            }
        });

        threadPool.execute(() -> {
            HashSet<String> pluginSet = new HashSet<>();
            try {
                while (SettingsFrame.isNotMainExit()) {
                    if (isStartSearch) {
                        isStartSearch = false;
                        pluginSet.clear();
                        for (String each : NAME_PLUGIN_INFO_URL_MAP.keySet()) {
                            if (each.contains(searchKeywords)) {
                                pluginSet.add(each);
                            }
                        }
                        listPlugins.setListData(pluginSet.toArray());
                    }
                    Thread.sleep(500);
                }
            } catch (InterruptedException ignored) {

            }
        });
    }

    private void checkDownloadTask(JLabel label, JButton button, String fileName, String originButtonString) throws InterruptedException {
        //设置进度显示线程
        double progress;
        String pluginName;
        if (DownloadUtil.getInstance().hasTask(fileName)) {
            progress = DownloadUtil.getInstance().getDownloadProgress(fileName);
            label.setText(getTranslation("Downloading:") + progress * 100 + "%");

            int downloadingStatus = DownloadUtil.getInstance().getDownloadStatus(fileName);
            if (downloadingStatus == DownloadManager.DOWNLOAD_DONE) {
                //下载完成，禁用按钮
                label.setText(getTranslation("Download Done"));
                label.setText(getTranslation("Downloaded"));
                label.setEnabled(false);
                File updatePluginSign = new File("user/update");
                if (!updatePluginSign.exists()) {
                    try {
                        updatePluginSign.createNewFile();
                    } catch (IOException ignored) {
                    }
                }
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_ERROR) {
                //下载错误，重置button
                label.setText(getTranslation("Download failed"));
                button.setText(getTranslation(originButtonString));
                button.setEnabled(true);
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_DOWNLOADING) {
                //正在下载
                button.setText(getTranslation("Cancel"));
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_INTERRUPTED) {
                //用户自行中断
                label.setText("");
                button.setText(getTranslation(originButtonString));
                button.setEnabled(true);
            }
        } else {
            int index = fileName.indexOf(".");
            pluginName = fileName.substring(0, index);
            if (PluginUtil.getIdentifierByName(pluginName) != null) {
                label.setText("");
                button.setText(getTranslation("Installed"));
                button.setEnabled(false);
            } else {
                label.setText("");
                button.setText(getTranslation(originButtonString));
                button.setEnabled(true);
            }
        }
        Thread.sleep(100);
    }

    public void showWindow() {
        initPluginList();
        ImageIcon frameIcon = new ImageIcon(PluginMarket.class.getResource("/icons/frame.png"));
        labelIcon.setIcon(null);
        labelAuthor.setText("");
        labelOfficialSite.setText("");
        labelPluginName.setText("");
        labelVersion.setText("");
        textAreaPluginDescription.setText("");
        buttonInstall.setText(getTranslation("Install"));
        panel.setSize(800, 600);
        frame.setSize(800, 600);
        frame.setContentPane(PluginMarketBuilder.INSTANCE.panel);
        frame.setIconImage(frameIcon.getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public void hideWindow() {
        frame.setVisible(false);
    }

    private void addButtonInstallListener() {
        buttonInstall.addActionListener(e -> {
            String pluginName = (String) listPlugins.getSelectedValue();
            String pluginFullName = pluginName + ".jar";
            if (!DownloadUtil.getInstance().hasTask(pluginFullName)) {
                //没有下载过，开始下载
                JSONObject info = getPluginDetailInfo(pluginName);
                if (info != null) {
                    String downloadUrl = info.getString("url");
                    DownloadUtil.getInstance().downLoadFromUrl(downloadUrl, pluginFullName, "tmp/pluginsUpdate");
                    buttonInstall.setText(getTranslation("Cancel"));
                }
            } else {
                //取消下载
                DownloadUtil instance = DownloadUtil.getInstance();
                instance.cancelDownload(pluginFullName);
                threadPool.execute(() -> {
                    try {
                        while (instance.getDownloadStatus(pluginFullName) != DownloadManager.DOWNLOAD_INTERRUPTED) {
                            if (instance.getDownloadStatus(pluginFullName) == DownloadManager.DOWNLOAD_ERROR) {
                                break;
                            }
                            if (buttonInstall.isEnabled()) {
                                buttonInstall.setEnabled(false);
                            }
                            Thread.sleep(50);
                        }
                    } catch (InterruptedException ignored) {
                    }
                    //复位button
                    buttonInstall.setEnabled(true);
                    buttonInstall.setText(getTranslation("Install"));
                });
            }
        });
    }

    private void addSearchPluginListener() {
        textFieldSearchPlugin.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                searchKeywords = textFieldSearchPlugin.getText();
                isStartSearch = true;
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                searchKeywords = textFieldSearchPlugin.getText();
                isStartSearch = true;
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
            }
        });
    }

    private void addOpenPluginOfficialSiteListener() {
        labelOfficialSite.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Desktop desktop;
                String url;
                int firstIndex;
                int lastIndex;
                if (Desktop.isDesktopSupported()) {
                    desktop = Desktop.getDesktop();
                    try {
                        String text = labelOfficialSite.getText();
                        firstIndex = text.indexOf("'");
                        lastIndex = text.lastIndexOf("'");
                        url = text.substring(firstIndex + 1, lastIndex);
                        URI uri = new URI(url);
                        desktop.browse(uri);
                    } catch (Exception ignored) {
                    }
                }
            }
        });
    }

    private void addSelectPluginOnListListener() {
        listPlugins.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                String officialSite;
                String version;
                ImageIcon icon = null;
                String author;
                String pluginName;
                String description;
                pluginName = (String) listPlugins.getSelectedValue();
                JSONObject info = getPluginDetailInfo(pluginName);
                if (info != null) {
                    officialSite = info.getString("officialSite");
                    version = info.getString("version");
                    String imageUrl = info.getString("icon");
                    try {
                        icon = getImageByUrl(imageUrl, pluginName);
                    } catch (InterruptedException | IOException ignored) {
                    }
                    description = info.getString("description");
                    author = info.getString("author");
                    labelVersion.setText("Version:" + version);
                    labelIcon.setIcon(icon);
                    labelOfficialSite.setText("<html><a href='" + officialSite + "'><font size=\"4\">" + pluginName + "</font></a></html>");
                    labelPluginName.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                    textAreaPluginDescription.setText(description);
                    labelAuthor.setText(author);
                    buttonInstall.setVisible(true);
                }
            }
        });
    }

    private ImageIcon getImageByUrl(String url, String pluginName) throws InterruptedException, IOException {
        File icon = new File("tmp/$$" + pluginName);

        if (!icon.exists()) {
            DownloadUtil.getInstance().downLoadFromUrl(url, icon.getName(), "tmp");

            while (DownloadUtil.getInstance().getDownloadStatus(icon.getName()) != DownloadManager.DOWNLOAD_DONE) {
                if (DownloadUtil.getInstance().getDownloadStatus(icon.getName()) == DownloadManager.DOWNLOAD_ERROR) {
                    return null;
                }
                Thread.sleep(50);
            }
        }

        BufferedImage bitmap = ImageIO.read(icon);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        ImageIO.write(bitmap, "PNG", bytes);
        return new ImageIcon(bytes.toByteArray());
    }

    private JSONObject getPluginDetailInfo(String pluginName) {
        String infoUrl = NAME_PLUGIN_INFO_URL_MAP.get(pluginName);
        if (infoUrl != null) {
            try {
                return getPluginInfo(infoUrl);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    private static JSONObject getPluginInfo(String url) throws IOException {
        StringBuilder jsonUpdate = new StringBuilder();
        URL updateServer = new URL(url);
        URLConnection uc = updateServer.openConnection();
        uc.setConnectTimeout(3 * 1000);
        //防止屏蔽程序抓取而返回403错误
        uc.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36 Edg/80.0.361.57");
        try (BufferedReader br = new BufferedReader(new InputStreamReader(uc.getInputStream(), StandardCharsets.UTF_8))) {
            String eachLine;
            while ((eachLine = br.readLine()) != null) {
                jsonUpdate.append(eachLine);
            }
        }
        return JSONObject.parseObject(jsonUpdate.toString());
    }

    private void initPluginList() {
        try {
            JSONObject allPlugins = getPluginInfo("https://gitee.com/xuanxuF/File-Engine/raw/master/plugins.json");
            Set<String> pluginSet = allPlugins.keySet();
            for (String each : pluginSet) {
                NAME_PLUGIN_INFO_URL_MAP.put(each, allPlugins.getString(each));
            }
            if (NAME_PLUGIN_INFO_URL_MAP.isEmpty()) {
                buttonInstall.setEnabled(false);
            }
            buttonInstall.setVisible(false);
            listPlugins.setListData(pluginSet.toArray());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static PluginMarket getInstance() {
        return PluginMarketBuilder.INSTANCE;
    }
}
