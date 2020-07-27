package FileEngine.frames;

import FileEngine.download.DownloadManager;
import FileEngine.download.DownloadUtil;
import FileEngine.pluginSystem.PluginUtil;
import FileEngine.threadPool.CachedThreadPool;
import FileEngine.translate.TranslateUtil;
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
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

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

    private PluginMarket() {
        addSelectPluginOnListListener();
        addSearchPluginListener();
        addButtonInstallListener();
        addOpenPluginOfficialSiteListener();
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                String pluginName;
                String originString = buttonInstall.getText();
                while (SettingsFrame.isNotMainExit()) {
                    pluginName = (String) listPlugins.getSelectedValue();
                    if (pluginName != null) {
                        checkDownloadTask(labelProgress, buttonInstall, pluginName + ".jar", originString);
                    } else {
                        buttonInstall.setEnabled(false);
                        buttonInstall.setVisible(false);
                        labelAuthor.setText("");
                        labelIcon.setIcon(null);
                        labelOfficialSite.setText("");
                        labelPluginName.setText("");
                        labelProgress.setText("");
                        labelVersion.setText("");
                        textAreaPluginDescription.setText("");
                    }
                }
            } catch (InterruptedException ignored) {
            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            HashSet<String> pluginSet = new HashSet<>();
            try {
                while (SettingsFrame.isNotMainExit()) {
                    if (isStartSearch) {
                        isStartSearch = false;
                        if (searchKeywords == null || searchKeywords.isEmpty()) {
                            listPlugins.setListData(NAME_PLUGIN_INFO_URL_MAP.keySet().toArray());
                        } else {
                            pluginSet.clear();
                            for (String each : NAME_PLUGIN_INFO_URL_MAP.keySet()) {
                                if (each.toLowerCase().contains(searchKeywords.toLowerCase())) {
                                    pluginSet.add(each);
                                }
                            }
                            listPlugins.setListData(pluginSet.toArray());
                        }
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
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        int downloadingStatus = DownloadUtil.getInstance().getDownloadStatus(fileName);
        if (downloadingStatus != DownloadManager.DOWNLOAD_NO_TASK) {
            progress = DownloadUtil.getInstance().getDownloadProgress(fileName);
            label.setText(translateUtil.getTranslation("Downloading:") + (int) (progress * 100) + "%");

            if (downloadingStatus == DownloadManager.DOWNLOAD_DONE) {
                //下载完成，禁用按钮
                label.setText(translateUtil.getTranslation("Download Done"));
                label.setText(translateUtil.getTranslation("Downloaded"));
                button.setEnabled(false);
                button.setVisible(true);
                File updatePluginSign = new File("user/updatePlugin");
                if (!updatePluginSign.exists()) {
                    try {
                        updatePluginSign.createNewFile();
                    } catch (IOException ignored) {
                    }
                }
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_ERROR) {
                //下载错误，重置button
                label.setText(translateUtil.getTranslation("Download failed"));
                button.setText(translateUtil.getTranslation(originButtonString));
                button.setEnabled(true);
                button.setVisible(true);
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_DOWNLOADING) {
                //正在下载
                button.setText(translateUtil.getTranslation("Cancel"));
                button.setVisible(true);
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_INTERRUPTED) {
                //用户自行中断
                label.setText("");
                button.setText(translateUtil.getTranslation(originButtonString));
                button.setEnabled(true);
                button.setVisible(true);
            }
        } else {
            int index = fileName.indexOf(".");
            pluginName = fileName.substring(0, index);
            if (PluginUtil.getIdentifierByName(pluginName) != null) {
                label.setText("");
                button.setText(translateUtil.getTranslation("Installed"));
                button.setEnabled(false);
            } else {
                label.setText("");
                button.setText(translateUtil.getTranslation(originButtonString));
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
        textFieldSearchPlugin.setText("");
        buttonInstall.setText(TranslateUtil.getInstance().getTranslation("Install"));
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
            if (DownloadUtil.getInstance().getDownloadStatus(pluginFullName) == DownloadManager.DOWNLOAD_NO_TASK) {
                //没有下载过，开始下载
                JSONObject info = getPluginDetailInfo(pluginName);
                if (info != null) {
                    String downloadUrl = info.getString("url");
                    DownloadUtil.getInstance().downLoadFromUrl(downloadUrl, pluginFullName, "tmp/pluginsUpdate");
                    buttonInstall.setText(TranslateUtil.getInstance().getTranslation("Cancel"));
                }
            } else {
                //取消下载
                DownloadUtil instance = DownloadUtil.getInstance();
                instance.cancelDownload(pluginFullName);
                CachedThreadPool.getInstance().executeTask(() -> {
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
                    buttonInstall.setText(TranslateUtil.getInstance().getTranslation("Install"));
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
                    labelVersion.setText(TranslateUtil.getInstance().getTranslation("Version") + ":" + version);
                    labelIcon.setIcon(icon);
                    labelOfficialSite.setText("<html><a href='" + officialSite + "'><font size=\"4\">" + pluginName + "</font></a></html>");
                    labelPluginName.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                    textAreaPluginDescription.setText(description);
                    labelAuthor.setText(author);
                    buttonInstall.setVisible(true);
                    buttonInstall.setEnabled(true);
                }
            }
        });
    }

    private ImageIcon getImageByUrl(String url, String pluginName) throws InterruptedException, IOException {
        File icon = new File("tmp/$$" + pluginName);
        int count = 0;
        if (!icon.exists()) {
            DownloadUtil.getInstance().downLoadFromUrl(url, icon.getName(), "tmp");

            while (DownloadUtil.getInstance().getDownloadStatus(icon.getName()) != DownloadManager.DOWNLOAD_DONE) {
                if (DownloadUtil.getInstance().getDownloadStatus(icon.getName()) == DownloadManager.DOWNLOAD_ERROR) {
                    return null;
                }
                if (count > 30) {
                    break;
                }
                count++;
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
                return getPluginInfo(infoUrl, pluginName + ".json");
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    private static JSONObject getPluginInfo(String url, String saveFileName) throws IOException, InterruptedException {
        DownloadUtil downloadUtil = DownloadUtil.getInstance();
        int downloadStatus = downloadUtil.getDownloadStatus(saveFileName);
        if (downloadStatus != DownloadManager.DOWNLOAD_DOWNLOADING) {
            //判断是否已下载完成
            if (downloadStatus != DownloadManager.DOWNLOAD_DONE) {
                downloadUtil.downLoadFromUrl(url,
                        saveFileName, "tmp");
                int count = 0;
                boolean isError = false;
                //wait for task
                while (downloadUtil.getDownloadStatus(saveFileName) != DownloadManager.DOWNLOAD_DONE) {
                    count++;
                    if (count >= 3) {
                        isError = true;
                        break;
                    }
                    Thread.sleep(1000);
                }
                if (isError) {
                    return null;
                }
            }
            //已下载完，返回json
            String eachLine;
            StringBuilder strBuilder = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("tmp/" + saveFileName), StandardCharsets.UTF_8))) {
                while ((eachLine = br.readLine()) != null) {
                    strBuilder.append(eachLine);
                }
            }
            return JSONObject.parseObject(strBuilder.toString());
        }
        return null;
    }

    private void initPluginList() {
        try {
            JSONObject allPlugins = getPluginInfo(
                    "https://raw.githubusercontent.com/XUANXUQAQ/File-Engine-Version/master/plugins.json",
                    "allPluginsList.json");
            if (allPlugins != null) {
                Set<String> pluginSet = allPlugins.keySet();
                for (String each : pluginSet) {
                    NAME_PLUGIN_INFO_URL_MAP.put(each, allPlugins.getString(each));
                }
                listPlugins.setListData(pluginSet.toArray());
            }
            buttonInstall.setEnabled(false);
            buttonInstall.setVisible(false);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static PluginMarket getInstance() {
        return PluginMarketBuilder.INSTANCE;
    }
}
