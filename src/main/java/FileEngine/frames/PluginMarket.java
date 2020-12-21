package FileEngine.frames;

import FileEngine.configs.AllConfigs;
import FileEngine.configs.Enums;
import FileEngine.download.DownloadUtil;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventHandler;
import FileEngine.eventHandler.EventUtil;
import FileEngine.eventHandler.impl.SetDefaultSwingLaf;
import FileEngine.eventHandler.impl.download.StartDownloadEvent;
import FileEngine.eventHandler.impl.download.StopDownloadEvent;
import FileEngine.eventHandler.impl.frame.pluginMarket.HidePluginMarketEvent;
import FileEngine.eventHandler.impl.frame.pluginMarket.ShowPluginMarket;
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
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class PluginMarket {

    private static volatile PluginMarket INSTANCE = null;

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
        frame.dispose();
        frame.setUndecorated(true);
        frame.getRootPane().setWindowDecorationStyle(JRootPane.FRAME);
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                String pluginName;
                String originString = buttonInstall.getText();
                while (EventUtil.getInstance().isNotMainExit()) {
                    TimeUnit.MILLISECONDS.sleep(100);
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
            } catch (InterruptedException | IOException ignored) {
            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            HashSet<String> pluginSet = new HashSet<>();
            try {
                while (EventUtil.getInstance().isNotMainExit()) {
                    TimeUnit.MILLISECONDS.sleep(100);
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
                    TimeUnit.MILLISECONDS.sleep(100);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void checkDownloadTask(JLabel label, JButton button, String fileName, String originButtonString) throws IOException {
        //设置进度显示线程
        double progress;
        String pluginName;
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        Enums.DownloadStatus downloadingStatus = DownloadUtil.getInstance().getDownloadStatus(fileName);
        if (downloadingStatus != Enums.DownloadStatus.DOWNLOAD_NO_TASK) {
            progress = DownloadUtil.getInstance().getDownloadProgress(fileName);
            label.setText(translateUtil.getTranslation("Downloading:") + (int) (progress * 100) + "%");

            if (downloadingStatus == Enums.DownloadStatus.DOWNLOAD_DONE) {
                //下载完成，禁用按钮
                label.setText(translateUtil.getTranslation("Download Done"));
                label.setText(translateUtil.getTranslation("Downloaded"));
                button.setEnabled(false);
                button.setVisible(true);
                File updatePluginSign = new File("user/updatePlugin");
                if (!updatePluginSign.exists()) {
                    updatePluginSign.createNewFile();
                }
            } else if (downloadingStatus == Enums.DownloadStatus.DOWNLOAD_ERROR) {
                //下载错误，重置button
                label.setText(translateUtil.getTranslation("Download failed"));
                button.setText(translateUtil.getTranslation(originButtonString));
                button.setEnabled(true);
                button.setVisible(true);
            } else if (downloadingStatus == Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                //正在下载
                button.setText(translateUtil.getTranslation("Cancel"));
                button.setVisible(true);
            } else if (downloadingStatus == Enums.DownloadStatus.DOWNLOAD_INTERRUPTED) {
                //用户自行中断
                label.setText("");
                button.setText(translateUtil.getTranslation(originButtonString));
                button.setEnabled(true);
                button.setVisible(true);
            }
        } else {
            int index = fileName.indexOf(".");
            pluginName = fileName.substring(0, index);
            if (PluginUtil.getInstance().getIdentifierByName(pluginName) != null) {
                label.setText("");
                button.setText(translateUtil.getTranslation("Installed"));
                button.setEnabled(false);
            } else {
                label.setText("");
                button.setText(translateUtil.getTranslation(originButtonString));
                button.setEnabled(true);
            }
        }
    }

    private void showWindow() {
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
        frame.setContentPane(getInstance().panel);
        frame.setIconImage(frameIcon.getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setTitle(TranslateUtil.getInstance().getTranslation("Plugin Market"));
        frame.setVisible(true);
        EventUtil.getInstance().putEvent(new SetDefaultSwingLaf());
    }

    public static void registerEventHandler() {
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.register(ShowPluginMarket.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().showWindow();
            }
        });

        eventUtil.register(HidePluginMarketEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().hideWindow();
            }
        });
    }

    private void hideWindow() {
        frame.setVisible(false);
    }

    private void addButtonInstallListener() {
        DownloadUtil instance = DownloadUtil.getInstance();
        buttonInstall.addActionListener(e -> {
            String pluginName = (String) listPlugins.getSelectedValue();
            String pluginFullName = pluginName + ".jar";
            Enums.DownloadStatus downloadStatus = instance.getDownloadStatus(pluginFullName);
            if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_NO_TASK ||
                    downloadStatus == Enums.DownloadStatus.DOWNLOAD_INTERRUPTED ||
                    downloadStatus == Enums.DownloadStatus.DOWNLOAD_ERROR) {
                //没有下载过，开始下载
                JSONObject info = getPluginDetailInfo(pluginName);
                if (info != null) {
                    String downloadUrl = info.getString("url");
                    EventUtil.getInstance().putEvent(new StartDownloadEvent(
                            downloadUrl, pluginFullName,
                            new File(AllConfigs.getInstance().getTmp(), "pluginsUpdate").getAbsolutePath()));

                    buttonInstall.setText(TranslateUtil.getInstance().getTranslation("Cancel"));
                }
            } else if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                //取消下载
                EventUtil.getInstance().putEvent(new StopDownloadEvent(pluginFullName));
                //复位button
                buttonInstall.setEnabled(true);
                buttonInstall.setText(TranslateUtil.getInstance().getTranslation("Install"));
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
        AtomicBoolean isStartGetPluginInfo = new AtomicBoolean(false);
        listPlugins.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                buttonInstall.setVisible(false);
                buttonInstall.setEnabled(false);
                isStartGetPluginInfo.set(true);
                labelVersion.setText("");
                labelOfficialSite.setText("");
                labelPluginName.setText("");
                textAreaPluginDescription.setText("");
                labelAuthor.setText("");
                labelIcon.setIcon(null);
            }
        });
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (EventUtil.getInstance().isNotMainExit()) {
                    if (isStartGetPluginInfo.get()) {
                        isStartGetPluginInfo.set(false);
                        String officialSite;
                        String version;
                        ImageIcon icon;
                        String author;
                        String pluginName;
                        String description;
                        pluginName = (String) listPlugins.getSelectedValue();
                        if (pluginName != null) {
                            if (isStartGetPluginInfo.get()) {
                                //用户重新点击
                                continue;
                            }
                            JSONObject info = getPluginDetailInfo(pluginName);
                            if (info != null) {
                                officialSite = info.getString("officialSite");
                                version = info.getString("version");
                                String imageUrl = info.getString("icon");
                                description = info.getString("description");
                                author = info.getString("author");
                                labelVersion.setText(TranslateUtil.getInstance().getTranslation("Version") + ":" + version);
                                labelOfficialSite.setText("<html><a href='" + officialSite + "'><font size=\"4\">" + pluginName + "</font></a></html>");
                                labelPluginName.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                                textAreaPluginDescription.setText(description);
                                labelAuthor.setText(author);
                                buttonInstall.setVisible(true);
                                buttonInstall.setEnabled(true);
                                if (isStartGetPluginInfo.get()) {
                                    //用户重新点击
                                    continue;
                                } else {
                                    try {
                                        icon = getImageByUrl(imageUrl, pluginName);
                                        labelIcon.setIcon(icon);
                                    } catch (IOException ignored) {
                                    }
                                }
                            } else {
                                labelPluginName.setText(TranslateUtil.getInstance().getTranslation("Failed to obtain plugin information"));
                            }
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(100);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private ImageIcon getImageByUrl(String url, String pluginName) throws InterruptedException, IOException {
        File icon = new File("tmp/$$" + pluginName);
        int count = 0;
        if (!icon.exists()) {
            EventUtil.getInstance().putEvent(new StartDownloadEvent(url, icon.getName(), "tmp"));

            while (DownloadUtil.getInstance().getDownloadStatus(icon.getName()) != Enums.DownloadStatus.DOWNLOAD_DONE) {
                if (DownloadUtil.getInstance().getDownloadStatus(icon.getName()) == Enums.DownloadStatus.DOWNLOAD_ERROR) {
                    return null;
                }
                if (count > 30) {
                    break;
                }
                count++;
                TimeUnit.MILLISECONDS.sleep(50);
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
        Enums.DownloadStatus downloadStatus = downloadUtil.getDownloadStatus(saveFileName);
        if (downloadStatus != Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
            //判断是否已下载完成
            if (downloadStatus != Enums.DownloadStatus.DOWNLOAD_DONE) {
                EventUtil.getInstance().putEvent(new StartDownloadEvent(url, saveFileName, AllConfigs.getInstance().getTmp().getAbsolutePath()));

                int count = 0;
                boolean isError = false;
                //wait for task
                while (downloadUtil.getDownloadStatus(saveFileName) != Enums.DownloadStatus.DOWNLOAD_DONE) {
                    count++;
                    if (count >= 3) {
                        isError = true;
                        break;
                    }
                    TimeUnit.SECONDS.sleep(1);
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

    private String getPluginListUrl() {
        //todo 添加更新服务器地址
        switch (AllConfigs.getInstance().getUpdateAddress()) {
            case "jsdelivr CDN":
                return "https://cdn.jsdelivr.net/gh/XUANXUQAQ/File-Engine-Version/plugins.json";
            case "GitHub":
                return "https://raw.githubusercontent.com/XUANXUQAQ/File-Engine-Version/master/plugins.json";
            case "GitHack":
                return "https://raw.githack.com/XUANXUQAQ/File-Engine-Version/master/plugins.json";
            case "Gitee":
                return "https://gitee.com/XUANXUQAQ/file-engine-version/raw/master/plugins.json";
            default:
                return null;
        }
    }

    private void initPluginList() {
        try {
            String url = getPluginListUrl();
            if (url != null) {
                JSONObject allPlugins = getPluginInfo(
                        url,
                        "allPluginsList.json");
                if (allPlugins != null) {
                    Set<String> pluginSet = allPlugins.keySet();
                    for (String each : pluginSet) {
                        NAME_PLUGIN_INFO_URL_MAP.put(each, allPlugins.getString(each));
                    }
                    listPlugins.setListData(pluginSet.toArray());
                }
            }
            buttonInstall.setEnabled(false);
            buttonInstall.setVisible(false);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    private static PluginMarket getInstance() {
        if (INSTANCE == null) {
            synchronized (PluginMarket.class) {
                if (INSTANCE == null) {
                    INSTANCE = new PluginMarket();
                }
            }
        }
        return INSTANCE;
    }
}
