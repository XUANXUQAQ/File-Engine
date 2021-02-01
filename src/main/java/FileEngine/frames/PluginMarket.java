package FileEngine.frames;

import FileEngine.annotation.EventRegister;
import FileEngine.configs.AllConfigs;
import FileEngine.eventHandler.EventManagement;
import FileEngine.eventHandler.impl.download.StartDownloadEvent;
import FileEngine.eventHandler.impl.download.StopDownloadEvent;
import FileEngine.eventHandler.impl.frame.pluginMarket.ShowPluginMarket;
import FileEngine.eventHandler.impl.stop.RestartEvent;
import FileEngine.services.download.DownloadManager;
import FileEngine.services.download.DownloadService;
import FileEngine.services.pluginSystem.PluginService;
import FileEngine.utils.CachedThreadPoolUtil;
import FileEngine.utils.TranslateUtil;
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
import java.lang.reflect.Method;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
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
    private JScrollPane scrollpanePlugins;
    private JLabel placeholder0;
    private JLabel placeholder1;
    private JLabel placeholder2;
    private final JFrame frame = new JFrame("Plugin Market");
    //保存插件名称和url的映射关系
    private final HashMap<String, String> NAME_PLUGIN_INFO_URL_MAP = new HashMap<>();

    private PluginMarket() {
        addSelectPluginOnListListener();
        addSearchPluginListener();
        addButtonInstallListener();
        addOpenPluginOfficialSiteListener();
        frame.dispose();
        frame.setUndecorated(true);
        frame.getRootPane().setWindowDecorationStyle(JRootPane.FRAME);
        CachedThreadPoolUtil.getInstance().executeTask(new Thread(() -> {
            try {
                String pluginName;
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.isNotMainExit()) {
                    TimeUnit.MILLISECONDS.sleep(100);
                    pluginName = (String) listPlugins.getSelectedValue();
                    if (pluginName == null) {
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
        }, "show plugin info on gui"));
    }

    private void showWindow() {
        ImageIcon frameIcon = new ImageIcon(PluginMarket.class.getResource("/icons/frame.png"));
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        labelIcon.setIcon(null);
        labelAuthor.setText("");
        labelOfficialSite.setText("");
        labelPluginName.setText("");
        labelVersion.setText("");
        textAreaPluginDescription.setText("");
        textFieldSearchPlugin.setText("");
        buttonInstall.setText(translateUtil.getTranslation("Install"));
        panel.setSize(800, 600);
        frame.setSize(800, 600);
        frame.setContentPane(getInstance().panel);
        frame.setIconImage(frameIcon.getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setTitle(translateUtil.getTranslation("Plugin Market"));
        SwingUtilities.invokeLater(() -> frame.setVisible(true));
        LoadingPanel loadingPanel = new LoadingPanel("loading...");
        loadingPanel.setSize(800, 600);
        frame.setGlassPane(loadingPanel);
        loadingPanel.start();
        initPluginList();
        loadingPanel.stop();
    }

    private void getAllPluginsDetailInfo() {
        for (String each : NAME_PLUGIN_INFO_URL_MAP.keySet()) {
            getPluginDetailInfo(each);
        }
    }

    @EventRegister
    @SuppressWarnings("unused")
    public static void registerEventHandler() {
        EventManagement eventManagement = EventManagement.getInstance();

        eventManagement.register(ShowPluginMarket.class, event -> getInstance().showWindow());

        eventManagement.registerListener(RestartEvent.class, () -> getInstance().hideWindow());
    }

    private void hideWindow() {
        SwingUtilities.invokeLater(() -> frame.setVisible(false));
    }

    private void addButtonInstallListener() {
        EventManagement eventManagement = EventManagement.getInstance();
        AtomicBoolean isDownloadStarted = new AtomicBoolean(false);
        ConcurrentHashMap<String, DownloadManager> downloadManagerConcurrentHashMap = new ConcurrentHashMap<>();
        buttonInstall.addActionListener(e -> {
            String pluginName = (String) listPlugins.getSelectedValue();
            if (isDownloadStarted.get()) {
                //取消下载
                eventManagement.putEvent(new StopDownloadEvent(downloadManagerConcurrentHashMap.get(pluginName)));
            } else {
                String pluginFullName = pluginName + ".jar";
                JSONObject info = getPluginDetailInfo(pluginName);
                if (info != null) {
                    DownloadManager downloadManager = new DownloadManager(
                            info.getString("url"),
                            pluginFullName,
                            new File("tmp", "pluginsUpdate").getAbsolutePath()
                    );
                    downloadManagerConcurrentHashMap.put(pluginName, downloadManager);
                    eventManagement.putEvent(new StartDownloadEvent(downloadManager));
                    var getStringMethod = new Object() {
                        Method getString = null;
                    };
                    try {
                        getStringMethod.getString = listPlugins.getClass().getDeclaredMethod("getSelectedValue");
                    } catch (NoSuchMethodException noSuchMethodException) {
                        noSuchMethodException.printStackTrace();
                    }
                    CachedThreadPoolUtil.getInstance().executeTask(
                            new Thread(() -> SetDownloadProgress.setProgress(labelProgress,
                                    buttonInstall,
                                    downloadManager,
                                    isDownloadStarted,
                                    new File("user/updatePlugin"),
                                    pluginName,
                                    getStringMethod.getString,
                                    listPlugins), "show plugin download progress")
                    );
                }
            }
        });
    }

    private void addSearchPluginListener() {
        class search {
            final String searchKeywords;
            final CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
            search(String searchKeywords) {
                this.searchKeywords = searchKeywords;
            }
            void doSearch() {
                cachedThreadPoolUtil.executeTask(new Thread(() -> {
                    String tmp;
                    HashSet<String> pluginSet = new HashSet<>();
                    if ((tmp = searchKeywords) == null || tmp.isEmpty()) {
                        listPlugins.setListData(NAME_PLUGIN_INFO_URL_MAP.keySet().toArray());
                    } else {
                        for (String each : NAME_PLUGIN_INFO_URL_MAP.keySet()) {
                            if (each.toLowerCase().contains(tmp.toLowerCase())) {
                                pluginSet.add(each);
                            }
                        }
                        listPlugins.setListData(pluginSet.toArray());
                    }
                }, "search plugin"));
            }
        }

        textFieldSearchPlugin.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                new search(textFieldSearchPlugin.getText()).doSearch();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                new search(textFieldSearchPlugin.getText()).doSearch();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {}
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
        final AtomicBoolean isStartGetPluginInfo = new AtomicBoolean(false);

        class getPluginInfo {
            final CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
            final String pluginName;
            final TranslateUtil translateUtil = TranslateUtil.getInstance();
            getPluginInfo(String pluginName) {
                this.pluginName = pluginName;
            }
            void doGet() {
                isStartGetPluginInfo.set(false);
                cachedThreadPoolUtil.executeTask(new Thread(() -> {
                    if (isStartGetPluginInfo.get()) {
                        //用户重新点击
                        return;
                    }
                    JSONObject info = getPluginDetailInfo(pluginName);
                    if (info == null){
                        return;
                    }
                    String officialSite = info.getString("officialSite");
                    String version = info.getString("version");
                    String imageUrl = info.getString("icon");
                    String description = info.getString("description");
                    String author = info.getString("author");
                    labelVersion.setText(TranslateUtil.getInstance().getTranslation("Version") + ":" + version);
                    labelOfficialSite.setText("<html><a href='" + officialSite + "'><font size=\"4\">" + pluginName + "</font></a></html>");
                    labelPluginName.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                    textAreaPluginDescription.setText(description);
                    labelAuthor.setText(author);
                    buttonInstall.setVisible(true);
                    boolean hasPlugin = PluginService.getInstance().hasPlugin(pluginName);
                    boolean downloaded = DownloadService.getInstance().isTaskDone(
                            new DownloadManager(
                                    null,
                                    pluginName + ".jar",
                                    new File("tmp", "pluginsUpdate").getAbsolutePath()
                            )
                    );
                    if (hasPlugin) {
                        buttonInstall.setEnabled(false);
                        buttonInstall.setText(translateUtil.getTranslation("Installed"));
                    } else if (downloaded) {
                        buttonInstall.setEnabled(false);
                        buttonInstall.setText(translateUtil.getTranslation("Downloaded"));
                    } else {
                        buttonInstall.setEnabled(true);
                        buttonInstall.setText(translateUtil.getTranslation("Install"));
                    }
                    //用户重新点击
                    try {
                        ImageIcon icon = getImageByUrl(imageUrl, pluginName);
                        if (isStartGetPluginInfo.get()) {
                            return;
                        }
                        labelIcon.setIcon(icon);
                    } catch (IOException ignored) {
                    }

                }, "get plugin json info"));
            }
        }
        listPlugins.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                String pluginName = (String) listPlugins.getSelectedValue();
                buttonInstall.setVisible(false);
                buttonInstall.setEnabled(false);
                isStartGetPluginInfo.set(true);
                labelVersion.setText("");
                labelOfficialSite.setText("");
                labelPluginName.setText("");
                textAreaPluginDescription.setText("");
                labelAuthor.setText("");
                labelIcon.setIcon(null);
                labelProgress.setText("");
                if (pluginName != null) {
                    new getPluginInfo(pluginName).doGet();
                }
            }
        });
    }

    private ImageIcon getImageByUrl(String url, String pluginName) throws IOException {
        File icon = new File("tmp/$$" + pluginName);
        DownloadService downloadService = DownloadService.getInstance();
        DownloadManager downloadManager = new DownloadManager(
                url,
                icon.getName(),
                "tmp"
        );
        EventManagement.getInstance().putEvent(new StartDownloadEvent(downloadManager));
        downloadService.waitForDownloadTask(downloadManager, 10000);

        BufferedImage bitmap = ImageIO.read(icon);
        try (ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            ImageIO.write(bitmap, "PNG", bytes);
            return new ImageIcon(bytes.toByteArray());
        }
    }

    private JSONObject getPluginDetailInfo(String pluginName) {
        String infoUrl = NAME_PLUGIN_INFO_URL_MAP.get(pluginName);
        if (infoUrl != null) {
            try {
                return getPluginInfo(infoUrl, pluginName + ".json");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    private static JSONObject getPluginInfo(String url, String saveFileName) throws IOException {
        DownloadService downloadService = DownloadService.getInstance();
        DownloadManager downloadManager = new DownloadManager(
                url,
                saveFileName,
                new File("tmp").getAbsolutePath()
        );
        EventManagement.getInstance().putEvent(new StartDownloadEvent(downloadManager));
        downloadService.waitForDownloadTask(downloadManager, 10000);
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

    private String getPluginListUrl() {
        AllConfigs allConfigs = AllConfigs.getInstance();
        return allConfigs.getUpdateUrlFromMap().pluginListUrl;
    }

    private void initPluginList() {
        try {
            buttonInstall.setEnabled(false);
            buttonInstall.setVisible(false);
            String url = getPluginListUrl();

            JSONObject allPlugins = getPluginInfo(url, "allPluginsList.json");
            if (allPlugins == null) {
                return;
            }
            Set<String> pluginSet = allPlugins.keySet();
            for (String each : pluginSet) {
                NAME_PLUGIN_INFO_URL_MAP.put(each, allPlugins.getString(each));
            }
            listPlugins.setListData(pluginSet.toArray());
            getAllPluginsDetailInfo();
            //防止执行太快导致窗口无法正常关闭
            TimeUnit.MILLISECONDS.sleep(10);
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
