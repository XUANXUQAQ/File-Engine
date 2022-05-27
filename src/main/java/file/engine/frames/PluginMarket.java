package file.engine.frames;

import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.download.IsTaskDoneBeforeEvent;
import file.engine.event.handler.impl.download.StartDownloadEvent;
import file.engine.event.handler.impl.download.StopDownloadEvent;
import file.engine.event.handler.impl.frame.pluginMarket.ShowPluginMarket;
import file.engine.event.handler.impl.plugin.CheckPluginExistEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.frames.components.LoadingPanel;
import file.engine.services.TranslateService;
import file.engine.services.download.DownloadManager;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.utils.DpiUtil;
import file.engine.utils.gson.GsonUtil;

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
import java.util.*;
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
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                String pluginName;
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.notMainExit()) {
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
        });
    }

    /**
     * 显示插件窗口
     */
    private void showWindow() {
        double dpi = DpiUtil.getDpi();
        int width = (int) (1000 / dpi), height = (int) (600 / dpi);
        ImageIcon frameIcon = new ImageIcon(Objects.requireNonNull(PluginMarket.class.getResource("/icons/frame.png")));
        TranslateService translateService = TranslateService.getInstance();
        labelIcon.setIcon(null);
        labelAuthor.setText("");
        labelOfficialSite.setText("");
        labelPluginName.setText("");
        labelVersion.setText("");
        textAreaPluginDescription.setText("");
        textFieldSearchPlugin.setText("");
        buttonInstall.setText(translateService.getTranslation("Install"));
        panel.setSize(width, height);
        frame.setSize(width, height);
        frame.setContentPane(getInstance().panel);
        frame.setIconImage(frameIcon.getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.setResizable(true);
        frame.setLocationRelativeTo(null);
        frame.setTitle(translateService.getTranslation("Plugin Market"));
        SwingUtilities.invokeLater(() -> frame.setVisible(true));
        LoadingPanel loadingPanel = new LoadingPanel("loading...");
        loadingPanel.setSize(width, height);
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

    @EventRegister(registerClass = ShowPluginMarket.class)
    private static void showPluginMarketEvent(Event event) {
        getInstance().showWindow();
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void hideWindowListener(Event event) {
        getInstance().hideWindow();
    }

    private void hideWindow() {
        SwingUtilities.invokeLater(() -> frame.setVisible(false));
    }

    /**
     * 点击安装插件
     */
    private void addButtonInstallListener() {
        EventManagement eventManagement = EventManagement.getInstance();
        ConcurrentHashMap<String, DownloadManager> downloadManagerConcurrentHashMap = new ConcurrentHashMap<>();
        buttonInstall.addActionListener(e -> {
            String pluginName = (String) listPlugins.getSelectedValue();
            DownloadManager downloadManager = downloadManagerConcurrentHashMap.get(pluginName);
            if (downloadManager != null && downloadManager.getDownloadStatus() == Constants.Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                //取消下载
                eventManagement.putEvent(new StopDownloadEvent(downloadManager));
            } else {
                String pluginFullName = pluginName + ".jar";
                Map<String, Object> info = getPluginDetailInfo(pluginName);
                if (info != null) {
                    downloadManager = new DownloadManager(
                            (String) info.get("url"),
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
                    DownloadManager finalDownloadManager = downloadManager;
                    CachedThreadPoolUtil.getInstance().executeTask(
                            () -> SetDownloadProgress.setProgress(labelProgress,
                                    buttonInstall,
                                    finalDownloadManager,
                                    () -> finalDownloadManager.fileName.equals(listPlugins.getSelectedValue() + ".jar"),
                                    new File("user/updatePlugin"),
                                    pluginName,
                                    getStringMethod.getString,
                                    listPlugins)
                    );
                }
            }
        });
    }

    /**
     * 根据关键字搜索插件
     */
    private void addSearchPluginListener() {
        class search {
            final String searchKeywords;
            final CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();

            search(String searchKeywords) {
                this.searchKeywords = searchKeywords;
            }

            void doSearch() {
                cachedThreadPoolUtil.executeTask(() -> {
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
                });
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
            public void changedUpdate(DocumentEvent e) {
            }
        });
    }

    /**
     * 当用户点击官网后打开浏览器进入官网
     */
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

    /**
     * 当鼠标点击后在右边显示插件的基本信息
     */
    private void addSelectPluginOnListListener() {
        final AtomicBoolean isStartGetPluginInfo = new AtomicBoolean(false);

        class getPluginInfo {
            final CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
            final String pluginName;
            final TranslateService translateUtil = TranslateService.getInstance();

            getPluginInfo(String pluginName) {
                this.pluginName = pluginName;
            }

            void doGet() {
                isStartGetPluginInfo.set(false);
                cachedThreadPoolUtil.executeTask(() -> {
                    if (isStartGetPluginInfo.get()) {
                        //用户重新点击
                        return;
                    }
                    Map<String, Object> info = getPluginDetailInfo(pluginName);
                    if (info == null) {
                        return;
                    }
                    String officialSite = (String) info.get("officialSite");
                    String version = (String) info.get("version");
                    String imageUrl = (String) info.get("icon");
                    String description = (String) info.get("description");
                    String author = (String) info.get("author");
                    labelVersion.setText(TranslateService.getInstance().getTranslation("Version") + ":" + version);
                    labelOfficialSite.setText("<html><a href='" + officialSite + "'><font size=\"4\">" + pluginName + "</font></a></html>");
                    labelPluginName.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                    textAreaPluginDescription.setText(description);
                    labelAuthor.setText(author);
                    buttonInstall.setVisible(true);
                    CheckPluginExistEvent checkPluginExistEvent = new CheckPluginExistEvent(pluginName);
                    EventManagement eventManagement = EventManagement.getInstance();
                    eventManagement.putEvent(checkPluginExistEvent);
                    eventManagement.waitForEvent(checkPluginExistEvent);
                    Optional<Boolean> hasPluginOptional = checkPluginExistEvent.getReturnValue();
                    IsTaskDoneBeforeEvent isTaskDoneBeforeEvent = new IsTaskDoneBeforeEvent(new DownloadManager(
                            null,
                            pluginName + ".jar",
                            new File("tmp", "pluginsUpdate").getAbsolutePath()
                    ));
                    eventManagement.putEvent(isTaskDoneBeforeEvent);
                    eventManagement.waitForEvent(isTaskDoneBeforeEvent);
                    Optional<Boolean> downloadedOptional = isTaskDoneBeforeEvent.getReturnValue();
                    var obj = new Object() {
                        boolean hasPlugin;
                        boolean downloaded;
                    };

                    hasPluginOptional.ifPresentOrElse((ret) -> obj.hasPlugin = ret, () -> obj.hasPlugin = false);
                    downloadedOptional.ifPresentOrElse((ret) -> obj.downloaded = ret, () -> obj.downloaded = false);
                    if (obj.hasPlugin) {
                        buttonInstall.setEnabled(false);
                        buttonInstall.setText(translateUtil.getTranslation("Installed"));
                    } else if (obj.downloaded) {
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
                });
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

    /**
     * 通过url获取图片
     *
     * @param url        url
     * @param pluginName 插件名
     * @return 图片
     * @throws IOException IOException
     */
    private ImageIcon getImageByUrl(String url, String pluginName) throws IOException {
        File icon = new File("tmp/$$" + pluginName);
        DownloadManager downloadManager = new DownloadManager(
                url,
                icon.getName(),
                "tmp"
        );
        EventManagement.getInstance().putEvent(new StartDownloadEvent(downloadManager));
        if (!downloadManager.waitFor(10000)) {
            return null;
        }

        BufferedImage bitmap = ImageIO.read(icon);
        try (ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            ImageIO.write(bitmap, "PNG", bytes);
            return new ImageIcon(bytes.toByteArray());
        }
    }

    private Map<String, Object> getPluginDetailInfo(String pluginName) {
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

    /**
     * 获取插件的基本信息
     *
     * @param url          获取url
     * @param saveFileName 保存的文件名
     * @return 插件信息Map
     * @throws IOException IOException
     */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> getPluginInfo(String url, String saveFileName) throws IOException {
        DownloadManager downloadManager = new DownloadManager(
                url,
                saveFileName,
                new File("tmp").getAbsolutePath()
        );
        EventManagement.getInstance().putEvent(new StartDownloadEvent(downloadManager));
        downloadManager.waitFor(10000);
        //已下载完，返回json
        String eachLine;
        StringBuilder strBuilder = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("tmp/" + saveFileName), StandardCharsets.UTF_8))) {
            while ((eachLine = br.readLine()) != null) {
                strBuilder.append(eachLine);
            }
        }
        return GsonUtil.getInstance().getGson().fromJson(strBuilder.toString(), Map.class);
    }

    /**
     * 获取所有插件列表url
     *
     * @return url
     */
    private String getPluginListUrl() {
        AllConfigs allConfigs = AllConfigs.getInstance();
        return allConfigs.getUpdateUrlFromMap().pluginListUrl;
    }

    /**
     * 初始化插件列表
     */
    private void initPluginList() {
        try {
            buttonInstall.setEnabled(false);
            buttonInstall.setVisible(false);
            String url = getPluginListUrl();

            Map<String, Object> allPlugins = getPluginInfo(url, "allPluginsList.json");
            if (allPlugins == null) {
                return;
            }
            Set<String> pluginSet = allPlugins.keySet();
            for (String each : pluginSet) {
                NAME_PLUGIN_INFO_URL_MAP.put(each, (String) allPlugins.get(each));
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
