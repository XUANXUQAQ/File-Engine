package file.engine.frames;

import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.download.StartDownloadEvent;
import file.engine.event.handler.impl.download.StopDownloadEvent;
import file.engine.event.handler.impl.frame.pluginMarket.ShowPluginMarket;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.frames.components.LoadingPanel;
import file.engine.services.TranslateService;
import file.engine.services.download.DownloadManager;
import file.engine.services.download.DownloadService;
import file.engine.services.plugin.system.PluginService;
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
    private boolean isFramePrepared = false;

    private PluginMarket() {
    }

    private void prepareFrame() {
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
        if (!isFramePrepared) {
            isFramePrepared = true;
            prepareFrame();
        }
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
        if (INSTANCE == null) {
            return;
        }
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

                    var obj = new Object() {
                        boolean hasPlugin;
                        boolean downloaded;
                    };
                    obj.hasPlugin = PluginService.getInstance().hasPlugin(pluginName);
                    obj.downloaded = DownloadService.getInstance().isTaskDoneBefore(new DownloadManager(
                            null,
                            pluginName + ".jar",
                            new File("tmp", "pluginsUpdate").getAbsolutePath()
                    ));
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

    {
// GUI initializer generated by IntelliJ IDEA GUI Designer
// >>> IMPORTANT!! <<<
// DO NOT EDIT OR ADD ANY CODE HERE!
        $$$setupUI$$$();
    }

    /**
     * Method generated by IntelliJ IDEA GUI Designer
     * >>> IMPORTANT!! <<<
     * DO NOT edit this method OR call it in your code!
     *
     * @noinspection ALL
     */
    private void $$$setupUI$$$() {
        panel = new JPanel();
        panel.setLayout(new com.intellij.uiDesigner.core.GridLayoutManager(5, 8, new Insets(0, 0, 0, 0), -1, -1));
        scrollPluginDescription = new JScrollPane();
        panel.add(scrollPluginDescription, new com.intellij.uiDesigner.core.GridConstraints(4, 1, 1, 7, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_CENTER, com.intellij.uiDesigner.core.GridConstraints.FILL_BOTH, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        textAreaPluginDescription = new JTextArea();
        textAreaPluginDescription.setEditable(false);
        scrollPluginDescription.setViewportView(textAreaPluginDescription);
        textFieldSearchPlugin = new JTextField();
        panel.add(textFieldSearchPlugin, new com.intellij.uiDesigner.core.GridConstraints(0, 0, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        final com.intellij.uiDesigner.core.Spacer spacer1 = new com.intellij.uiDesigner.core.Spacer();
        panel.add(spacer1, new com.intellij.uiDesigner.core.GridConstraints(0, 7, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_CENTER, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        final com.intellij.uiDesigner.core.Spacer spacer2 = new com.intellij.uiDesigner.core.Spacer();
        panel.add(spacer2, new com.intellij.uiDesigner.core.GridConstraints(1, 7, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_CENTER, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        labelVersion = new JLabel();
        labelVersion.setText("  ");
        panel.add(labelVersion, new com.intellij.uiDesigner.core.GridConstraints(3, 6, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProgress = new JLabel();
        labelProgress.setText("  ");
        panel.add(labelProgress, new com.intellij.uiDesigner.core.GridConstraints(2, 7, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final com.intellij.uiDesigner.core.Spacer spacer3 = new com.intellij.uiDesigner.core.Spacer();
        panel.add(spacer3, new com.intellij.uiDesigner.core.GridConstraints(0, 1, 1, 6, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_CENTER, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        labelIcon = new JLabel();
        labelIcon.setText(" ");
        panel.add(labelIcon, new com.intellij.uiDesigner.core.GridConstraints(1, 1, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelPluginName = new JLabel();
        labelPluginName.setText("  ");
        panel.add(labelPluginName, new com.intellij.uiDesigner.core.GridConstraints(1, 2, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelOfficialSite = new JLabel();
        labelOfficialSite.setText("  ");
        panel.add(labelOfficialSite, new com.intellij.uiDesigner.core.GridConstraints(2, 1, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelAuthor = new JLabel();
        labelAuthor.setText("  ");
        panel.add(labelAuthor, new com.intellij.uiDesigner.core.GridConstraints(2, 6, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonInstall = new JButton();
        buttonInstall.setText("Install");
        panel.add(buttonInstall, new com.intellij.uiDesigner.core.GridConstraints(1, 6, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_CENTER, com.intellij.uiDesigner.core.GridConstraints.FILL_HORIZONTAL, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        scrollpanePlugins = new JScrollPane();
        panel.add(scrollpanePlugins, new com.intellij.uiDesigner.core.GridConstraints(1, 0, 4, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_CENTER, com.intellij.uiDesigner.core.GridConstraints.FILL_BOTH, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_CAN_SHRINK | com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        listPlugins = new JList();
        scrollpanePlugins.setViewportView(listPlugins);
        placeholder0 = new JLabel();
        placeholder0.setText("     ");
        panel.add(placeholder0, new com.intellij.uiDesigner.core.GridConstraints(1, 3, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholder1 = new JLabel();
        placeholder1.setText("     ");
        panel.add(placeholder1, new com.intellij.uiDesigner.core.GridConstraints(1, 4, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholder2 = new JLabel();
        placeholder2.setText("     ");
        panel.add(placeholder2, new com.intellij.uiDesigner.core.GridConstraints(1, 5, 1, 1, com.intellij.uiDesigner.core.GridConstraints.ANCHOR_WEST, com.intellij.uiDesigner.core.GridConstraints.FILL_NONE, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, com.intellij.uiDesigner.core.GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
    }

    /**
     * @noinspection ALL
     */
    public JComponent $$$getRootComponent$$$() {
        return panel;
    }
}
