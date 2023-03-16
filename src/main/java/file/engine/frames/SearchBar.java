package file.engine.frames;


import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.ConfigEntity;
import file.engine.configs.Constants;
import file.engine.dllInterface.EmptyRecycleBin;
import file.engine.dllInterface.GetHandle;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BootSystemEvent;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.database.*;
import file.engine.event.handler.impl.frame.searchBar.*;
import file.engine.event.handler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import file.engine.event.handler.impl.open.file.OpenFileEvent;
import file.engine.event.handler.impl.plugin.GetPluginByIdentifierEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.frames.components.LoadingPanel;
import file.engine.frames.components.MouseDragInfo;
import file.engine.frames.components.RoundBorder;
import file.engine.services.DatabaseService;
import file.engine.services.TranslateService;
import file.engine.services.plugin.system.Plugin;
import file.engine.services.plugin.system.PluginService;
import file.engine.utils.*;
import file.engine.utils.file.FileUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.SneakyThrows;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.Transferable;
import java.awt.event.*;
import java.awt.geom.RoundRectangle2D;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;


@SuppressWarnings({"IndexOfReplaceableByContains", "ListIndexOfReplaceableByContains"})
public class SearchBar {
    private static final String RESULT_LABEL_NAME_HOLDER = "filled";
    private final AtomicBoolean isLockMouseMotion = new AtomicBoolean(false);
    private final AtomicBoolean isOpenLastFolderPressed = new AtomicBoolean(false);
    private final AtomicBoolean isRunAsAdminPressed = new AtomicBoolean(false);
    private final AtomicBoolean isCopyPathPressed = new AtomicBoolean(false);
    private final AtomicBoolean isUserPressed = new AtomicBoolean(false);
    private final AtomicBoolean isMouseDraggedInWindow = new AtomicBoolean(false);
    private final AtomicBoolean startSearchSignal = new AtomicBoolean(false); // 同时修改
    private final AtomicBoolean isSearchNotStarted = new AtomicBoolean(true); // 同时修改
    private final AtomicBoolean isCudaSearchNotStarted = new AtomicBoolean(true); // 同时修改
    private final AtomicBoolean isBorderThreadNotExist = new AtomicBoolean(true);
    private final AtomicBoolean isLockMouseMotionThreadNotExist = new AtomicBoolean(true);
    private final AtomicBoolean isTryToShowResultThreadNotExist = new AtomicBoolean(true);
    private final AtomicBoolean isMergeThreadExist = new AtomicBoolean();
    private final AtomicBoolean isRoundRadiusSet = new AtomicBoolean();
    private final AtomicBoolean isPreviewMode = new AtomicBoolean();
    private final AtomicBoolean isTutorialMode = new AtomicBoolean();
    private final AtomicBoolean isSwitchToNormalManual = new AtomicBoolean();
    private Border fullBorder;
    private Border topBorder;
    private Border middleBorder;
    private Border bottomBorder;
    private Border explorerFullBorder;
    private Border explorerTopBorder;
    private Border explorerMiddleBorder;
    private Border explorerBottomBorder;
    private Border pluginFullBorder;
    private Border pluginTopBorder;
    private Border pluginMiddleBorder;
    private Border pluginBottomBorder;
    private final JFrame searchBar = new JFrame();
    private final JPanel searchBarContentPane = new JPanel();
    private final JLabel label1 = new JLabel();
    private final JLabel label2 = new JLabel();
    private final JLabel label3 = new JLabel();
    private final JLabel label4 = new JLabel();
    private final JLabel label5 = new JLabel();
    private final JLabel label6 = new JLabel();
    private final JLabel label7 = new JLabel();
    private final JLabel label8 = new JLabel();
    private final ConcurrentHashMap<JLabel, String> labelShowingPathInfo = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<JLabel, String> labelLastShowingPathInfo = new ConcurrentHashMap<>();
    private JLabel searchInfoLabel;
    private final AtomicInteger currentResultCount;  //保存当前选中的结果是在listResults中的第几个 范围 0 - listResults.size()
    private JTextField textField;
    private Color searchBarColor;
    private Color labelColor;
    private Color backgroundColor;
    private Color fontColorWithCoverage;
    private Color labelFontColor;
    private volatile long startTime = 0;
    private volatile RunningMode runningMode;
    private volatile Constants.Enums.ShowingSearchBarMode showingMode;
    private long mouseWheelTime = 0;
    private int iconSideLength;
    private volatile long visibleStartTime = 0;  //记录窗口开始可见的事件，窗口默认最短可见时间0.5秒，防止窗口快速闪烁
    private volatile long firstResultStartShowingTime = 0;  //记录开始显示结果的时间，用于防止刚开始移动到鼠标导致误触
    private volatile ArrayList<String> listResults = new ArrayList<>();  //保存从数据库中找出符合条件的记录（文件路径）
    private volatile String[] searchCase;
    private volatile String searchText = "";
    private volatile String[] keywords;
    private final AtomicInteger currentLabelSelectedPosition;   //保存当前是哪个label被选中 范围 0 - 7
    private volatile Plugin currentUsingPlugin;
    private volatile String currentPluginIdentifier;
    private final JPopupMenu menu = new JPopupMenu();
    private final JMenuItem open;
    private final JMenuItem openAsAdmin;
    private final JMenuItem copyDir;
    private final JMenuItem openLast;
    private int lastMousePositionX = 0;
    private int lastMousePositionY = 0;
    private static final String PLUGIN_RESULT_SPLITTER_STR = "-@-@-";
    private static final float SEARCH_BAR_WIDTH_RATIO = 0.3f;
    private static final float SEARCH_BAR_HEIGHT_RATIO = 0.4f;
    private static final float TEXT_FIELD_HEIGHT_RATIO = 0.7f;

    private static volatile SearchBar instance = null;

    private SearchBar() {
        currentResultCount = new AtomicInteger(0);
        TranslateService translateService = TranslateService.getInstance();
        open = new JMenuItem(translateService.getTranslation("Open"));
        openAsAdmin = new JMenuItem(translateService.getTranslation("Open as administrator"));
        copyDir = new JMenuItem(translateService.getTranslation("Copy file path"));
        openLast = new JMenuItem(translateService.getTranslation("Open parent folder"));
        menu.add(open);
        menu.add(openAsAdmin);
        menu.add(copyDir);
        menu.add(openLast);

        runningMode = RunningMode.NORMAL_MODE;
        showingMode = Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING;
        currentLabelSelectedPosition = new AtomicInteger(0);

        initGUI();
        initMenuItems();
        //添加textField搜索变更检测
        addTextFieldDocumentListener();
        //添加结果的鼠标事件响应
        addSearchBarMouseListener();
        //添加结果的鼠标滚轮响应
        addSearchBarMouseWheelListener();
        //添加结果的鼠标移动事件响应
        addSearchBarMouseMotionListener();
        //添加textField对键盘的响应
        addTextFieldKeyListener();
        addTextFieldFocusListener();
    }

    /**
     * 初始化窗口
     *
     * @param positionX       X坐标
     * @param positionY       Y坐标
     * @param searchBarWidth  宽度
     * @param searchBarHeight 高度
     */
    private void initFrame(int positionX, int positionY, int searchBarWidth, int searchBarHeight) {
        //frame
        searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
        searchBar.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        searchBar.setUndecorated(true);
        searchBar.getRootPane().setWindowDecorationStyle(JRootPane.NONE);
        searchBar.setBackground(new Color(0, 0, 0, 0));
        searchBar.setOpacity(AllConfigs.getInstance().getConfigEntity().getTransparency());
        searchBar.setContentPane(searchBarContentPane);
        searchBar.setType(JFrame.Type.UTILITY);
        searchBar.setAlwaysOnTop(true);
        //用于C++判断是否点击了当前窗口
        searchBar.setTitle("File-Engine-SearchBar");
    }

    private void initGUI() {
        var screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
        int width = screenSize.width;
        int height = screenSize.height;
        int searchBarWidth = (int) (width * SEARCH_BAR_WIDTH_RATIO);
        int searchBarHeight = (int) (height * SEARCH_BAR_HEIGHT_RATIO);
        final int positionX = width / 2 - searchBarWidth / 2;
        final int positionY = height / 2 - searchBarHeight / 3;
        final Color transparentColor = new Color(0, 0, 0, 0);
        AllConfigs allConfigs = AllConfigs.getInstance();
        var configs = allConfigs.getConfigEntity();
        searchBarColor = new Color(configs.getSearchBarColor());
        labelColor = new Color(configs.getLabelColor());
        fontColorWithCoverage = new Color(configs.getFontColorWithCoverage());
        backgroundColor = new Color(configs.getDefaultBackgroundColor());
        labelFontColor = new Color(configs.getFontColor());
        initBorder(allConfigs.getBorderType(), new Color(configs.getBorderColor()), configs.getBorderThickness());

        initFrame(positionX, positionY, searchBarWidth, searchBarHeight);

        int labelHeight = searchBarHeight / 9;
        int textFieldHeight = (int) (labelHeight * TEXT_FIELD_HEIGHT_RATIO);
        //TextField
        textField = new JTextField(1000);
        textField.setSize(searchBarWidth, textFieldHeight);
        Font textFieldFont = new Font(Font.SANS_SERIF, Font.PLAIN, getTextFieldFontSizeByTextFieldHeight(textFieldHeight));
        textField.setFont(textFieldFont);
        textField.setForeground(Color.BLACK);
        textField.setHorizontalAlignment(JTextField.LEFT);
        textField.setBackground(Color.WHITE);
        textField.setLocation(0, 0);
        textField.setOpaque(true);

        //labels
        final Font labelFont = new Font(Font.SANS_SERIF, Font.BOLD, getLabelFontSizeBySearchBarHeight(searchBarHeight));
        searchInfoLabel = new JLabel();
        searchInfoLabel.setHorizontalAlignment(SwingConstants.CENTER);

        int searchInfoHeight = labelHeight - textFieldHeight;
        initLabel(labelFont, searchBarWidth, searchInfoHeight, textFieldHeight, searchInfoLabel);
        initLabel(labelFont, searchBarWidth, labelHeight, labelHeight, label1);
        initLabel(labelFont, searchBarWidth, labelHeight, labelHeight * 2, label2);
        initLabel(labelFont, searchBarWidth, labelHeight, labelHeight * 3, label3);
        initLabel(labelFont, searchBarWidth, labelHeight, labelHeight * 4, label4);
        initLabel(labelFont, searchBarWidth, labelHeight, labelHeight * 5, label5);
        initLabel(labelFont, searchBarWidth, labelHeight, labelHeight * 6, label6);
        initLabel(labelFont, searchBarWidth, labelHeight, labelHeight * 7, label7);
        initLabel(labelFont, searchBarWidth, labelHeight, labelHeight * 8, label8);

        iconSideLength = labelHeight / 3; //定义图标边长

        URL icon = this.getClass().getResource("/icons/taskbar_32x32.png");
        if (icon != null) {
            Image image = new ImageIcon(icon).getImage();
            searchBar.setIconImage(image);
        }
        //panel
        searchBarContentPane.setDoubleBuffered(true);
        searchBarContentPane.setBounds(0, 0, searchBarWidth, searchBarHeight);
        searchBarContentPane.setLayout(null);
        searchBarContentPane.setBackground(transparentColor);
        searchBarContentPane.add(textField);
        searchBarContentPane.add(searchInfoLabel);
        searchBarContentPane.add(label1);
        searchBarContentPane.add(label2);
        searchBarContentPane.add(label3);
        searchBarContentPane.add(label4);
        searchBarContentPane.add(label5);
        searchBarContentPane.add(label6);
        searchBarContentPane.add(label7);
        searchBarContentPane.add(label8);
    }

    /**
     * 插件返回的结果为plugin[pluginResultSplitStr]pluginIdentifier[pluginResultSplitStr]result
     *
     * @param res 结果
     * @return 以pluginResultSplitStr分开的字符串
     */
    private String[] splitPluginResult(String res) {
        String[] ret = new String[3];
        String[] resultWithPluginInfo = RegexUtil.getPattern(PLUGIN_RESULT_SPLITTER_STR, 0).split(res);
        StringBuilder pluginResultBuilder = new StringBuilder();
        for (int i = 2; i < resultWithPluginInfo.length; i++) {
            pluginResultBuilder.append(resultWithPluginInfo[2]).append(PLUGIN_RESULT_SPLITTER_STR);
        }
        String pluginResult = "";
        if (!pluginResultBuilder.isEmpty()) {
            pluginResult = pluginResultBuilder.substring(0, pluginResultBuilder.length() - PLUGIN_RESULT_SPLITTER_STR.length());
        }
        ret[0] = resultWithPluginInfo[0];
        ret[1] = resultWithPluginInfo[1];
        ret[2] = pluginResult;
        return ret;
    }

    /**
     * 添加对右键菜单的相应
     */
    private void initMenuItems() {
        open.addActionListener(e -> {
            if (isPreviewMode.get() || isTutorialMode.get()) {
                return;
            }
            if (!listResults.isEmpty()) {
                String res = listResults.get(currentResultCount.get());
                if (runningMode == RunningMode.NORMAL_MODE) {
                    if (res.startsWith("plugin")) {
                        String[] split = splitPluginResult(res);
                        res = split[2];
                    }
                    openWithoutAdmin(res);
                } else {
                    String[] commandInfo = RegexUtil.semicolon.split(res);
                    boolean isExecuted = runInternalCommand(RegexUtil.colon.split(commandInfo[0])[1]);
                    if (isExecuted) {
                        return;
                    }
                    File open = new File(commandInfo[1]);
                    openWithoutAdmin(open.getAbsolutePath());
                }
            }
            detectShowingModeAndClose();
        });

        openAsAdmin.addActionListener(e -> {
            if (isPreviewMode.get() || isTutorialMode.get()) {
                return;
            }
            if (!listResults.isEmpty()) {
                String res = listResults.get(currentResultCount.get());
                if (runningMode == RunningMode.NORMAL_MODE) {
                    if (res.startsWith("plugin")) {
                        String[] split = splitPluginResult(res);
                        res = split[2];
                    }
                    openWithAdmin(res);
                } else {
                    String[] commandInfo = RegexUtil.semicolon.split(res);
                    boolean isExecuted = runInternalCommand(RegexUtil.colon.split(commandInfo[0])[1]);
                    if (isExecuted) {
                        return;
                    }
                    File open = new File(commandInfo[1]);
                    openWithAdmin(open.getAbsolutePath());
                }
            }
            detectShowingModeAndClose();
        });

        copyDir.addActionListener(e -> {
            if (isPreviewMode.get() || isTutorialMode.get()) {
                return;
            }
            if (!listResults.isEmpty()) {
                String res = listResults.get(currentResultCount.get());
                if (runningMode == RunningMode.NORMAL_MODE) {
                    if (res.startsWith("plugin")) {
                        String[] split = splitPluginResult(res);
                        res = split[2];
                    }
                    copyToClipBoard(res, true);
                } else {
                    String[] commandInfo = RegexUtil.semicolon.split(res);
                    boolean isExecuted = runInternalCommand(RegexUtil.colon.split(commandInfo[0])[1]);
                    if (isExecuted) {
                        return;
                    }
                    File open = new File(commandInfo[1]);
                    copyToClipBoard(open.getAbsolutePath(), true);
                }
            }
            detectShowingModeAndClose();
        });

        openLast.addActionListener(e -> {
            if (isPreviewMode.get() || isTutorialMode.get()) {
                return;
            }
            if (!listResults.isEmpty()) {
                String res = listResults.get(currentResultCount.get());
                if (runningMode == RunningMode.NORMAL_MODE) {
                    if (res.startsWith("plugin")) {
                        String[] split = splitPluginResult(res);
                        res = split[2];
                    }
                    openFolderByExplorer(res);
                } else {
                    String[] commandInfo = RegexUtil.semicolon.split(res);
                    boolean isExecuted = runInternalCommand(RegexUtil.colon.split(commandInfo[0])[1]);
                    if (isExecuted) {
                        return;
                    }
                    File open = new File(commandInfo[1]);
                    openFolderByExplorer(open.getAbsolutePath());
                }
            }
            detectShowingModeAndClose();
        });
    }

    /**
     * 初始化所有边框
     *
     * @param borderType      边框类型
     * @param borderColor     边框颜色
     * @param borderThickness 边框厚度
     */
    private void initBorder(Constants.Enums.BorderType borderType, Color borderColor, float borderThickness) {
        double roundRadius = AllConfigs.getInstance().getConfigEntity().getRoundRadius();
        int borderThickNessInt = (int) borderThickness;
        Color explorerAttachBorderColor;
        if (ColorUtil.isDark(searchBarColor.getRGB())) {
            explorerAttachBorderColor = Color.WHITE;
        } else {
            explorerAttachBorderColor = Color.BLACK;
        }
        RoundBorder topRound = new RoundBorder(borderColor,
                borderThickness,
                (int) roundRadius,
                RoundBorder.RoundedCorners.TOP_LEFT | RoundBorder.RoundedCorners.TOP_RIGHT,
                RoundBorder.ShowLines.TOP | RoundBorder.ShowLines.LEFT | RoundBorder.ShowLines.RIGHT);
        RoundBorder explorerTopRound = new RoundBorder(explorerAttachBorderColor,
                borderThickness,
                (int) roundRadius,
                RoundBorder.RoundedCorners.TOP_LEFT | RoundBorder.RoundedCorners.TOP_RIGHT,
                RoundBorder.ShowLines.TOP | RoundBorder.ShowLines.LEFT | RoundBorder.ShowLines.RIGHT);
        RoundBorder bottomRound = new RoundBorder(borderColor,
                borderThickness,
                (int) roundRadius,
                RoundBorder.RoundedCorners.BOTTOM_LEFT | RoundBorder.RoundedCorners.BOTTOM_RIGHT,
                RoundBorder.ShowLines.BOTTOM | RoundBorder.ShowLines.LEFT | RoundBorder.ShowLines.RIGHT);
        RoundBorder explorerBottomRound = new RoundBorder(explorerAttachBorderColor,
                borderThickness,
                (int) roundRadius,
                RoundBorder.RoundedCorners.BOTTOM_LEFT | RoundBorder.RoundedCorners.BOTTOM_RIGHT,
                RoundBorder.ShowLines.BOTTOM | RoundBorder.ShowLines.LEFT | RoundBorder.ShowLines.RIGHT);
        RoundBorder fullRound = new RoundBorder(borderColor,
                borderThickness,
                (int) roundRadius,
                RoundBorder.RoundedCorners.ALL,
                RoundBorder.ShowLines.ALL);
        RoundBorder explorerFullRound = new RoundBorder(explorerAttachBorderColor,
                borderThickness,
                (int) roundRadius,
                RoundBorder.RoundedCorners.ALL,
                RoundBorder.ShowLines.ALL);
        if (Constants.Enums.BorderType.AROUND == borderType) {
            topBorder = topRound;
            middleBorder = BorderFactory.createMatteBorder(0, borderThickNessInt, 0, borderThickNessInt, borderColor);
            bottomBorder = bottomRound;
            fullBorder = fullRound;
            explorerTopBorder = explorerTopRound;
            explorerMiddleBorder = BorderFactory.createMatteBorder(0, borderThickNessInt, 0, borderThickNessInt, explorerAttachBorderColor);
            explorerBottomBorder = explorerBottomRound;
            explorerFullBorder = explorerFullRound;
        } else if (Constants.Enums.BorderType.EMPTY == borderType) {
            Border emptyBorder = BorderFactory.createEmptyBorder();
            topBorder = emptyBorder;
            middleBorder = emptyBorder;
            bottomBorder = emptyBorder;
            fullBorder = emptyBorder;
            explorerTopBorder = emptyBorder;
            explorerMiddleBorder = emptyBorder;
            explorerBottomBorder = emptyBorder;
            explorerFullBorder = emptyBorder;
        } else if (borderType == Constants.Enums.BorderType.FULL) {
            Border lineBorder = BorderFactory.createMatteBorder(
                    0,
                    borderThickNessInt,
                    borderThickNessInt,
                    borderThickNessInt,
                    borderColor);
            topBorder = topRound;
            middleBorder = lineBorder;
            bottomBorder = bottomRound;
            fullBorder = fullRound;
            lineBorder = BorderFactory.createMatteBorder(
                    0,
                    borderThickNessInt,
                    borderThickNessInt,
                    borderThickNessInt,
                    explorerAttachBorderColor);
            explorerTopBorder = explorerTopRound;
            explorerMiddleBorder = lineBorder;
            explorerBottomBorder = explorerBottomRound;
            explorerFullBorder = explorerFullRound;
        }
        Color highContrast = ColorUtil.generateHighContrastColor(searchBarColor);
        pluginTopBorder = new RoundBorder(highContrast,
                2,
                (int) roundRadius,
                RoundBorder.RoundedCorners.TOP_LEFT | RoundBorder.RoundedCorners.TOP_RIGHT,
                RoundBorder.ShowLines.TOP | RoundBorder.ShowLines.LEFT | RoundBorder.ShowLines.RIGHT);
        pluginBottomBorder = new RoundBorder(highContrast,
                2,
                (int) roundRadius,
                RoundBorder.RoundedCorners.BOTTOM_LEFT | RoundBorder.RoundedCorners.BOTTOM_RIGHT,
                RoundBorder.ShowLines.BOTTOM | RoundBorder.ShowLines.LEFT | RoundBorder.ShowLines.RIGHT);
        pluginFullBorder = new RoundBorder(highContrast,
                2,
                (int) roundRadius,
                RoundBorder.RoundedCorners.ALL,
                RoundBorder.ShowLines.ALL);
        if (borderType == Constants.Enums.BorderType.FULL) {
            pluginMiddleBorder = BorderFactory.createMatteBorder(0, 2, 2, 2, highContrast);
        } else {
            pluginMiddleBorder = BorderFactory.createMatteBorder(0, 2, 0, 2, highContrast);
        }
    }

    private static SearchBar getInstance() {
        if (instance == null) {
            synchronized (SearchBar.class) {
                if (instance == null) {
                    instance = new SearchBar();
                }
            }
        }
        return instance;
    }

    private void addTextFieldFocusListener() {
        AllConfigs allConfigs = AllConfigs.getInstance();
        textField.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent e) {
                if (!IsDebug.isDebug()) {
                    resetAllStatus();
                }
            }

            @Override
            public void focusLost(FocusEvent e) {
                if (System.currentTimeMillis() - visibleStartTime > Constants.MIN_FRAME_VISIBLE_TIME) {
                    if (menu.isVisible()) {
                        return;
                    }
                    if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING && allConfigs.getConfigEntity().isLoseFocusClose()) {
                        if (!isTutorialMode.get()) {
                            closeSearchBar();
                        }
                    } else if (showingMode == Constants.Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                        closeWithoutHideSearchBar();
                    }
                }
            }
        });
    }

    /**
     * 初始化label
     *
     * @param font      字体
     * @param width     宽
     * @param height    高
     * @param positionY Y坐标值
     * @param label     需要初始化的label
     */
    private void initLabel(Font font, int width, int height, int positionY, JLabel label) {
        label.setSize(width, height);
        label.setLocation(0, positionY);
        label.setFont(font);
        label.setForeground(labelFontColor);
        label.setOpaque(true);
        label.setBackground(null);
        label.setFocusable(false);
    }

    /**
     * 用于模式切换时实时修改label大小
     *
     * @param width  宽
     * @param height 高
     * @param label  需要修改大小的label
     */
    private void setLabelSize(int width, int height, JLabel label) {
        label.setSize(width, height);
    }


    /**
     * 让搜索窗口响应鼠标双击事件以打开文件
     */
    private void addSearchBarMouseListener() {
        searchBar.addMouseListener(new MouseAdapter() {
            private final AllConfigs allConfigs = AllConfigs.getInstance();

            @Override
            public void mousePressed(MouseEvent e) {
                if (listResults.isEmpty()) {
                    return;
                }
                int count = e.getClickCount();
                String res = listResults.get(currentResultCount.get());
                // 如果点击的是插件显示的结果，则发送鼠标事件到对应插件
                if (res.startsWith("plugin")) {
                    String[] split = splitPluginResult(res);
                    GetPluginByIdentifierEvent getPluginByIdentifierEvent = new GetPluginByIdentifierEvent(split[1]);
                    EventManagement eventManagement = EventManagement.getInstance();
                    eventManagement.putEvent(getPluginByIdentifierEvent);
                    eventManagement.waitForEvent(getPluginByIdentifierEvent);
                    Optional<PluginService.PluginInfo> pluginInfoOptional = getPluginByIdentifierEvent.getReturnValue();
                    pluginInfoOptional.ifPresent((pluginInfo) -> pluginInfo.plugin.mousePressed(e, split[2]));
                    if (count == 2) {
                        detectShowingModeAndClose();
                    }
                } else {
                    if (count == 2) {
                        if (isPreviewMode.get() || isTutorialMode.get()) {
                            return;
                        }
                        if (runningMode == RunningMode.NORMAL_MODE) {
                            String searchBarText = getSearchBarText();
                            if (searchBarText.charAt(0) == '>') {
                                SwingUtilities.invokeLater(() -> textField.setText(">" + res + " "));
                                return;
                            } else {
                                if (res.startsWith("plugin")) {
                                    return;
                                }
                                if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                                    if (isOpenLastFolderPressed.get()) {
                                        //打开上级文件夹
                                        openFolderByExplorer(res);
                                    } else if (allConfigs.getConfigEntity().isDefaultAdmin() || isRunAsAdminPressed.get()) {
                                        openWithAdmin(res);
                                    } else if (isCopyPathPressed.get()) {
                                        copyToClipBoard(res, true);
                                    } else {
                                        openWithoutAdmin(res);
                                    }
                                }
                            }
                        } else if (runningMode == RunningMode.COMMAND_MODE) {
                            String[] commandInfo = RegexUtil.semicolon.split(res);
                            boolean isExecuted = runInternalCommand(RegexUtil.colon.split(commandInfo[0])[1]);
                            if (isExecuted) {
                                return;
                            }
                            File open = new File(commandInfo[1]);
                            if (isOpenLastFolderPressed.get()) {
                                //打开上级文件夹
                                openFolderByExplorer(open.getAbsolutePath());
                            } else if (allConfigs.getConfigEntity().isDefaultAdmin() || isRunAsAdminPressed.get()) {
                                openWithAdmin(open.getAbsolutePath());
                            } else if (isCopyPathPressed.get()) {
                                copyToClipBoard(open.getAbsolutePath(), true);
                            } else {
                                openWithoutAdmin(open.getAbsolutePath());
                            }
                        }
                        detectShowingModeAndClose();
                    }
                }

                if (runningMode == RunningMode.PLUGIN_MODE &&
                        showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING &&
                        currentUsingPlugin != null) {
                    if (!listResults.isEmpty()) {
                        String[] split = splitPluginResult(listResults.get(currentResultCount.get()));
                        currentUsingPlugin.mousePressed(e, split[2]);
                        if (count == 2) {
                            detectShowingModeAndClose();
                        }
                    }
                }

                if (e.getButton() == MouseEvent.BUTTON3 && runningMode != RunningMode.PLUGIN_MODE) {
                    //右键被点击
                    Point mouseLocation = MouseInfo.getPointerInfo().getLocation();
                    menu.show(null, (int) (mouseLocation.getX()), (int) (mouseLocation.getY()));
                } else if (e.getButton() == MouseEvent.BUTTON1) {
                    menu.setVisible(false);
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                if (listResults.isEmpty()) {
                    return;
                }
                String res = listResults.get(currentResultCount.get());
                if (res.startsWith("plugin")) {
                    String[] split = splitPluginResult(res);
                    GetPluginByIdentifierEvent getPluginByIdentifierEvent = new GetPluginByIdentifierEvent(split[1]);
                    EventManagement eventManagement = EventManagement.getInstance();
                    eventManagement.putEvent(getPluginByIdentifierEvent);
                    eventManagement.waitForEvent(getPluginByIdentifierEvent);

                    Optional<PluginService.PluginInfo> pluginInfoOptional =
                            getPluginByIdentifierEvent.getReturnValue();
                    pluginInfoOptional.ifPresent((pluginInfo -> pluginInfo.plugin.mouseReleased(e, split[2])));
                }
                if (runningMode == RunningMode.PLUGIN_MODE && currentUsingPlugin != null) {
                    String[] split = splitPluginResult(res);
                    res = split[2];
                    currentUsingPlugin.mouseReleased(e, res);
                }
            }
        });
    }


    //在explorer attach模式时操作鼠标和键盘以快速跳转到文件位置
    private void quickJump(String result) {
        if (FileUtil.isFile(result)) {
            result = FileUtil.getParentPath(result);
        }
        saveCache(result);
        GetHandle.INSTANCE.setEditPath(new String(result.getBytes(StandardCharsets.UTF_8), StandardCharsets.UTF_8));
    }

    /**
     * 复制信息到系统剪贴板
     *
     * @param res 需要复制的信息
     */
    private void copyToClipBoard(String res, @SuppressWarnings("SameParameterValue") boolean isNotifyUser) {
        Transferable trans = new StringSelection(res);
        copyToClipBoard(trans, isNotifyUser);
    }

    private void copyToClipBoard(Transferable data, boolean isNotifyUser) {
        TranslateService translateService = TranslateService.getInstance();
        Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
        clipboard.setContents(data, null);
        if (isNotifyUser) {
            EventManagement.getInstance().putEvent(new ShowTaskBarMessageEvent(
                    translateService.getTranslation("Info"),
                    translateService.getTranslation("The result has been copied to the clipboard")));
        }
    }

    private String getSearchBarText() {
        try {
            return textField.getText();
        } catch (Exception e) {
            return "";
        }
    }

    /**
     * 使窗口检测键盘事件，用于检测enter被点击时，打开文件，或打开文件所在位置，或复制文件路径，或以管理员方式打开
     */
    private void addTextFieldKeyListener() {
        textField.addKeyListener(new KeyListener() {
            private static final int timeLimit = 50;
            long pressTime;
            boolean isFirstPress = true;
            final AllConfigs allConfigs = AllConfigs.getInstance();

            @Override
            public void keyPressed(KeyEvent arg0) {
                int key = arg0.getKeyCode();
                if (key == KeyEvent.VK_BACK_SPACE && getSearchBarText().isEmpty()) {
                    //消除搜索框为空时按删除键发出的无效提示音
                    arg0.consume();
                    if (currentUsingPlugin != null) {
                        currentUsingPlugin = null;
                        String substring = ">" + currentPluginIdentifier.substring(0, currentPluginIdentifier.length() - 1);
                        SwingUtilities.invokeLater(() -> textField.setText(substring));
                    }
                }
                if (key == KeyEvent.VK_ESCAPE) {
                    closeSearchBar();
                    return;
                }
                if (listResults.isEmpty()) {
                    return;
                }
                var configs = allConfigs.getConfigEntity();
                if (38 == key) {
                    handleUpKey();
                } else if (40 == key) {
                    handleDownKey();
                } else if (10 == key) {
                    if (handleEnterKey(configs)) {
                        return;
                    }
                } else if (configs.getOpenLastFolderKeyCode() == key) {
                    //打开上级文件夹热键被点击
                    isOpenLastFolderPressed.set(true);
                } else if (configs.getRunAsAdminKeyCode() == key) {
                    //以管理员方式运行热键被点击
                    isRunAsAdminPressed.set(true);
                } else if (configs.getCopyPathKeyCode() == key) {
                    isCopyPathPressed.set(true);
                }

                if (!listResults.isEmpty()) {
                    if (key != 38 && key != 40) {
                        String res = listResults.get(currentResultCount.get());
                        if (runningMode == RunningMode.NORMAL_MODE) {
                            if (res.startsWith("plugin")) {
                                String[] split = splitPluginResult(res);
                                var getPluginByIdentifierEvent = new GetPluginByIdentifierEvent(split[1]);
                                var eventManagement = EventManagement.getInstance();
                                eventManagement.putEvent(getPluginByIdentifierEvent);
                                eventManagement.waitForEvent(getPluginByIdentifierEvent);
                                Optional<PluginService.PluginInfo> pluginInfoOptional = getPluginByIdentifierEvent.getReturnValue();
                                pluginInfoOptional.ifPresent((pluginInfo -> pluginInfo.plugin.keyPressed(arg0, split[2])));
                            }
                        } else if (runningMode == RunningMode.PLUGIN_MODE) {
                            if (currentUsingPlugin != null) {
                                String[] split = splitPluginResult(res);
                                currentUsingPlugin.keyPressed(arg0, split[2]);
                            }
                        }
                    }
                }
            }

            private boolean handleEnterKey(ConfigEntity configs) {
                //enter被点击
                if (isPreviewMode.get() || isTutorialMode.get()) {
                    return true;
                }
                String res = listResults.get(currentResultCount.get());
                if (runningMode == RunningMode.NORMAL_MODE) {
                    String searchBarText = getSearchBarText();
                    if (searchBarText.charAt(0) == '>') {
                        SwingUtilities.invokeLater(() -> textField.setText(">" + res + " "));
                        return true;
                    } else {
                        if (res.startsWith("plugin")) {
                            return true;
                        }
                        if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                            if (isOpenLastFolderPressed.get()) {
                                //打开上级文件夹
                                openFolderByExplorer(res);
                            } else if (configs.isDefaultAdmin() || isRunAsAdminPressed.get()) {
                                openWithAdmin(res);
                            } else if (isCopyPathPressed.get()) {
                                copyToClipBoard(res, true);
                            } else {
                                openWithoutAdmin(res);
                            }
                            detectShowingModeAndClose();
                        } else if (showingMode == Constants.Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                            if (isCopyPathPressed.get()) {
                                copyToClipBoard(res, true);
                            } else {
                                quickJump(res);
                            }
                        }
                    }
                } else if (runningMode == RunningMode.COMMAND_MODE) {
                    String[] commandInfo = RegexUtil.semicolon.split(res);
                    boolean isExecuted = runInternalCommand(RegexUtil.colon.split(commandInfo[0])[1]);
                    if (!isExecuted) {
                        File open = new File(commandInfo[1]);
                        if (isOpenLastFolderPressed.get()) {
                            //打开上级文件夹
                            openFolderByExplorer(open.getAbsolutePath());
                        } else if (configs.isDefaultAdmin() || isRunAsAdminPressed.get()) {
                            openWithAdmin(open.getAbsolutePath());
                        } else if (isCopyPathPressed.get()) {
                            copyToClipBoard(open.getAbsolutePath(), true);
                        } else {
                            openWithoutAdmin(open.getAbsolutePath());
                        }
                        detectShowingModeAndClose();
                    }
                }
                return false;
            }

            private void handleDownKey() {
                //下键被点击
                if (isFirstPress || System.currentTimeMillis() - pressTime > timeLimit) {
                    pressTime = System.currentTimeMillis();
                    isFirstPress = false;
                    if (isLabelNotEmpty(label1) && isLabelNotEmpty(label2) && isLabelNotEmpty(label3) && isLabelNotEmpty(label4)
                            && isLabelNotEmpty(label5) && isLabelNotEmpty(label6) && isLabelNotEmpty(label7) && isLabelNotEmpty(label8)) {
                        isUserPressed.set(true);
                    }
                    boolean isNextLabelValid = isNextLabelValid();
                    //当下一个label有数据时才移动到下一个
                    if (isNextLabelValid) {
                        if (!getSearchBarText().isEmpty()) {
                            currentResultCount.incrementAndGet();
                            int size = listResults.size();
                            if (currentResultCount.get() >= size) {
                                currentResultCount.set(size - 1);
                            }
                            if (currentResultCount.get() <= 0) {
                                currentResultCount.set(0);
                            }
                            moveDownward(getCurrentLabelPos());
                        }
                    }
                }
            }

            private void handleUpKey() {
                //上键被点击
                if (isFirstPress || System.currentTimeMillis() - pressTime > timeLimit) {
                    pressTime = System.currentTimeMillis();
                    isFirstPress = false;
                    if (isLabelNotEmpty(label1) && isLabelNotEmpty(label2) && isLabelNotEmpty(label3) && isLabelNotEmpty(label4)
                            && isLabelNotEmpty(label5) && isLabelNotEmpty(label6) && isLabelNotEmpty(label7) && isLabelNotEmpty(label8)) {
                        isUserPressed.set(true);
                    }

                    if (!getSearchBarText().isEmpty()) {
                        currentResultCount.decrementAndGet();
                        int size = listResults.size();
                        if (currentResultCount.get() >= size) {
                            currentResultCount.set(size - 1);
                        }
                        if (currentResultCount.get() <= 0) {
                            currentResultCount.set(0);
                        }
                        moveUpward(getCurrentLabelPos());
                    }
                }
            }

            @Override
            public void keyReleased(KeyEvent arg0) {
                int key = arg0.getKeyCode();
                if (listResults.isEmpty()) {
                    return;
                }
                var configs = allConfigs.getConfigEntity();

                if (key != 38 && key != 40) {
                    String res = listResults.get(currentResultCount.get());
                    if (runningMode == RunningMode.NORMAL_MODE) {
                        if (res.startsWith("plugin")) {
                            String[] split = splitPluginResult(res);
                            GetPluginByIdentifierEvent getPluginByIdentifierEvent = new GetPluginByIdentifierEvent(split[1]);
                            EventManagement eventManagement = EventManagement.getInstance();
                            eventManagement.putEvent(getPluginByIdentifierEvent);
                            eventManagement.waitForEvent(getPluginByIdentifierEvent);
                            Optional<PluginService.PluginInfo> pluginInfoOptional = getPluginByIdentifierEvent.getReturnValue();
                            pluginInfoOptional.ifPresent(pluginInfo -> pluginInfo.plugin.keyReleased(arg0, split[2]));
                        }
                    } else if (runningMode == RunningMode.PLUGIN_MODE) {
                        if (currentUsingPlugin != null) {
                            currentUsingPlugin.keyReleased(arg0, splitPluginResult(res)[2]);
                        }
                    }
                }

                if (configs.getOpenLastFolderKeyCode() == key) {
                    //复位按键状态
                    isOpenLastFolderPressed.set(false);
                } else if (configs.getRunAsAdminKeyCode() == key) {
                    isRunAsAdminPressed.set(false);
                } else if (configs.getCopyPathKeyCode() == key) {
                    isCopyPathPressed.set(false);
                }
            }

            @Override
            public void keyTyped(KeyEvent arg0) {
                int key = arg0.getKeyCode();
                if (listResults.isEmpty()) {
                    return;
                }

                if (key != 38 && key != 40) {
                    String res = listResults.get(currentResultCount.get());
                    if (runningMode == RunningMode.NORMAL_MODE) {
                        if (res.startsWith("plugin")) {
                            String[] split = splitPluginResult(res);
                            GetPluginByIdentifierEvent getPluginByIdentifierEvent = new GetPluginByIdentifierEvent(split[1]);
                            EventManagement eventManagement = EventManagement.getInstance();
                            eventManagement.putEvent(getPluginByIdentifierEvent);
                            eventManagement.waitForEvent(getPluginByIdentifierEvent);
                            Optional<PluginService.PluginInfo> pluginInfoOptional = getPluginByIdentifierEvent.getReturnValue();
                            pluginInfoOptional.ifPresent(pluginInfo -> pluginInfo.plugin.keyTyped(arg0, split[2]));
                        }
                    } else if (runningMode == RunningMode.PLUGIN_MODE) {
                        if (currentUsingPlugin != null) {
                            currentUsingPlugin.keyTyped(arg0, splitPluginResult(res)[2]);
                        }
                    }
                }
            }
        });
    }

    private void openFolderByExplorer(String dir) {
        EventManagement.getInstance().putEvent(new OpenFileEvent(OpenFileEvent.OpenStatus.LAST_DIR, dir), event -> saveCache(dir), null);
    }

    /**
     * 在command模式下，检测当前输入信息是否是软件已经定义的内部命令
     * clearbin update help version
     * return true only the internal command was executed. Otherwise false
     */
    private boolean runInternalCommand(String commandName) {
        TranslateService translateService = TranslateService.getInstance();
        EventManagement eventManagement = EventManagement.getInstance();
        switch (commandName) {
            case "clearbin" -> {
                detectShowingModeAndClose();
//                if (JOptionPane.showConfirmDialog(null, translateService.getTranslation(
//                        "Are you sure you want to empty the recycle bin")) == JOptionPane.OK_OPTION) {
//                    try {
//                        File[] roots = File.listRoots();
//                        for (File root : roots) {
//                            Runtime.getRuntime().exec("cmd.exe /c rd /s /q " + root.getAbsolutePath() + "$Recycle.Bin");
//                        }
//                        JOptionPane.showMessageDialog(null, translateService.getTranslation(
//                                "Successfully empty the recycle bin"));
//                    } catch (IOException e) {
//                        JOptionPane.showMessageDialog(null, translateService.getTranslation(
//                                "Failed to empty the recycle bin"));
//                    }
//                }
                while (isVisible()) {
                    closeSearchBar();
                    try {
                        TimeUnit.MILLISECONDS.sleep(250);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                EmptyRecycleBin.INSTANCE.emptyRecycleBin();
                return true;
            }
            case "update" -> {
                detectShowingModeAndClose();
                eventManagement.putEvent(new ShowTaskBarMessageEvent(
                        translateService.getTranslation("Info"),
                        translateService.getTranslation("Updating file index")));
                eventManagement.putEvent(new UpdateDatabaseEvent(false),
                        event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(
                                TranslateService.getInstance().getTranslation("Info"),
                                TranslateService.getInstance().getTranslation("Search Done"))),
                        event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(
                                TranslateService.getInstance().getTranslation("Warning"),
                                TranslateService.getInstance().getTranslation("Search Failed"))));
                startSearchSignal.set(false);
                isSearchNotStarted.set(false);
                isCudaSearchNotStarted.set(false);
                return true;
            }
            case "clearUpdate" -> {
                detectShowingModeAndClose();
                eventManagement.putEvent(new ShowTaskBarMessageEvent(
                        translateService.getTranslation("Info"),
                        translateService.getTranslation("Updating file index")));
                eventManagement.putEvent(new UpdateDatabaseEvent(true),
                        event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(
                                TranslateService.getInstance().getTranslation("Info"),
                                TranslateService.getInstance().getTranslation("Search Done"))),
                        event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(
                                TranslateService.getInstance().getTranslation("Warning"),
                                TranslateService.getInstance().getTranslation("Search Failed"))));
                startSearchSignal.set(false);
                isSearchNotStarted.set(false);
                isCudaSearchNotStarted.set(false);
                return true;
            }
            case "help" -> {
                detectShowingModeAndClose();
                if (JOptionPane.showConfirmDialog(null, translateService.getTranslation("Whether to view help"))
                        == JOptionPane.OK_OPTION) {
                    isTutorialMode.set(true);
                    ThreadPoolUtil.getInstance().executeTask(() -> {
                        showTutorial();
                        isTutorialMode.set(false);
                    });
                }
                return true;
            }
            case "version" -> {
                detectShowingModeAndClose();
                JOptionPane.showMessageDialog(null, translateService.getTranslation(
                        "Current Version:") + Constants.version);
                return true;
            }
            default -> {
                return false;
            }
        }
    }

    private void showTutorial() {
        if (isPreviewMode.get()) {
            return;
        }
        final int maxWaiting = 60_000;
        final long startWait = System.currentTimeMillis();
        AtomicBoolean isCanceled = new AtomicBoolean(false);
        TranslateService translateService = TranslateService.getInstance();
        DatabaseService databaseService = DatabaseService.getInstance();
        //检查数据库是否正常
        if (databaseService.getStatus() != Constants.Enums.DatabaseStatus.NORMAL || AllConfigs.isFirstRun()) {
            closeSearchBar();
            JFrame frame = new JFrame();
            frame.setUndecorated(true);
            frame.getRootPane().setWindowDecorationStyle(JRootPane.FRAME);
            LoadingPanel glassPane = new LoadingPanel(translateService.getTranslation("Waiting for searching disks")
                    + ", "
                    + translateService.getTranslation("Please wait up to 60 seconds"));
            glassPane.setSize(600, 400);
            frame.setGlassPane(glassPane);
            glassPane.start();//开始动画加载效果
            frame.setSize(600, 400);
            frame.setLocationRelativeTo(null);
            frame.setResizable(false);
            frame.setVisible(true);
            frame.addWindowListener(new WindowAdapter() {
                @Override
                public void windowClosing(WindowEvent e) {
                    isCanceled.set(true);
                }
            });
            try {
                //二次检查并尝试等待
                while (System.currentTimeMillis() - startWait <= maxWaiting) {
                    if (databaseService.getStatus() == Constants.Enums.DatabaseStatus.NORMAL) {
                        break;
                    }
                    TimeUnit.MILLISECONDS.sleep(1);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                glassPane.stop();
                frame.setVisible(false);
            }
        }
        if (isCanceled.get()) {
            return;
        }
        if (System.currentTimeMillis() - startWait > maxWaiting) {
            JOptionPane.showMessageDialog(null, translateService.getTranslation("Waiting overtime"));
            return;
        }
        EventManagement eventManagement = EventManagement.getInstance();
        showSearchbar();
        JOptionPane.showMessageDialog(searchBar, translateService.getTranslation("Welcome to the tutorial of File-Engine") + "\n" +
                translateService.getTranslation("The default Ctrl + Alt + K calls out the search bar, which can be changed in the settings.") +
                translateService.getTranslation("You can enter the keywords you want to search here"));
        JOptionPane.showMessageDialog(searchBar, translateService.getTranslation("Let's see an example"));
        SwingUtilities.invokeLater(() -> textField.setText("test"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("When you enter \"test\" in the search bar") + ",\n" +
                        translateService.getTranslation("files with \"test\" in the name will be displayed below the search bar"));
        SwingUtilities.invokeLater(() -> textField.setText("test;file"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("If you know multiple keywords of a file") + "\n" +
                        translateService.getTranslation("(for example, the file name contains both \"file\" and \"test\")") + ",\n" +
                        translateService.getTranslation("you can separate them with \";\" (semicolon) to search together as keywords."));
        SwingUtilities.invokeLater(() -> textField.setText("/test"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("When entering \"/test\" in the search bar") + ", " +
                        translateService.getTranslation("the file containing \"test\" in the path will be displayed below the search bar"));
        SwingUtilities.invokeLater(() -> textField.setText(""));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("Add \"|\" + suffix after the keyword to achieve a more precise search") + "\n" +
                        translateService.getTranslation("The program has the following four suffixes") + "\n" +
                        "|d     |f     |full     |case" + "\n" +
                        translateService.getTranslation("not case sensitive"));
        SwingUtilities.invokeLater(() -> textField.setText("test|d"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("\"|d\" is the suffix for searching only folders"));
        SwingUtilities.invokeLater(() -> textField.setText("test|f"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("\"|f\" is the suffix to search only for files"));
        SwingUtilities.invokeLater(() -> textField.setText("test|full"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("\"|full\" means full word matching, but case insensitive"));
        SwingUtilities.invokeLater(() -> textField.setText("test|case"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("\"|case\" means case sensitive"));
        SwingUtilities.invokeLater(() -> textField.setText("test|d;full"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("You can also combine different suffixes to use") + "\n" +
                        translateService.getTranslation("you can separate them with \";\" (semicolon) to search together as keywords."));
        SwingUtilities.invokeLater(() -> textField.setText("test;/file|d;case"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("Different keywords are separated by \";\" (semicolon), suffix and keywords are separated by \"|\" (vertical bar)"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("You can drag any search result out to create a shortcut on the desktop or any folder"));
        //判断是否为中文
        if ("简体中文".equals(translateService.getLanguage())) {
            SwingUtilities.invokeLater(() -> textField.setText("pinyin"));
            JOptionPane.showMessageDialog(searchBar, "你可以使用拼音来代替汉字");
        }
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("Click \"Enter\" to open the file directly") + "\n" +
                        translateService.getTranslation("Click \"Ctrl + Enter\" to open the folder where the file is located") + "\n" +
                        translateService.getTranslation("Click \"Shift + Enter\" to open the file as an administrator (use with caution)") + "\n" +
                        translateService.getTranslation("Click \"Alt+ Enter\" to copy the file path") + "\n\n" +
                        translateService.getTranslation("You can modify these hotkeys in the settings"));
        SwingUtilities.invokeLater(() -> textField.setText(":"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("Enter \":\" (colon) at the front of the search box to enter the command mode") + "\n" +
                        translateService.getTranslation("There are built-in commands, you can also add custom commands in the settings"));
        JOptionPane.showMessageDialog(searchBar,
                translateService.getTranslation("If you find that some files cannot be searched, you can enter \":update\" in the search bar to rebuild the index."));
        closeSearchBar();
        eventManagement.putEvent(new ShowSettingsFrameEvent());
        JOptionPane.showMessageDialog(null,
                translateService.getTranslation("This is the settings window") + "\n" +
                        translateService.getTranslation("You can modify many settings here") + "\n" +
                        translateService.getTranslation("Including the color of the window, the hot key to call out the search box, the transparency of the window, custom commands and so on."));
        if (JOptionPane.showConfirmDialog(null,
                translateService.getTranslation("End of the tutorial") + "\n" +
                        translateService.getTranslation("You can enter \":help\" in the search bar at any time to enter the tutorial") + "\n" +
                        translateService.getTranslation("There are more detailed tutorials on the Github wiki. Would you like to check it out?"))
                == JOptionPane.OK_OPTION) {
            try {
                Desktop desktop;
                //打开wiki页面
                if (Desktop.isDesktopSupported()) {
                    desktop = Desktop.getDesktop();
                    desktop.browse(new URI("https://github.com/XUANXUQAQ/File-Engine/wiki/Usage"));
                }
            } catch (URISyntaxException | IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 设置当前label为选中状态
     *
     * @param label 需要设置的label
     */
    private void setLabelChosen(JLabel label) {
        if (label != null) {
            SwingUtilities.invokeLater(() -> {
                String name = label.getName();
                if (!((name == null) || name.isEmpty() || RESULT_LABEL_NAME_HOLDER.equals(name))) {
                    String currentText = label.getText();
                    //indexOf效率更高
                    if (currentText == null || currentText.indexOf(":\\") == -1) {
                        //当前显示的不是路径
                        label.setName(currentText);
                        label.setText(name);
                    }
                }
                label.setBackground(labelColor);
            });
        }
    }

    /**
     * 设置当前label为未选中
     *
     * @param label 需要设置的label
     */
    private void setLabelNotChosen(JLabel label) {
        if (label != null) {
            SwingUtilities.invokeLater(() -> {
                String name = label.getName();
                if (!(name == null || name.isEmpty() || RESULT_LABEL_NAME_HOLDER.equals(name))) {
                    String currentText = label.getText();
                    //indexOf效率更高
                    if (currentText == null || currentText.indexOf(":\\") != -1) {
                        //当前显示的不是名称
                        label.setName(currentText);
                        label.setText(name);
                    }
                }
                label.setBackground(backgroundColor);
            });
        }
    }

    /**
     * 检测鼠标在窗口的位置，并设置鼠标所在位置的label为选中
     */
    private void addSearchBarMouseMotionListener() {
        AtomicBoolean shouldSaveMousePos = new AtomicBoolean(false);
        final int minMouseMoveDistance = label1.getHeight() / 3;
        //添加一个线程不断更新鼠标保存时间
        ThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.notMainExit()) {
                    shouldSaveMousePos.set(true);
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        searchBar.addMouseMotionListener(new MouseAdapter() {
            double absoluteDistance;
            final AtomicBoolean isIconCreated = new AtomicBoolean(false);

            @Override
            public void mouseDragged(MouseEvent e) {
                RunningMode mode = runningMode;
                isMouseDraggedInWindow.set(
                        mode == RunningMode.NORMAL_MODE ||
                                mode == RunningMode.COMMAND_MODE
                );
                if (isMouseDraggedInWindow.get() && !isIconCreated.get() && !listResults.isEmpty()) {
                    isIconCreated.set(true);
                    String result = listResults.get(currentResultCount.get());
                    File f = null;
                    if (runningMode == RunningMode.NORMAL_MODE) {
                        //普通模式直接获取文件路径
                        f = new File(result);
                    } else if (runningMode == RunningMode.COMMAND_MODE) {
                        String[] commandInfo = RegexUtil.semicolon.split(result);
                        //获取命令后的文件路径
                        if (commandInfo == null || commandInfo.length <= 1) {
                            return;
                        }
                        f = new File(commandInfo[1]);
                    }
                    if (f != null) {
                        File finalF = f;
                        new MouseDragInfo().showDragInfo(f.getAbsolutePath(),
                                () -> MouseInfo.getPointerInfo().getLocation(),
                                () -> !GetHandle.INSTANCE.isKeyPressed(0x01),
                                () -> {
                                    //创建快捷方式
                                    try {
                                        String writePath = GetHandle.INSTANCE.getExplorerPath();
                                        if (writePath != null && !writePath.isEmpty()) {
                                            ShortcutCreateUtil.createShortCut(finalF.getAbsolutePath(),
                                                    writePath + File.separator + finalF.getName(),
                                                    AllConfigs.getInstance().getConfigEntity().isShowTipCreatingLnk());
                                        }
                                    } catch (Exception exception) {
                                        exception.printStackTrace();
                                    }
                                    isIconCreated.set(false);
                                    isMouseDraggedInWindow.set(false);
                                });
                    }
                }
            }

            @Override
            public void mouseMoved(MouseEvent e) {
                if (menu.isVisible()) {
                    return;
                }
                //判断鼠标位置
                int offset = label1.getHeight();
                int labelPosition = label1.getY();
                int labelPosition2 = labelPosition + offset;
                int labelPosition3 = labelPosition + offset * 2;
                int labelPosition4 = labelPosition + offset * 3;
                int labelPosition5 = labelPosition + offset * 4;
                int labelPosition6 = labelPosition + offset * 5;
                int labelPosition7 = labelPosition + offset * 6;
                int labelPosition8 = labelPosition + offset * 7;
                int labelPosition9 = labelPosition + offset * 8;

                //开始显示500ms后才开始响应鼠标移动事件
                if (System.currentTimeMillis() - firstResultStartShowingTime > 500 && firstResultStartShowingTime != 0) {
                    int currentX = e.getX();
                    int currentY = e.getY();
                    if (lastMousePositionX == 0 || lastMousePositionY == 0) {
                        lastMousePositionX = currentX;
                        lastMousePositionY = currentY;
                    }
                    //计算鼠标当前位置到上次位置的直线距离
                    absoluteDistance = Math.sqrt(Math.pow((currentX - lastMousePositionX), 2) + Math.pow((currentY - lastMousePositionY), 2));
                    if (shouldSaveMousePos.get()) {
                        //超过50毫秒，保存一次鼠标位置
                        shouldSaveMousePos.set(false);
                        lastMousePositionX = currentX;
                        lastMousePositionY = currentY;
                    }
                    //距离大于鼠标最小移动值
                    if (absoluteDistance > minMouseMoveDistance) {
                        //判定当前位置
                        if (!isLockMouseMotion.get()) {
                            int position = getCurrentLabelPos();
                            int mouseOnWhichLabel = 0;
                            if (labelPosition2 <= e.getY() && e.getY() < labelPosition3) {
                                mouseOnWhichLabel = 1;
                            } else if (labelPosition3 <= e.getY() && e.getY() < labelPosition4) {
                                mouseOnWhichLabel = 2;
                            } else if (labelPosition4 <= e.getY() && e.getY() < labelPosition5) {
                                mouseOnWhichLabel = 3;
                            } else if (labelPosition5 <= e.getY() && e.getY() < labelPosition6) {
                                mouseOnWhichLabel = 4;
                            } else if (labelPosition6 <= e.getY() && e.getY() < labelPosition7) {
                                mouseOnWhichLabel = 5;
                            } else if (labelPosition7 <= e.getY() && e.getY() < labelPosition8) {
                                mouseOnWhichLabel = 6;
                            } else if (labelPosition8 <= e.getY() && e.getY() < labelPosition9) {
                                mouseOnWhichLabel = 7;
                            }
                            if (mouseOnWhichLabel < listResults.size()) {
                                int ret;
                                if (position < mouseOnWhichLabel) {
                                    ret = mouseOnWhichLabel - position;
                                } else {
                                    ret = -(position - mouseOnWhichLabel);
                                }
                                currentResultCount.getAndAdd(ret);
                                currentLabelSelectedPosition.getAndAdd(ret);
                                switch (mouseOnWhichLabel) {
                                    case 0 -> {
                                        if (isLabelNotEmpty(label1)) {
                                            setLabelChosen(label1);
                                        }
                                        if (isLabelNotEmpty(label2)) {
                                            setLabelNotChosen(label2);
                                        }
                                        if (isLabelNotEmpty(label3)) {
                                            setLabelNotChosen(label3);
                                        }
                                        if (isLabelNotEmpty(label4)) {
                                            setLabelNotChosen(label4);
                                        }
                                        if (isLabelNotEmpty(label5)) {
                                            setLabelNotChosen(label5);
                                        }
                                        if (isLabelNotEmpty(label6)) {
                                            setLabelNotChosen(label6);
                                        }
                                        if (isLabelNotEmpty(label7)) {
                                            setLabelNotChosen(label7);
                                        }
                                        if (isLabelNotEmpty(label8)) {
                                            setLabelNotChosen(label8);
                                        }
                                    }
                                    case 1 -> {
                                        if (isLabelNotEmpty(label1)) {
                                            setLabelNotChosen(label1);
                                        }
                                        if (isLabelNotEmpty(label2)) {
                                            setLabelChosen(label2);
                                        }
                                        if (isLabelNotEmpty(label3)) {
                                            setLabelNotChosen(label3);
                                        }
                                        if (isLabelNotEmpty(label4)) {
                                            setLabelNotChosen(label4);
                                        }
                                        if (isLabelNotEmpty(label5)) {
                                            setLabelNotChosen(label5);
                                        }
                                        if (isLabelNotEmpty(label6)) {
                                            setLabelNotChosen(label6);
                                        }
                                        if (isLabelNotEmpty(label7)) {
                                            setLabelNotChosen(label7);
                                        }
                                        if (isLabelNotEmpty(label8)) {
                                            setLabelNotChosen(label8);
                                        }
                                    }
                                    case 2 -> {
                                        if (isLabelNotEmpty(label1)) {
                                            setLabelNotChosen(label1);
                                        }
                                        if (isLabelNotEmpty(label2)) {
                                            setLabelNotChosen(label2);
                                        }
                                        if (isLabelNotEmpty(label3)) {
                                            setLabelChosen(label3);
                                        }
                                        if (isLabelNotEmpty(label4)) {
                                            setLabelNotChosen(label4);
                                        }
                                        if (isLabelNotEmpty(label5)) {
                                            setLabelNotChosen(label5);
                                        }
                                        if (isLabelNotEmpty(label6)) {
                                            setLabelNotChosen(label6);
                                        }
                                        if (isLabelNotEmpty(label7)) {
                                            setLabelNotChosen(label7);
                                        }
                                        if (isLabelNotEmpty(label8)) {
                                            setLabelNotChosen(label8);
                                        }
                                    }
                                    case 3 -> {
                                        if (isLabelNotEmpty(label1)) {
                                            setLabelNotChosen(label1);
                                        }
                                        if (isLabelNotEmpty(label2)) {
                                            setLabelNotChosen(label2);
                                        }
                                        if (isLabelNotEmpty(label3)) {
                                            setLabelNotChosen(label3);
                                        }
                                        if (isLabelNotEmpty(label4)) {
                                            setLabelChosen(label4);
                                        }
                                        if (isLabelNotEmpty(label5)) {
                                            setLabelNotChosen(label5);
                                        }
                                        if (isLabelNotEmpty(label6)) {
                                            setLabelNotChosen(label6);
                                        }
                                        if (isLabelNotEmpty(label7)) {
                                            setLabelNotChosen(label7);
                                        }
                                        if (isLabelNotEmpty(label8)) {
                                            setLabelNotChosen(label8);
                                        }
                                    }
                                    case 4 -> {
                                        if (isLabelNotEmpty(label1)) {
                                            setLabelNotChosen(label1);
                                        }
                                        if (isLabelNotEmpty(label2)) {
                                            setLabelNotChosen(label2);
                                        }
                                        if (isLabelNotEmpty(label3)) {
                                            setLabelNotChosen(label3);
                                        }
                                        if (isLabelNotEmpty(label4)) {
                                            setLabelNotChosen(label4);
                                        }
                                        if (isLabelNotEmpty(label5)) {
                                            setLabelChosen(label5);
                                        }
                                        if (isLabelNotEmpty(label6)) {
                                            setLabelNotChosen(label6);
                                        }
                                        if (isLabelNotEmpty(label7)) {
                                            setLabelNotChosen(label7);
                                        }
                                        if (isLabelNotEmpty(label8)) {
                                            setLabelNotChosen(label8);
                                        }
                                    }
                                    case 5 -> {
                                        if (isLabelNotEmpty(label1)) {
                                            setLabelNotChosen(label1);
                                        }
                                        if (isLabelNotEmpty(label2)) {
                                            setLabelNotChosen(label2);
                                        }
                                        if (isLabelNotEmpty(label3)) {
                                            setLabelNotChosen(label3);
                                        }
                                        if (isLabelNotEmpty(label4)) {
                                            setLabelNotChosen(label4);
                                        }
                                        if (isLabelNotEmpty(label5)) {
                                            setLabelNotChosen(label5);
                                        }
                                        if (isLabelNotEmpty(label6)) {
                                            setLabelChosen(label6);
                                        }
                                        if (isLabelNotEmpty(label7)) {
                                            setLabelNotChosen(label7);
                                        }
                                        if (isLabelNotEmpty(label8)) {
                                            setLabelNotChosen(label8);
                                        }
                                    }
                                    case 6 -> {
                                        if (isLabelNotEmpty(label1)) {
                                            setLabelNotChosen(label1);
                                        }
                                        if (isLabelNotEmpty(label2)) {
                                            setLabelNotChosen(label2);
                                        }
                                        if (isLabelNotEmpty(label3)) {
                                            setLabelNotChosen(label3);
                                        }
                                        if (isLabelNotEmpty(label4)) {
                                            setLabelNotChosen(label4);
                                        }
                                        if (isLabelNotEmpty(label5)) {
                                            setLabelNotChosen(label5);
                                        }
                                        if (isLabelNotEmpty(label6)) {
                                            setLabelNotChosen(label6);
                                        }
                                        if (isLabelNotEmpty(label7)) {
                                            setLabelChosen(label7);
                                        }
                                        if (isLabelNotEmpty(label8)) {
                                            setLabelNotChosen(label8);
                                        }
                                    }
                                    case 7 -> {
                                        if (isLabelNotEmpty(label1)) {
                                            setLabelNotChosen(label1);
                                        }
                                        if (isLabelNotEmpty(label2)) {
                                            setLabelNotChosen(label2);
                                        }
                                        if (isLabelNotEmpty(label3)) {
                                            setLabelNotChosen(label3);
                                        }
                                        if (isLabelNotEmpty(label4)) {
                                            setLabelNotChosen(label4);
                                        }
                                        if (isLabelNotEmpty(label5)) {
                                            setLabelNotChosen(label5);
                                        }
                                        if (isLabelNotEmpty(label6)) {
                                            setLabelNotChosen(label6);
                                        }
                                        if (isLabelNotEmpty(label7)) {
                                            setLabelNotChosen(label7);
                                        }
                                        if (isLabelNotEmpty(label8)) {
                                            setLabelChosen(label8);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * 检测当前选中的label的下一个label是否可用（是否有结果在显示）
     *
     * @return true false
     */
    private boolean isNextLabelValid() {
        boolean isNextLabelValid = false;
        switch (currentResultCount.get()) {
            case 0 -> {
                if (isLabelNotEmpty(label2)) {
                    isNextLabelValid = true;
                }
            }
            case 1 -> {
                if (isLabelNotEmpty(label3)) {
                    isNextLabelValid = true;
                }
            }
            case 2 -> {
                if (isLabelNotEmpty(label4)) {
                    isNextLabelValid = true;
                }
            }
            case 3 -> {
                if (isLabelNotEmpty(label5)) {
                    isNextLabelValid = true;
                }
            }
            case 4 -> {
                if (isLabelNotEmpty(label6)) {
                    isNextLabelValid = true;
                }
            }
            case 5 -> {
                if (isLabelNotEmpty(label7)) {
                    isNextLabelValid = true;
                }
            }
            case 6 -> {
                if (isLabelNotEmpty(label8)) {
                    isNextLabelValid = true;
                }
            }
            default -> {
                if (listResults.size() > 8) {
                    return true;
                }
            }
        }
        return isNextLabelValid;
    }

    private boolean isLabelEmpty(JLabel label) {
        boolean isEmpty = true;
        String text;
        if (label != null) {
            text = label.getText();
            if (text != null) {
                isEmpty = text.isEmpty();
            }
        }
        return isEmpty;
    }

    /**
     * 检测当前窗口是否未显示任何结果
     *
     * @param label 判断的label
     * @return true如果label上有显示 否则false
     */
    private boolean isLabelNotEmpty(JLabel label) {
        return !isLabelEmpty(label);
    }

    //获取当前选中label的编号 从0-7
    private int getCurrentLabelPos() {
        return currentLabelSelectedPosition.get();
    }

    private void addSearchBarMouseWheelListener() {
        searchBar.addMouseWheelListener(e -> {
            mouseWheelTime = System.currentTimeMillis();
            isLockMouseMotion.set(true);
            if (e.getPreciseWheelRotation() > 0) {
                //向下滚动
                if (isLabelNotEmpty(label1) && isLabelNotEmpty(label2) && isLabelNotEmpty(label3) && isLabelNotEmpty(label4)
                        && isLabelNotEmpty(label5) && isLabelNotEmpty(label6) && isLabelNotEmpty(label7) && isLabelNotEmpty(label8)) {
                    isUserPressed.set(false);
                }
                if (isNextLabelValid()) {
                    if (!getSearchBarText().isEmpty()) {
                        currentResultCount.incrementAndGet();
                        int size = listResults.size();
                        if (currentResultCount.get() >= size) {
                            currentResultCount.set(size - 1);
                        }
                        if (currentResultCount.get() <= 0) {
                            currentResultCount.set(0);
                        }
                        moveDownward(getCurrentLabelPos());
                    }
                }
            } else if (e.getPreciseWheelRotation() < 0) {
                //向上滚动
                if (isLabelNotEmpty(label1) && isLabelNotEmpty(label2) && isLabelNotEmpty(label3) && isLabelNotEmpty(label4)
                        && isLabelNotEmpty(label5) && isLabelNotEmpty(label6) && isLabelNotEmpty(label7) && isLabelNotEmpty(label8)) {
                    isUserPressed.set(true);
                }
                if (!getSearchBarText().isEmpty()) {
                    currentResultCount.getAndDecrement();
                    int size = listResults.size();
                    if (currentResultCount.get() >= size) {
                        currentResultCount.set(size - 1);
                    }
                    if (currentResultCount.get() <= 0) {
                        currentResultCount.set(0);
                    }
                    moveUpward(getCurrentLabelPos());
                }
            }
        });
    }

    private void moveDownward(int position) {
        if (menu.isVisible()) {
            return;
        }
        currentLabelSelectedPosition.incrementAndGet();
        if (currentLabelSelectedPosition.get() > 7) {
            currentLabelSelectedPosition.set(7);
        }
        switch (position) {
            case 0:
                int size = listResults.size();
                if (size == 2) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                } else if (size == 3) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                } else if (size == 4) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                } else if (size == 5) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                } else if (size == 6) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                } else if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
                break;
            case 1:
                size = listResults.size();
                if (size == 3) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                } else if (size == 4) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                } else if (size == 5) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                } else if (size == 6) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                } else if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
                break;
            case 2:
                size = listResults.size();
                if (size == 4) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelChosen(label4);
                } else if (size == 5) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelChosen(label4);
                    setLabelNotChosen(label5);
                } else if (size == 6) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                } else if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
                break;
            case 3:
                size = listResults.size();
                if (size == 5) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelChosen(label5);
                } else if (size == 6) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelChosen(label5);
                    setLabelNotChosen(label6);
                } else if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
                break;
            case 4:
                size = listResults.size();
                if (size == 6) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelChosen(label6);
                } else if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
                break;
            case 5:
                size = listResults.size();
                if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelChosen(label7);
                    setLabelNotChosen(label8);
                }
                break;
            case 6:
                setLabelNotChosen(label1);
                setLabelNotChosen(label2);
                setLabelNotChosen(label3);
                setLabelNotChosen(label4);
                setLabelNotChosen(label5);
                setLabelNotChosen(label6);
                setLabelNotChosen(label7);
                setLabelChosen(label8);
                break;
            case 7:
                tryToShowResultsAndSetLastChosen();
                repaint();
                break;
        }
    }

    private void tryToShowResultsAndSetLastChosen() {
        if (runningMode == RunningMode.NORMAL_MODE) {
            //到达最下端，刷新显示
            try {
                String path = listResults.get(currentResultCount.get() - 7);
                showResultOnLabel(path, label1, false);

                path = listResults.get(currentResultCount.get() - 6);
                showResultOnLabel(path, label2, false);

                path = listResults.get(currentResultCount.get() - 5);
                showResultOnLabel(path, label3, false);

                path = listResults.get(currentResultCount.get() - 4);
                showResultOnLabel(path, label4, false);

                path = listResults.get(currentResultCount.get() - 3);
                showResultOnLabel(path, label5, false);

                path = listResults.get(currentResultCount.get() - 2);
                showResultOnLabel(path, label6, false);

                path = listResults.get(currentResultCount.get() - 1);
                showResultOnLabel(path, label7, false);

                path = listResults.get(currentResultCount.get());
                showResultOnLabel(path, label8, true);
            } catch (ArrayIndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        } else if (runningMode == RunningMode.COMMAND_MODE) {
            //到达了最下端，刷新显示
            try {
                String command = listResults.get(currentResultCount.get() - 7);
                showCommandOnLabel(command, label1, false);

                command = listResults.get(currentResultCount.get() - 6);
                showCommandOnLabel(command, label2, false);

                command = listResults.get(currentResultCount.get() - 5);
                showCommandOnLabel(command, label3, false);

                command = listResults.get(currentResultCount.get() - 4);
                showCommandOnLabel(command, label4, false);

                command = listResults.get(currentResultCount.get() - 3);
                showCommandOnLabel(command, label5, false);

                command = listResults.get(currentResultCount.get() - 2);
                showCommandOnLabel(command, label6, false);

                command = listResults.get(currentResultCount.get() - 1);
                showCommandOnLabel(command, label7, false);

                command = listResults.get(currentResultCount.get());
                showCommandOnLabel(command, label8, true);
            } catch (ArrayIndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        } else if (runningMode == RunningMode.PLUGIN_MODE) {
            try {
                String command = listResults.get(currentResultCount.get() - 7);
                showPluginResultOnLabel(command, label1, false);

                command = listResults.get(currentResultCount.get() - 6);
                showPluginResultOnLabel(command, label2, false);

                command = listResults.get(currentResultCount.get() - 5);
                showPluginResultOnLabel(command, label3, false);

                command = listResults.get(currentResultCount.get() - 4);
                showPluginResultOnLabel(command, label4, false);

                command = listResults.get(currentResultCount.get() - 3);
                showPluginResultOnLabel(command, label5, false);

                command = listResults.get(currentResultCount.get() - 2);
                showPluginResultOnLabel(command, label6, false);

                command = listResults.get(currentResultCount.get() - 1);
                showPluginResultOnLabel(command, label7, false);

                command = listResults.get(currentResultCount.get());
                showPluginResultOnLabel(command, label8, true);
            } catch (ArrayIndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 尝试显示结果，并将第一个label设置为选中
     */
    private void tryToShowResultsAndSetFirstChosen() {
        int size = listResults.size();
        if (runningMode == RunningMode.NORMAL_MODE) {
            //到达了最上端，刷新显示
            try {
                String path = listResults.get(currentResultCount.get());
                showResultOnLabel(path, label1, true);
                if (size > currentResultCount.get() + 1) {
                    path = listResults.get(currentResultCount.get() + 1);
                    showResultOnLabel(path, label2, false);
                }
                if (size > currentResultCount.get() + 2) {
                    path = listResults.get(currentResultCount.get() + 2);
                    showResultOnLabel(path, label3, false);
                }
                if (size > currentResultCount.get() + 3) {
                    path = listResults.get(currentResultCount.get() + 3);
                    showResultOnLabel(path, label4, false);
                }
                if (size > currentResultCount.get() + 4) {
                    path = listResults.get(currentResultCount.get() + 4);
                    showResultOnLabel(path, label5, false);
                }
                if (size > currentResultCount.get() + 5) {
                    path = listResults.get(currentResultCount.get() + 5);
                    showResultOnLabel(path, label6, false);
                }
                if (size > currentResultCount.get() + 6) {
                    path = listResults.get(currentResultCount.get() + 6);
                    showResultOnLabel(path, label7, false);
                }
                if (size > currentResultCount.get() + 7) {
                    path = listResults.get(currentResultCount.get() + 7);
                    showResultOnLabel(path, label8, false);
                }
            } catch (ArrayIndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        } else if (runningMode == RunningMode.COMMAND_MODE) {
            //到达了最上端，刷新显示
            try {
                String command = listResults.get(currentResultCount.get());
                showCommandOnLabel(command, label1, true);
                if (size > currentResultCount.get() + 1) {
                    command = listResults.get(currentResultCount.get() + 1);
                    showCommandOnLabel(command, label2, false);
                }
                if (size > currentResultCount.get() + 2) {
                    command = listResults.get(currentResultCount.get() + 2);
                    showCommandOnLabel(command, label3, false);
                }
                if (size > currentResultCount.get() + 3) {
                    command = listResults.get(currentResultCount.get() + 3);
                    showCommandOnLabel(command, label4, false);
                }
                if (size > currentResultCount.get() + 4) {
                    command = listResults.get(currentResultCount.get() + 4);
                    showCommandOnLabel(command, label5, false);
                }
                if (size > currentResultCount.get() + 5) {
                    command = listResults.get(currentResultCount.get() + 5);
                    showCommandOnLabel(command, label6, false);
                }
                if (size > currentResultCount.get() + 6) {
                    command = listResults.get(currentResultCount.get() + 6);
                    showCommandOnLabel(command, label7, false);
                }
                if (size > currentResultCount.get() + 7) {
                    command = listResults.get(currentResultCount.get() + 7);
                    showCommandOnLabel(command, label8, false);
                }
            } catch (ArrayIndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        } else if (runningMode == RunningMode.PLUGIN_MODE) {
            try {
                String command = listResults.get(currentResultCount.get());
                showPluginResultOnLabel(command, label1, true);

                if (size > currentResultCount.get() + 1) {
                    command = listResults.get(currentResultCount.get() + 1);
                    showPluginResultOnLabel(command, label2, false);
                }
                if (size > currentResultCount.get() + 2) {
                    command = listResults.get(currentResultCount.get() + 2);
                    showPluginResultOnLabel(command, label3, false);
                }
                if (size > currentResultCount.get() + 3) {
                    command = listResults.get(currentResultCount.get() + 3);
                    showPluginResultOnLabel(command, label4, false);
                }
                if (size > currentResultCount.get() + 4) {
                    command = listResults.get(currentResultCount.get() + 4);
                    showPluginResultOnLabel(command, label5, false);
                }
                if (size > currentResultCount.get() + 5) {
                    command = listResults.get(currentResultCount.get() + 5);
                    showPluginResultOnLabel(command, label6, false);
                }
                if (size > currentResultCount.get() + 6) {
                    command = listResults.get(currentResultCount.get() + 6);
                    showPluginResultOnLabel(command, label7, false);
                }
                if (size > currentResultCount.get() + 7) {
                    command = listResults.get(currentResultCount.get() + 7);
                    showPluginResultOnLabel(command, label8, false);
                }
            } catch (ArrayIndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        }
    }

    private void moveUpward(int position) {
        if (menu.isVisible()) {
            return;
        }
        currentLabelSelectedPosition.decrementAndGet();
        if (currentLabelSelectedPosition.get() < 0) {
            currentLabelSelectedPosition.set(0);
        }
        int size;
        switch (position) {
            case 0 -> {
                tryToShowResultsAndSetFirstChosen();
                repaint();
            }
            case 1 -> {
                size = listResults.size();
                if (size == 2) {
                    setLabelChosen(label1);
                    setLabelNotChosen(label2);
                } else if (size == 3) {
                    setLabelChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                } else if (size == 4) {
                    setLabelChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                } else if (size == 5) {
                    setLabelChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                } else if (size == 6) {
                    setLabelChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                } else if (size == 7) {
                    setLabelChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
            }
            case 2 -> {
                size = listResults.size();
                if (size == 3) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                } else if (size == 4) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                } else if (size == 5) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                } else if (size == 6) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                } else if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
            }
            case 3 -> {
                size = listResults.size();
                if (size == 4) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                } else if (size == 5) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                } else if (size == 6) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                } else if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
            }
            case 4 -> {
                size = listResults.size();
                if (size == 5) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelChosen(label4);
                    setLabelNotChosen(label5);
                } else if (size == 6) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                } else if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
            }
            case 5 -> {
                size = listResults.size();
                if (size == 6) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelChosen(label5);
                    setLabelNotChosen(label6);
                } else if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelChosen(label5);
                    setLabelNotChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
            }
            case 6 -> {
                size = listResults.size();
                if (size == 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelChosen(label6);
                    setLabelNotChosen(label7);
                } else if (size > 7) {
                    setLabelNotChosen(label1);
                    setLabelNotChosen(label2);
                    setLabelNotChosen(label3);
                    setLabelNotChosen(label4);
                    setLabelNotChosen(label5);
                    setLabelChosen(label6);
                    setLabelNotChosen(label7);
                    setLabelNotChosen(label8);
                }
            }
            case 7 -> {
                setLabelNotChosen(label1);
                setLabelNotChosen(label2);
                setLabelNotChosen(label3);
                setLabelNotChosen(label4);
                setLabelNotChosen(label5);
                setLabelNotChosen(label6);
                setLabelChosen(label7);
                setLabelNotChosen(label8);
            }
        }
    }

    /**
     * 只在重新输入需要初始化所有设置时使用
     */
    private void resetStatusOnTextChanged() {
        firstResultStartShowingTime = 0;
        currentResultCount.set(0);
        currentLabelSelectedPosition.set(0);
        isRoundRadiusSet.set(false);
        searchInfoLabel.setText("");
        searchInfoLabel.setIcon(null);
    }

    /**
     * 设置当前运行模式
     *
     * @return 是否发送startTime以及开始信号
     */
    private boolean setRunningMode(AtomicBoolean isPluginSearchBarClearReady) {
        if (currentUsingPlugin != null) {
            return true;
        }
        String searchBarText = getSearchBarText();
        if (searchBarText == null || searchBarText.isEmpty()) {
            runningMode = RunningMode.NORMAL_MODE;
        } else {
            char first = searchBarText.charAt(0);
            if (first == ':') {
                runningMode = RunningMode.COMMAND_MODE;
            } else if (first == '>') {
                String subText = searchBarText.substring(1);
                String[] splitByBlank = RegexUtil.blank.split(subText);
                if (searchBarText.charAt(searchBarText.length() - 1) == ' ') {
                    GetPluginByIdentifierEvent getPluginByIdentifierEvent = new GetPluginByIdentifierEvent(splitByBlank[0]);
                    EventManagement eventManagement = EventManagement.getInstance();
                    eventManagement.putEvent(getPluginByIdentifierEvent);
                    eventManagement.waitForEvent(getPluginByIdentifierEvent);
                    Optional<PluginService.PluginInfo> pluginInfoOptional = getPluginByIdentifierEvent.getReturnValue();
                    pluginInfoOptional.ifPresentOrElse(pluginInfo -> {
                        currentUsingPlugin = pluginInfo.plugin;
                        runningMode = RunningMode.PLUGIN_MODE;
                        currentPluginIdentifier = splitByBlank[0];
                        SwingUtilities.invokeLater(() -> {
                            textField.setText("");
                            isPluginSearchBarClearReady.compareAndSet(isPluginSearchBarClearReady.get(), true);
                        });
                    }, () -> runningMode = RunningMode.NORMAL_MODE);
                    return false;
                } else {
                    runningMode = RunningMode.NORMAL_MODE;
                }
            } else {
                runningMode = RunningMode.NORMAL_MODE;
            }
        }
        return true;
    }

    /**
     * 当窗口太小时自动缩小字体
     */
    private void changeFontOnDisplayFailed() {
        String testStr = getSearchBarText();
        Font origin = textField.getFont();
        if (origin.canDisplayUpTo(testStr) == -1) {
            return;
        }
        TranslateService translateService = TranslateService.getInstance();
        if (IsDebug.isDebug()) {
            System.out.println();
            System.err.println("正在切换字体");
            System.out.println();
        }
        Font labelFont = label1.getFont();
        Font newFont = translateService.getFitFont(labelFont.getStyle(), labelFont.getSize(), testStr);
        textField.setFont(translateService.getFitFont(origin.getStyle(), origin.getSize(), testStr));
        label1.setFont(newFont);
        label2.setFont(newFont);
        label3.setFont(newFont);
        label4.setFont(newFont);
        label5.setFont(newFont);
        label6.setFont(newFont);
        label7.setFont(newFont);
        label8.setFont(newFont);
    }

    private void addTextFieldDocumentListener() {
        textField.getDocument().addDocumentListener(new DocumentListener() {
            private boolean isSendSignal;
            private final AtomicBoolean isPluginSearchBarClearReady = new AtomicBoolean(false);

            @Override
            public void insertUpdate(DocumentEvent e) {
                if (runningMode != RunningMode.PLUGIN_MODE) {
                    isPluginSearchBarClearReady.compareAndSet(isPluginSearchBarClearReady.get(), false);
                }
                listResults = new ArrayList<>();
                changeFontOnDisplayFailed();
                clearAllLabels();
                resetStatusOnTextChanged();
                if (runningMode == RunningMode.PLUGIN_MODE && currentUsingPlugin != null && isPluginSearchBarClearReady.get()) {
                    currentUsingPlugin.clearResultQueue();
                    currentUsingPlugin.textChanged(getSearchBarText());
                }
                isSendSignal = setRunningMode(isPluginSearchBarClearReady);
                if (isSendSignal) {
                    startTime = System.currentTimeMillis();
                    startSearchSignal.set(true);
                    isSearchNotStarted.set(true);
                    isCudaSearchNotStarted.set(true);
                }
//                EventManagement.getInstance().putEvent(new StopSearchEvent());
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                if (runningMode != RunningMode.PLUGIN_MODE) {
                    isPluginSearchBarClearReady.compareAndSet(isPluginSearchBarClearReady.get(), false);
                }
                listResults = new ArrayList<>();
                changeFontOnDisplayFailed();
                clearAllLabels();
                resetStatusOnTextChanged();
                if (getSearchBarText().isEmpty()) {
                    lastMousePositionX = 0;
                    lastMousePositionY = 0;
                    currentResultCount.set(0);
                    startTime = System.currentTimeMillis();
                    startSearchSignal.set(false);
                    isSearchNotStarted.set(false);
                    isCudaSearchNotStarted.set(false);
                } else {
                    if (runningMode == RunningMode.PLUGIN_MODE && currentUsingPlugin != null && isPluginSearchBarClearReady.get()) {
                        currentUsingPlugin.clearResultQueue();
                        currentUsingPlugin.textChanged(getSearchBarText());
                    }
                    isSendSignal = setRunningMode(isPluginSearchBarClearReady);
                    if (isSendSignal) {
                        startTime = System.currentTimeMillis();
                        startSearchSignal.set(true);
                        isSearchNotStarted.set(true);
                        isCudaSearchNotStarted.set(true);
                    }
                }
//                EventManagement.getInstance().putEvent(new StopSearchEvent());
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                listResults = new ArrayList<>();
                clearAllLabels();
                resetStatusOnTextChanged();
                startTime = System.currentTimeMillis();
                startSearchSignal.set(false);
                isSearchNotStarted.set(false);
                isCudaSearchNotStarted.set(false);
//                EventManagement.getInstance().putEvent(new StopSearchEvent());
            }
        });
    }

    private void setLabelChosenOrNotChosenMouseMode(int labelNum, JLabel label) {
        if (!isUserPressed.get() && isLabelNotEmpty(label)) {
            if (currentLabelSelectedPosition.get() == labelNum) {
                setLabelChosen(label);
            } else {
                setLabelNotChosen(label);
            }
        }
    }

    /**
     * 在搜索磁盘，创建索引时，添加当完成搜索时自动发出开始信号的线程
     */
    private void addSearchWaiter(AtomicBoolean isWaiting) {
        if (!isWaiting.get()) {
            isWaiting.set(true);
            ThreadPoolUtil.getInstance().executeTask(() -> {
                try {
                    DatabaseService databaseService = DatabaseService.getInstance();
                    while (isWaiting.get()) {
                        if (databaseService.getStatus() == Constants.Enums.DatabaseStatus.NORMAL) {
                            startTime = System.currentTimeMillis() - 300;
                            startSearchSignal.set(true);
                            isSearchNotStarted.set(true);
                            isCudaSearchNotStarted.set(true);
                            return;
                        }
                        TimeUnit.MILLISECONDS.sleep(20);
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    isWaiting.set(false);
                }
            });
        }
    }

    private void setSearchBarRadius(int x, int y, double radius, int width, int height) {
        Graphics g = searchBar.getGraphics();
        Graphics2D graphics2d = (Graphics2D) g;
        graphics2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        searchBar.setShape(new RoundRectangle2D.Double(x, y, width, height, radius, radius));
    }

    private void setSearchBarRadius(double radius, int width, int height) {
        setSearchBarRadius(0, 0, radius, width, height);
    }

    /**
     * 自动设置圆角
     */
    private void autoSetSearchBarRadius() {
        if (isRoundRadiusSet.get()) {
            return;
        }
        double roundRadius = AllConfigs.getInstance().getConfigEntity().getRoundRadius();
        int length = Math.min(listResults.size() + 1, 9);
        if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
            if (listResults.isEmpty()) {
                long waitForInputAndStartSearchTimeoutInMills = AllConfigs
                        .getInstance()
                        .getConfigEntity()
                        .getAdvancedConfigEntity()
                        .getWaitForInputAndStartSearchTimeoutInMills();
                if (getSearchBarText().isEmpty() || System.currentTimeMillis() - startTime < waitForInputAndStartSearchTimeoutInMills) {
                    setSearchBarRadius(roundRadius, searchBar.getWidth(), label1.getHeight());
                } else {
                    setSearchBarRadius(roundRadius, searchBar.getWidth(), 2 * label1.getHeight());
                }
            } else {
                setSearchBarRadius(roundRadius, searchBar.getWidth(), length * label1.getHeight());
            }
        } else if (showingMode == Constants.Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
            if (listResults.isEmpty()) {
                if (getSearchBarText().isEmpty()) {
                    setSearchBarRadius(searchInfoLabel.getX(),
                            searchInfoLabel.getY(),
                            roundRadius,
                            searchBar.getWidth(),
                            label1.getHeight());
                } else {
                    setSearchBarRadius(searchInfoLabel.getX(),
                            searchInfoLabel.getY() - label1.getHeight(),
                            roundRadius,
                            searchBar.getWidth(),
                            2 * label1.getHeight());
                }
            } else {
                setSearchBarRadius(searchInfoLabel.getX(),
                        searchInfoLabel.getY() - label1.getHeight() * (length - 1),
                        roundRadius,
                        searchBar.getWidth(),
                        length * label1.getHeight());
            }
        }
        if (length == 9) {
            isRoundRadiusSet.set(true);
        }
    }

    /**
     * 初始化线程池
     */
    private void initThreadPool() {
        sendSignalAndShowCommandThread();
        sendGetIconTaskThread();
        switchSearchBarShowingMode();
    }

    @EventListener(listenClass = BootSystemEvent.class)
    private static void init(Event event) {
        SearchBar tempInstance = getInstance();
        //开启所有线程
        tempInstance.initThreadPool();
    }

    @EventRegister(registerClass = GrabFocusOnAttachModeEvent.class)
    private static void grabFocusOnAttachMode(Event event) {
        SearchBar searchBarInstance = getInstance();
        if (searchBarInstance.isVisible()) {
            searchBarInstance.grabFocus();
        }
    }

    @EventRegister(registerClass = ShowSearchBarEvent.class)
    private static void showSearchBarEvent(Event event) {
        var showSearchBarTask = (ShowSearchBarEvent) event;
        SearchBar searchBarInstance = getInstance();
        searchBarInstance.showSearchbar(showSearchBarTask.isGrabFocus, showSearchBarTask.isSwitchToNormal);
    }

    @EventRegister(registerClass = HideSearchBarEvent.class)
    @EventListener(listenClass = RestartEvent.class)
    private static void hideSearchBarEvent(Event event) {
        SearchBar searchBarInstance = getInstance();
        searchBarInstance.detectShowingModeAndClose();
    }

    @EventRegister(registerClass = SwitchVisibleStatusEvent.class)
    private static void switchVisibleStatus(Event event) {
        var switchVisibleStatusEvent = (SwitchVisibleStatusEvent) event;
        SearchBar searchBarInstance = getInstance();
        if (searchBarInstance.isVisible()) {
            searchBarInstance.detectShowingModeAndClose();
        } else {
            searchBarInstance.showSearchbar(switchVisibleStatusEvent.isGrabFocus, switchVisibleStatusEvent.isSwitchToNormal);
        }
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void stopIconThread(Event event) {
        GetIconUtil.getInstance().stopWorkingThread();
    }

    @EventListener(listenClass = SetConfigsEvent.class)
    private static void setSearchBar(Event event) {
        var configs = AllConfigs.getInstance().getConfigEntity();
        var searchBarInstance = getInstance();
        searchBarInstance.setTransparency(configs.getTransparency());
        searchBarInstance.setBorderColor(AllConfigs.getInstance().getBorderType(), configs.getBorderColor(), configs.getBorderThickness());
        var color = configs.getSearchBarColor();
        searchBarInstance.setSearchBarColor(color);
        searchBarInstance.setLabelColor(configs.getLabelColor());
        searchBarInstance.setDefaultBackgroundColor(configs.getDefaultBackgroundColor());
        searchBarInstance.setFontColorWithCoverage(configs.getFontColorWithCoverage());
        searchBarInstance.setLabelFontColor(configs.getFontColor());
        searchBarInstance.setSearchBarFontColor(configs.getSearchBarFontColor());
        searchBarInstance.searchInfoLabel.setBackground(new Color(color));
        if (ColorUtil.isDark(color)) {
            searchBarInstance.textField.setCaretColor(Color.WHITE);
        } else {
            searchBarInstance.textField.setCaretColor(Color.BLACK);
        }
    }

    private enum RunningMode {
        NORMAL_MODE, COMMAND_MODE, PLUGIN_MODE
    }

    static class IsStartTimeSet {
        static final AtomicBoolean isStartTimeSet = new AtomicBoolean(false);
    }

    @EventRegister(registerClass = PreviewSearchBarEvent.class)
    private static void previewSearchBarEvent(Event event) {
        SearchBar searchBar = getInstance();
        if (searchBar.isPreviewMode.get()) {
            EventManagement eventManagement = EventManagement.getInstance();
            PreviewSearchBarEvent preview = (PreviewSearchBarEvent) event;
            searchBar.setBorderColor(preview.borderType, preview.borderColor, preview.borderThickness);
            searchBar.setSearchBarColor(preview.searchBarColor);
            searchBar.setDefaultBackgroundColor(preview.defaultBackgroundColor);
            searchBar.setSearchBarFontColor(preview.searchBarFontColor);
            searchBar.setFontColorWithCoverage(preview.chosenLabelFontColor);
            searchBar.setLabelColor(preview.chosenLabelColor);
            searchBar.setLabelFontColor(preview.unchosenLabelFontColor);
            searchBar.searchInfoLabel.setBackground(new Color(preview.searchBarColor));
            if (ColorUtil.isDark(preview.searchBarColor)) {
                searchBar.textField.setCaretColor(Color.WHITE);
            } else {
                searchBar.textField.setCaretColor(Color.BLACK);
            }
            if (!searchBar.isVisible()) {
                eventManagement.putEvent(new ShowSearchBarEvent(false));
            }
            if (searchBar.getSearchBarText() == null || searchBar.getSearchBarText().isEmpty()) {
                try {
                    TimeUnit.MILLISECONDS.sleep(50);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                SwingUtilities.invokeLater(() -> {
                    if (!IsStartTimeSet.isStartTimeSet.get()) {
                        searchBar.textField.setText("a");
                        IsStartTimeSet.isStartTimeSet.set(true);
                        searchBar.startTime = System.currentTimeMillis();
                        searchBar.startSearchSignal.set(true);
                        searchBar.isSearchNotStarted.set(true);
                        searchBar.isCudaSearchNotStarted.set(true);
                    }
                });
            }
            searchBar.repaint();
        }
    }

    @EventRegister(registerClass = StartPreviewEvent.class)
    private static void startPreviewEvent(Event event) {
        getInstance().isPreviewMode.set(true);
    }

    @EventRegister(registerClass = StopPreviewEvent.class)
    private static void stopPreviewEvent(Event event) {
        getInstance().isPreviewMode.set(false);
        IsStartTimeSet.isStartTimeSet.set(false);
    }

    @EventRegister(registerClass = GetShowingModeEvent.class)
    private static void getShowingModeEvent(Event event) {
        event.setReturnValue(getInstance().showingMode);
    }

    private void sendGetIconTaskThread() {
        ThreadPoolUtil.getInstance().executeTask(() -> {
            final int labelNumber = 8;
            EventManagement eventManagement = EventManagement.getInstance();
            SearchBar searchBarInstance = getInstance();
            GetIconUtil getIconUtil = GetIconUtil.getInstance();
            HashMap<String, Field> objectHashMap = new HashMap<>();
            for (int i = 1; i <= labelNumber; i++) {
                String fieldName = "label" + i;
                Field declaredField;
                try {
                    declaredField = SearchBar.class.getDeclaredField(fieldName);
                } catch (NoSuchFieldException e) {
                    throw new RuntimeException(e);
                }
                declaredField.setAccessible(true);
                objectHashMap.put(fieldName, declaredField);
            }
            while (eventManagement.notMainExit()) {
                switch (runningMode) {
                    case NORMAL_MODE, COMMAND_MODE -> {
                        if (!isVisible()) {
                            break;
                        }
                        for (int i = 1; i <= labelNumber; ++i) {
                            String labelName = "label" + i;
                            Field labelField = objectHashMap.get(labelName);
                            JLabel labelInstance;
                            try {
                                labelInstance = (JLabel) labelField.get(searchBarInstance);
                            } catch (IllegalAccessException e) {
                                throw new RuntimeException(e);
                            }
                            var showPath = labelShowingPathInfo.getOrDefault(labelInstance, "");
                            String lastShowPath = labelLastShowingPathInfo.getOrDefault(labelInstance, "");
                            if (!showPath.equals(lastShowPath) && !showPath.isEmpty()) {
                                getIconUtil.getBigIcon(showPath, iconSideLength, iconSideLength, icon -> {
                                    SwingUtilities.invokeLater(() -> labelInstance.setIcon(icon));
                                    labelLastShowingPathInfo.put(labelInstance, showPath);
                                }, (icon, isTimeout) -> {
                                    if (isTimeout) {
                                        return;
                                    }
                                    SwingUtilities.invokeLater(() -> labelInstance.setIcon(icon));
                                    labelLastShowingPathInfo.put(labelInstance, showPath);
                                });
                            }
                        }
                    }
                    default -> {
                    }
                }
                try {
                    TimeUnit.MILLISECONDS.sleep(250);
                } catch (InterruptedException ignored) {
                    // ignored
                }
            }
        });
    }

    /**
     * 自动切换显示模式线程
     */
    private void switchSearchBarShowingMode() {
        ThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            GetHandle.INSTANCE.start();
            AllConfigs allConfigs = AllConfigs.getInstance();
            var screenInfo = new Object() {
                Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
                long screenSizeUpdateTime = System.currentTimeMillis();
            };
            final long updateScreenTimeLimit = 5000;
            while (eventManagement.notMainExit()) {
                // 每隔5s更新一次屏幕大小
                if (System.currentTimeMillis() - screenInfo.screenSizeUpdateTime > updateScreenTimeLimit) {
                    screenInfo.screenSizeUpdateTime = System.currentTimeMillis();
                    screenInfo.screenSize = Toolkit.getDefaultToolkit().getScreenSize();
                }
                if (showingMode == Constants.Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                    String searchInfoLabelText = searchInfoLabel.getText();
                    if (searchInfoLabelText != null && searchInfoLabelText.isEmpty()) {
                        SwingUtilities.invokeLater(() ->
                                searchInfoLabel.setText(TranslateService
                                        .getInstance()
                                        .getTranslation("Double-click shift to switch here")));
                    }
                    getExplorerSizeAndChangeSearchBarSizeExplorerMode(screenInfo.screenSize);
                } else {
                    int width = screenInfo.screenSize.width;
                    int height = screenInfo.screenSize.height;
                    int searchBarWidth = (int) (width * SEARCH_BAR_WIDTH_RATIO);
                    int searchBarHeight = (int) (height * SEARCH_BAR_HEIGHT_RATIO);
                    int positionX, positionY;
                    if (isPreviewMode.get()) {
                        positionX = 50;
                        positionY = 50;
                    } else {
                        positionX = width / 2 - searchBarWidth / 2;
                        positionY = height / 2 - searchBarHeight / 3;
                    }
                    changeSearchBarSize(positionX, positionY, searchBarWidth, searchBarHeight);
                    setTextFieldAtTop(searchBarHeight);
                }

                boolean isChangeToAttach = GetHandle.INSTANCE.changeToAttach();
                boolean attachExplorer = allConfigs.getConfigEntity().isAttachExplorer();
                if (isChangeToAttach && attachExplorer && !isSwitchToNormalManual.get()) {
                    try {
                        switchToExplorerAttachMode();
                    } catch (InterruptedException ignored) {
                        // ignored
                    }
                } else {
                    if (GetHandle.INSTANCE.changeToNormal() &&
                            showingMode != Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        switchToNormalMode(true, screenInfo.screenSize);
                    } else if (isSwitchToNormalManual.get() &&
                            showingMode != Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        grabFocus();
                        switchToNormalMode(false, screenInfo.screenSize);
                    }
                }
                try {
                    TimeUnit.MILLISECONDS.sleep(10);
                } catch (InterruptedException ignored) {
                    // ignored
                }
            }

            GetHandle.INSTANCE.stop();
        });
    }

    /**
     * 获取explorer窗口大小，并修改显示模式和大小
     */
    private void getExplorerSizeAndChangeSearchBarSizeExplorerMode(Dimension screenSize) {
        double dpi = DpiUtil.getDpi();
        long explorerWidth = (long) (GetHandle.INSTANCE.getExplorerWidth() / dpi);
        long explorerHeight = (long) (GetHandle.INSTANCE.getExplorerHeight() / dpi);
        long explorerX = (long) (GetHandle.INSTANCE.getExplorerX() / dpi);
        long explorerY = (long) (GetHandle.INSTANCE.getExplorerY() / dpi);
        int searchBarWidth = (int) (explorerWidth * SEARCH_BAR_WIDTH_RATIO);
        int searchBarHeight = (int) (screenSize.height * SEARCH_BAR_HEIGHT_RATIO);

        int labelHeight = searchBarHeight / 9;
        //explorer窗口大于20像素才开始显示，防止误判其他系统窗口
        if (labelHeight > 20) {
            int positionX;
            int positionY;
            if (GetHandle.INSTANCE.isDialogWindow()) {
                positionX = (int) (explorerX + (explorerWidth / 2 - searchBarWidth / 2));
                positionY = (int) (explorerY + explorerHeight - searchBarHeight + labelHeight);
            } else {
                positionX = (int) (explorerX + explorerWidth - searchBarWidth - 25 / dpi);
                positionY = (int) (explorerY + explorerHeight - searchBarHeight - labelHeight);
            }
            //设置窗口大小
            changeSearchBarSize(positionX, positionY, searchBarWidth, searchBarHeight, labelHeight);
            setLabelAtTop(searchBarHeight);
            if (!isVisible()) {
                showSearchbar();
            }
        }
    }

    /**
     * 将label置于上方，输入框位于下方，用于贴靠模式
     *
     * @param searchBarHeight 搜索框大小，用于计算坐标偏移
     */
    private void setLabelAtTop(int searchBarHeight) {
        int labelHeight = searchBarHeight / 9;
        SwingUtilities.invokeLater(() -> {
            textField.setLocation(0, (int) (labelHeight * 8 + labelHeight * (1 - TEXT_FIELD_HEIGHT_RATIO)));
            searchInfoLabel.setLocation(0, labelHeight * 8);
            int offset = 8 - listResults.size();
            offset = Math.max(0, offset);
            offset = offset == 8 ? 7 : offset;
            label1.setLocation(0, labelHeight * offset);
            label2.setLocation(0, labelHeight * (offset + 1));
            label3.setLocation(0, labelHeight * (offset + 2));
            label4.setLocation(0, labelHeight * (offset + 3));
            label5.setLocation(0, labelHeight * (offset + 4));
            label6.setLocation(0, labelHeight * (offset + 5));
            label7.setLocation(0, labelHeight * (offset + 6));
            label8.setLocation(0, labelHeight * (offset + 7));
        });
    }

    /**
     * 将输入框置于label的上方
     *
     * @param searchBarHeight 搜索框大小，用于计算坐标偏移
     */
    private void setTextFieldAtTop(int searchBarHeight) {
        int labelHeight = searchBarHeight / 9;
        int textFieldHeight = (int) (labelHeight * TEXT_FIELD_HEIGHT_RATIO);
        SwingUtilities.invokeLater(() -> {
            textField.setLocation(0, 0);
            searchInfoLabel.setLocation(0, textFieldHeight);
            label1.setLocation(0, labelHeight);
            label2.setLocation(0, labelHeight * 2);
            label3.setLocation(0, labelHeight * 3);
            label4.setLocation(0, labelHeight * 4);
            label5.setLocation(0, labelHeight * 5);
            label6.setLocation(0, labelHeight * 6);
            label7.setLocation(0, labelHeight * 7);
            label8.setLocation(0, labelHeight * 8);
        });
    }

    /**
     * 修改搜索框的大小和位置
     *
     * @param positionX       X
     * @param positionY       Y
     * @param searchBarWidth  宽度
     * @param searchBarHeight 高度
     * @param labelHeight     每个label的高度
     */
    private void changeSearchBarSize(int positionX, int positionY, int searchBarWidth, int searchBarHeight, int labelHeight) {
        if (positionX != searchBar.getX()
                || positionY != searchBar.getY()
                || searchBarWidth != searchBar.getWidth()
                || searchBarHeight != searchBar.getHeight()) {
            SwingUtilities.invokeLater(() -> {
                //设置窗口大小
                searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
                //设置label大小
                int textFieldHeight = (int) (labelHeight * TEXT_FIELD_HEIGHT_RATIO);
                setLabelSize(searchBarWidth, labelHeight, label1);
                setLabelSize(searchBarWidth, labelHeight, label2);
                setLabelSize(searchBarWidth, labelHeight, label3);
                setLabelSize(searchBarWidth, labelHeight, label4);
                setLabelSize(searchBarWidth, labelHeight, label5);
                setLabelSize(searchBarWidth, labelHeight, label6);
                setLabelSize(searchBarWidth, labelHeight, label7);
                setLabelSize(searchBarWidth, labelHeight, label8);
                setLabelSize(searchBarWidth, labelHeight - textFieldHeight, searchInfoLabel);
                //设置textField大小
                textField.setSize(searchBarWidth, textFieldHeight);
            });
        }
    }

    /**
     * 修改搜索框的大小和位置
     *
     * @param positionX       X
     * @param positionY       Y
     * @param searchBarWidth  宽度
     * @param searchBarHeight 高度
     */
    private void changeSearchBarSize(int positionX, int positionY, int searchBarWidth, int searchBarHeight) {
        int labelHeight = searchBarHeight / 9;
        changeSearchBarSize(positionX, positionY, searchBarWidth, searchBarHeight, labelHeight);
    }

    private void switchToExplorerAttachMode() throws InterruptedException {
        int searchBarHeight = (int) (GetHandle.INSTANCE.getExplorerHeight() * 0.75);
        int labelHeight = searchBarHeight / 9;
        if (labelHeight > 35) {
            if (showingMode != Constants.Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                //设置字体
                int textFieldHeight = (int) (labelHeight * TEXT_FIELD_HEIGHT_RATIO);
                Font textFieldFont = new Font(null, Font.PLAIN, getTextFieldFontSizeByTextFieldHeight(textFieldHeight));
                textField.setFont(textFieldFont);
                Font labelFont = new Font(null, Font.BOLD, getLabelFontSizeBySearchBarHeight(searchBarHeight));
                label1.setFont(labelFont);
                label2.setFont(labelFont);
                label3.setFont(labelFont);
                label4.setFont(labelFont);
                label5.setFont(labelFont);
                label6.setFont(labelFont);
                label7.setFont(labelFont);
                label8.setFont(labelFont);
                showingMode = Constants.Enums.ShowingSearchBarMode.EXPLORER_ATTACH;
                TimeUnit.MILLISECONDS.sleep(150);
            }
        }
    }

    private int getTextFieldFontSizeByTextFieldHeight(int textFieldHeight) {
        return (int) ((textFieldHeight / 0.7 * 9 * 0.3) / 96 * 72) / 4;
    }

    private int getLabelFontSizeBySearchBarHeight(int searchBarHeight) {
        return (int) (((searchBarHeight * 0.2) / 96 * 72) / 4.5);
    }

    private void switchToNormalMode(boolean isCloseWindow, Dimension screenSize) {
        if (isCloseWindow) {
            closeSearchBar();
        } else {
            closeWithoutHideSearchBar();
        }
        int height = screenSize.height;
        int searchBarHeight = (int) (height * 0.5);
        //设置字体
        int labelHeight = searchBarHeight / 9;
        int textFieldHeight = (int) (labelHeight * TEXT_FIELD_HEIGHT_RATIO);
        Font labelFont = new Font(null, Font.BOLD, getLabelFontSizeBySearchBarHeight(searchBarHeight));
        Font textFieldFont = new Font(null, Font.PLAIN, getTextFieldFontSizeByTextFieldHeight(textFieldHeight));
        textField.setFont(textFieldFont);
        label1.setFont(labelFont);
        label2.setFont(labelFont);
        label3.setFont(labelFont);
        label4.setFont(labelFont);
        label5.setFont(labelFont);
        label6.setFont(labelFont);
        label7.setFont(labelFont);
        label8.setFont(labelFont);
        showingMode = Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING;
        searchBar.setOpacity(AllConfigs.getInstance().getConfigEntity().getTransparency());
        try {
            TimeUnit.MILLISECONDS.sleep(150);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 在鼠标滚轮往下滑的过程中，不检测鼠标指针的移动事件
     */
    private void lockMouseMotionThread() {
        ThreadPoolUtil.getInstance().executeTask(() -> {
            //锁住MouseMotion检测，阻止同时发出两个动作
            try {
                while (isVisible()) {
                    if (System.currentTimeMillis() - mouseWheelTime > 500) {
                        isLockMouseMotion.set(false);
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                isLockMouseMotionThreadNotExist.set(true);
            }
        });
    }

    /**
     * 不断尝试显示结果
     */
    private void tryToShowRecordsThread() {
        ThreadPoolUtil.getInstance().executeTask(() -> {
            //显示结果线程
            try {
                clearAllLabels();
                while (isVisible()) {
                    String text = getSearchBarText();
                    if (text.isEmpty()) {
                        clearAllLabels();
                    } else if (!listResults.isEmpty()) {
                        //在结果不足8个的时候不断尝试显示
                        //设置窗口上的文字和图片显示
                        var hasLabelEmpty = isLabelEmpty(label1) ||
                                isLabelEmpty(label2) ||
                                isLabelEmpty(label3) ||
                                isLabelEmpty(label4) ||
                                isLabelEmpty(label5) ||
                                isLabelEmpty(label6) ||
                                isLabelEmpty(label7) ||
                                isLabelEmpty(label8);
                        if (currentResultCount.get() == 0 && hasLabelEmpty) {
                            tryToShowResultsAndSetFirstChosen();
                        }
                        //设置窗口是被选中还是未被选中，鼠标模式
                        setLabelChosenOrNotChosenMouseMode(0, label1);
                        setLabelChosenOrNotChosenMouseMode(1, label2);
                        setLabelChosenOrNotChosenMouseMode(2, label3);
                        setLabelChosenOrNotChosenMouseMode(3, label4);
                        setLabelChosenOrNotChosenMouseMode(4, label5);
                        setLabelChosenOrNotChosenMouseMode(5, label6);
                        setLabelChosenOrNotChosenMouseMode(6, label7);
                        setLabelChosenOrNotChosenMouseMode(7, label8);

                        if (firstResultStartShowingTime == 0) {
                            firstResultStartShowingTime = System.currentTimeMillis();
                        }
                    } else {
                        label1.setText(null);
                        label1.setName(null);
                        label1.setIcon(null);
                        clearALabel(label2);
                        clearALabel(label3);
                        clearALabel(label4);
                        clearALabel(label5);
                        clearALabel(label6);
                        clearALabel(label7);
                        clearALabel(label8);
                    }
                    SwingUtilities.invokeLater(() -> {
                        autoSetSearchBarRadius();
                        if (getSearchBarText().isEmpty() && showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                            searchInfoLabel.setText("");
                            searchInfoLabel.setIcon(null);
                        }
                    });
                    repaint();
                    try {
                        TimeUnit.MILLISECONDS.sleep(200);
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                }
            } finally {
                isTryToShowResultThreadNotExist.set(true);
            }
        });
    }

    /**
     * 重绘Frame
     */
    private void repaint() {
        if (isPreviewMode.get()) {
            SwingUtilities.invokeLater(() -> SwingUtilities.updateComponentTreeUI(searchBar));
        }
        SwingUtilities.invokeLater(searchBar::repaint);
    }

    private void addShowSearchStatusThread(DatabaseService.SearchTask currentTaskRef) {
        ThreadPoolUtil.getInstance().executeTask(() -> {
            ArrayList<String> listResultsTemp = listResults;
            var isIconSetObj = new Object() {
                boolean isIconSet = false;
            };
            while (isVisible() && !getSearchBarText().isEmpty() && !currentTaskRef.isSearchDone() && listResultsTemp == listResults) {
                if (runningMode != RunningMode.NORMAL_MODE) {
                    return;
                }
                SwingUtilities.invokeLater(() -> {
                    if (!isIconSetObj.isIconSet) {
                        isIconSetObj.isIconSet = true;
                        searchInfoLabel.setIcon(GetIconUtil.getInstance().getBigIcon("loadingIcon", iconSideLength, iconSideLength));
                    }
                    searchInfoLabel.setText(TranslateService.INSTANCE.getTranslation("Searching") + "    " +
                            TranslateService.INSTANCE.getTranslation("Currently selected") + ": " + (currentResultCount.get() + 1) + "    " +
                            TranslateService.INSTANCE.getTranslation("Number of current results") + ": " + listResults.size());
                });
                repaint();
                try {
                    TimeUnit.MILLISECONDS.sleep(250);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            final long startShowSearchDoneTime = System.currentTimeMillis();
            while (isVisible() && !getSearchBarText().isEmpty() &&
                    System.currentTimeMillis() - startShowSearchDoneTime < 3000 && listResultsTemp == listResults) {
                if (runningMode != RunningMode.NORMAL_MODE) {
                    return;
                }
                SwingUtilities.invokeLater(() -> {
                    searchInfoLabel.setText(TranslateService.INSTANCE.getTranslation("Search Done") + "    " +
                            TranslateService.INSTANCE.getTranslation("Currently selected") + ": " + (currentResultCount.get() + 1) + "    " +
                            TranslateService.INSTANCE.getTranslation("Number of current results") + ": " + listResults.size());
                    searchInfoLabel.setIcon(
                            GetIconUtil.getInstance().getBigIcon("completeIcon", iconSideLength, iconSideLength));
                });
                try {
                    TimeUnit.MILLISECONDS.sleep(250);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            while (isVisible() && !getSearchBarText().isEmpty() && listResultsTemp == listResults) {
                if (runningMode != RunningMode.NORMAL_MODE) {
                    return;
                }
                showSelectInfoOnLabel();
                try {
                    TimeUnit.MILLISECONDS.sleep(250);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        });
    }

    /**
     * 显示一共有多少个结果，当前选中哪个
     */
    private void showSelectInfoOnLabel() {
        SwingUtilities.invokeLater(() -> {
            searchInfoLabel.setText(TranslateService.INSTANCE.getTranslation("Currently selected") + ": " + (currentResultCount.get() + 1) + "    " +
                    TranslateService.INSTANCE.getTranslation("Number of current results") + ": " + listResults.size());
            searchInfoLabel.setIcon(null);
        });
    }

    /**
     * 将tempResults以及插件返回的结果转移到listResults中来显示
     */
    @SneakyThrows
    private void mergeResults(DatabaseService.SearchTask currentSearchTask) {
        var pluginService = PluginService.getInstance();
        var listResultsTemp = listResults;
        var allPlugins = pluginService.getAllPlugins();
        while (isVisible()) {
            //用户重新输入关键字
            if (listResultsTemp != listResults) {
                return;
            }
            if (getSearchBarText().isEmpty()) {
                listResults.clear();
            } else {
                if (runningMode == RunningMode.NORMAL_MODE) {
                    mergeResultMethod(currentSearchTask, listResultsTemp, allPlugins);
                }
            }
            TimeUnit.MILLISECONDS.sleep(1);
        }
    }

    private void mergeResultMethod(DatabaseService.SearchTask tempSearchTask,
                                   ArrayList<String> listResultsTemp,
                                   Set<PluginService.PluginInfo> allPlugins) {
        if (tempSearchTask != null) {
            for (String each : tempSearchTask.getCacheAndPriorityResults()) {
                if (!listResultsTemp.contains(each)) {
                    listResultsTemp.add(each);
                }
                if (listResultsTemp != listResults) {
                    break;
                }
            }
        }
        out:
        for (var eachPlugin : allPlugins) {
            String each;
            while ((each = eachPlugin.plugin.pollFromResultQueue()) != null) {
                if (listResultsTemp != listResults) {
                    for (var allPlugin : allPlugins) {
                        allPlugin.plugin.clearResultQueue();
                    }
                    break out;
                }
                each = "plugin" + PLUGIN_RESULT_SPLITTER_STR + eachPlugin.plugin.identifier + PLUGIN_RESULT_SPLITTER_STR + each;
                if (!listResultsTemp.contains(each)) {
                    listResultsTemp.add(each);
                }
            }
        }
        if (tempSearchTask != null) {
            for (String each : tempSearchTask.getTempResults()) {
                if (!listResultsTemp.contains(each)) {
                    listResultsTemp.add(each);
                }
                if (listResultsTemp != listResults) {
                    break;
                }
            }
        }
    }

    private void clearAllLabelBorder() {
        label1.setBorder(null);
        label2.setBorder(null);
        label3.setBorder(null);
        label4.setBorder(null);
        label5.setBorder(null);
        label6.setBorder(null);
        label7.setBorder(null);
        label8.setBorder(null);
    }

    private void setBorder0(JComponent component, int type, Border topBorder, Border bottomBorder, Border middleBorder, Border fullBorder) {
        switch (type) {
            case 1 ->
                // 顶部
                    component.setBorder(topBorder);
            case 2 ->
                // 底部
                    component.setBorder(bottomBorder);
            case 3 ->
                // 左右
                    component.setBorder(middleBorder);
            case 4 ->
                // 全部
                    component.setBorder(fullBorder);
        }
    }

    private void chooseAndSetBorder(JComponent component, int type) {
        if (currentUsingPlugin != null && runningMode == RunningMode.PLUGIN_MODE) {
            setBorder0(component, type, pluginTopBorder, pluginBottomBorder, pluginMiddleBorder, pluginFullBorder);
        } else {
            if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                setBorder0(component, type, topBorder, bottomBorder, middleBorder, fullBorder);
            } else {
                setBorder0(component, type, explorerTopBorder, explorerBottomBorder, explorerMiddleBorder, explorerFullBorder);
            }
        }
    }

    private void setBorderOnVisible() {
        while (isVisible()) {
            String text = getSearchBarText();
            if (text == null || text.isEmpty() || System.currentTimeMillis() - startTime < 300) {
                clearAllLabelBorder();
                if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                    chooseAndSetBorder(textField, 1);
                    chooseAndSetBorder(searchInfoLabel, 2);
                } else {
                    chooseAndSetBorder(textField, 2);
                    chooseAndSetBorder(searchInfoLabel, 1);
                }
            } else {
                if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                    chooseAndSetBorder(textField, 1);
                    chooseAndSetBorder(searchInfoLabel, 2);
                } else {
                    chooseAndSetBorder(textField, 2);
                    chooseAndSetBorder(searchInfoLabel, 1);
                }
                int resultNum = listResults.size();
                if (resultNum == 0 || resultNum == 1) {
                    if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        chooseAndSetBorder(label1, 2);
                    } else {
                        chooseAndSetBorder(label1, 1);
                    }
                    label2.setBorder(null);
                    label3.setBorder(null);
                    label4.setBorder(null);
                    label5.setBorder(null);
                    label6.setBorder(null);
                    label7.setBorder(null);
                    label8.setBorder(null);
                } else if (resultNum == 2) {
                    if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        chooseAndSetBorder(label1, 3);
                        chooseAndSetBorder(label2, 2);
                    } else {
                        chooseAndSetBorder(label1, 1);
                        chooseAndSetBorder(label2, 3);
                    }
                    label3.setBorder(null);
                    label4.setBorder(null);
                    label5.setBorder(null);
                    label6.setBorder(null);
                    label7.setBorder(null);
                    label8.setBorder(null);
                } else if (resultNum == 3) {
                    if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        chooseAndSetBorder(label1, 3);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 2);
                    } else {
                        chooseAndSetBorder(label1, 1);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                    }
                    label4.setBorder(null);
                    label5.setBorder(null);
                    label6.setBorder(null);
                    label7.setBorder(null);
                    label8.setBorder(null);
                } else if (resultNum == 4) {
                    if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        chooseAndSetBorder(label1, 3);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                        chooseAndSetBorder(label4, 2);
                    } else {
                        chooseAndSetBorder(label1, 1);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);

                        label4.setBorder(middleBorder);
                    }
                    label5.setBorder(null);
                    label6.setBorder(null);
                    label7.setBorder(null);
                    label8.setBorder(null);
                } else if (resultNum == 5) {
                    if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        chooseAndSetBorder(label1, 3);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                        chooseAndSetBorder(label4, 3);
                        chooseAndSetBorder(label5, 2);
                    } else {
                        chooseAndSetBorder(label1, 1);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                        chooseAndSetBorder(label4, 3);
                        chooseAndSetBorder(label5, 3);
                    }
                    label6.setBorder(null);
                    label7.setBorder(null);
                    label8.setBorder(null);
                } else if (resultNum == 6) {
                    if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        chooseAndSetBorder(label1, 3);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                        chooseAndSetBorder(label4, 3);
                        chooseAndSetBorder(label5, 3);
                        chooseAndSetBorder(label6, 2);
                    } else {
                        chooseAndSetBorder(label1, 1);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                        chooseAndSetBorder(label4, 3);
                        chooseAndSetBorder(label5, 3);
                        chooseAndSetBorder(label6, 3);
                    }
                    label7.setBorder(null);
                    label8.setBorder(null);
                } else if (resultNum == 7) {
                    if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        chooseAndSetBorder(label1, 3);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                        chooseAndSetBorder(label4, 3);
                        chooseAndSetBorder(label5, 3);
                        chooseAndSetBorder(label6, 3);
                        chooseAndSetBorder(label7, 2);
                    } else {
                        chooseAndSetBorder(label1, 1);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                        chooseAndSetBorder(label4, 3);
                        chooseAndSetBorder(label5, 3);
                        chooseAndSetBorder(label6, 3);
                        chooseAndSetBorder(label7, 3);
                    }
                    label8.setBorder(null);
                } else {
                    if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        chooseAndSetBorder(label1, 3);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                        chooseAndSetBorder(label4, 3);
                        chooseAndSetBorder(label5, 3);
                        chooseAndSetBorder(label6, 3);
                        chooseAndSetBorder(label7, 3);
                        chooseAndSetBorder(label8, 2);
                    } else {
                        chooseAndSetBorder(label1, 1);
                        chooseAndSetBorder(label2, 3);
                        chooseAndSetBorder(label3, 3);
                        chooseAndSetBorder(label4, 3);
                        chooseAndSetBorder(label5, 3);
                        chooseAndSetBorder(label6, 3);
                        chooseAndSetBorder(label7, 3);
                        chooseAndSetBorder(label8, 2);
                    }
                }
            }
            try {
                TimeUnit.MILLISECONDS.sleep(150);
            } catch (InterruptedException ignored) {
                // ignored
            }
        }
        clearAllLabelBorder();
        isBorderThreadNotExist.set(true);
    }

    /**
     * 根据用户输入设置搜索关键字
     */
    private void setSearchKeywordsAndSearchCase() {
        String searchBarText = getSearchBarText();
        if (!searchBarText.isEmpty()) {
            final int i = searchBarText.lastIndexOf('|');
            if (i == -1) {
                searchText = searchBarText;
                searchCase = null;
            } else {
                searchText = searchBarText.substring(0, i);
                var searchCaseStr = searchBarText.substring(i + 1);
                if (searchCaseStr.isEmpty()) {
                    searchCase = null;
                } else {
                    String[] tmpSearchCase = RegexUtil.semicolon.split(searchCaseStr);
                    searchCase = new String[tmpSearchCase.length];
                    for (int j = 0; j < tmpSearchCase.length; j++) {
                        searchCase[j] = tmpSearchCase[j].trim();
                    }
                }
            }
            keywords = RegexUtil.semicolon.split(searchText);
        }
//        String searchBarText = getSearchBarText();
//        if (!searchBarText.isEmpty()) {
//            final int i = searchBarText.lastIndexOf(':');
//            if (i == -1) {
//                searchText = searchBarText;
//                searchCase = null;
//            } else {
//                if (searchBarText.length() - 1 > i) {
//                    final char c = searchBarText.charAt(i + 1);
//                    // 如 test;/C:/  test;/C:\  test;/D:  test;/D:;  则判断为搜索磁盘内的文件
//                    if (c == '/' || c == File.separatorChar || c == ' ' || c == ';') {
//                        searchText = searchBarText;
//                        searchCase = null;
//                    } else {
//                        searchText = searchBarText.substring(0, i);
//                        String[] tmpSearchCase = RegexUtil.semicolon.split(searchBarText.substring(i + 1));
//                        searchCase = new String[tmpSearchCase.length];
//                        for (int j = 0; j < tmpSearchCase.length; j++) {
//                            searchCase[j] = tmpSearchCase[j].trim();
//                        }
//                    }
//                } else {
//                    searchText = searchBarText;
//                    searchCase = null;
//                }
//            }
//            keywords = RegexUtil.semicolon.split(searchText);
//        }
    }

    private DatabaseService.SearchTask sendPrepareSearchEvent() {
        EventManagement eventManagement = EventManagement.getInstance();
        final DatabaseService.SearchTask[] ret = new DatabaseService.SearchTask[1];
        if (!getSearchBarText().isEmpty()) {
            isCudaSearchNotStarted.set(false);
            if (DatabaseService.getInstance().getStatus() == Constants.Enums.DatabaseStatus.NORMAL && runningMode == RunningMode.NORMAL_MODE) {
                searchCaseToLowerAndRemoveConflict();
                var prepareSearchEvent = new PrepareSearchEvent(() -> searchText, () -> searchCase, () -> keywords);
                eventManagement.putEvent(prepareSearchEvent);
                if (eventManagement.waitForEvent(prepareSearchEvent)) {
                    System.err.println("prepare search event failed.");
                    return null;
                }
                prepareSearchEvent.getReturnValue().ifPresent(res -> {
                    ret[0] = (DatabaseService.SearchTask) res;
                    listResults = new ArrayList<>();
                });
            }
        }
        return ret[0];
    }

    /**
     * 发送开始搜索事件
     */
    private void sendSearchEvent(DatabaseService.SearchTask preparedSearchTasks) {
        EventManagement eventManagement = EventManagement.getInstance();
        final DatabaseService.SearchTask[] workingSearchTask = {preparedSearchTasks};
        if (!getSearchBarText().isEmpty()) {
            isSearchNotStarted.set(false);
            if (DatabaseService.getInstance().getStatus() == Constants.Enums.DatabaseStatus.NORMAL &&
                    runningMode == RunningMode.NORMAL_MODE) {
                searchCaseToLowerAndRemoveConflict();
                var startSearchEvent = new StartSearchEvent(() -> searchText, () -> searchCase, () -> keywords);
                eventManagement.putEvent(startSearchEvent, event -> event.getReturnValue().ifPresent(res -> {
                    if (preparedSearchTasks != res) {
                        workingSearchTask[0] = (DatabaseService.SearchTask) res;
                        listResults = new ArrayList<>();
                    }
                    addShowSearchStatusThread(workingSearchTask[0]);
                    if (isMergeThreadExist.compareAndSet(false, true)) {
                        ThreadPoolUtil.getInstance().executeTask(() -> {
                            try {
                                mergeResults(workingSearchTask[0]);
                            } finally {
                                isMergeThreadExist.set(false);
                            }
                        });
                    }
                }), event -> System.err.println("send search event failed."));
            }
        }
    }

    private void sendSignalAndShowCommandThread() {
        ThreadPoolUtil.getInstance().executeTask(() -> {
            //缓存和常用文件夹搜索线程
            //每一次输入会更新一次startTime，该线程记录endTime
            EventManagement eventManagement = EventManagement.getInstance();
            TranslateService translateService = TranslateService.getInstance();
            if (AllConfigs.isFirstRun()) {
                runInternalCommand("help");
            }
            AllConfigs allConfigs = AllConfigs.getInstance();
            final AtomicBoolean isWaiting = new AtomicBoolean(false);
            while (eventManagement.notMainExit()) {
                var advancedConfigs = allConfigs.getConfigEntity().getAdvancedConfigEntity();
                final long endTime = System.currentTimeMillis();
                DatabaseService.SearchTask searchTask = null;
                long waitForInputAndPrepareSearchTimeoutInMills = advancedConfigs.getWaitForInputAndPrepareSearchTimeoutInMills();
                if ((endTime - startTime > waitForInputAndPrepareSearchTimeoutInMills) && isCudaSearchNotStarted.get() &&
                        startSearchSignal.get() && !getSearchBarText().startsWith(">")) {
                    setSearchKeywordsAndSearchCase();
                    searchTask = sendPrepareSearchEvent();
                }
                long waitForInputAndStartSearchTimeoutInMills = advancedConfigs.getWaitForInputAndStartSearchTimeoutInMills();
                if ((endTime - startTime > waitForInputAndStartSearchTimeoutInMills) && isSearchNotStarted.get() &&
                        startSearchSignal.get() && !getSearchBarText().startsWith(">")) {
                    setSearchKeywordsAndSearchCase();
                    sendSearchEvent(searchTask);
                }

                if ((endTime - startTime > waitForInputAndStartSearchTimeoutInMills) && startSearchSignal.get()) {
                    startSearchSignal.set(false); //开始搜索 计时停止
                    if (runningMode == RunningMode.COMMAND_MODE) {
                        //去掉冒号
                        String text = getSearchBarText();
                        if (text.length() <= 1 || !runInternalCommand(text.substring(1).toLowerCase())) {
                            LinkedHashSet<String> cmdSet = allConfigs.getCmdSet();
                            cmdSet.add(":clearbin;" + translateService.getTranslation("Clear the recycle bin"));
                            cmdSet.add(":update;" + translateService.getTranslation("Update file index"));
                            cmdSet.add(":clearUpdate;" + translateService.getTranslation("Clear the database and update file index"));
                            cmdSet.add(":help;" + translateService.getTranslation("View help"));
                            cmdSet.add(":version;" + translateService.getTranslation("View Version"));
                            for (String i : cmdSet) {
                                String[] cmdInfo = RegexUtil.semicolon.split(i);
                                if (cmdInfo[0].contains(text.substring(1))) {
                                    String result = translateService.getTranslation("Run command") + i;
                                    listResults.add(result);
                                }
                                if (cmdInfo[0].equals(text)) {
                                    detectShowingModeAndClose();
                                    openWithoutAdmin(cmdInfo[1]);
                                }
                            }
                        }
                    } else if (runningMode == RunningMode.NORMAL_MODE) {
                        String searchBarText = getSearchBarText();
                        if (!searchBarText.isEmpty() && searchBarText.charAt(0) == '>') {
                            if (searchBarText.length() > 1) {
                                searchBarText = searchBarText.substring(1);
                            } else {
                                searchBarText = "";
                            }
                            var pluginInfos = PluginService.getInstance().searchPluginByKeyword(searchBarText);
                            for (PluginService.PluginInfo pluginInfo : pluginInfos) {
                                listResults.add(pluginInfo.identifier);
                            }
                        } else {
                            DatabaseService databaseService = DatabaseService.getInstance();
                            if (databaseService.getStatus() == Constants.Enums.DatabaseStatus.NORMAL) {
                                setLabelChosen(label1);
                            } else if (databaseService.getStatus() == Constants.Enums.DatabaseStatus.MANUAL_UPDATE) {
                                setLabelChosen(label1);
                                eventManagement.putEvent(new ShowTaskBarMessageEvent(translateService.getTranslation("Info"),
                                        translateService.getTranslation("Updating file index") + "..."));
                            } else if (databaseService.getStatus() == Constants.Enums.DatabaseStatus.VACUUM) {
                                setLabelChosen(label1);
                                eventManagement.putEvent(new ShowTaskBarMessageEvent(translateService.getTranslation("Info"),
                                        translateService.getTranslation("Organizing database")));
                            }
                        }
                    } else if (runningMode == RunningMode.PLUGIN_MODE) {
                        String result;
                        //外层循环不断尝试获取消息
                        while (runningMode == RunningMode.PLUGIN_MODE) {
                            while (currentUsingPlugin != null &&
                                    (result = currentUsingPlugin.pollFromResultQueue()) != null &&
                                    runningMode == RunningMode.PLUGIN_MODE) {
                                result = "plugin" + PLUGIN_RESULT_SPLITTER_STR + currentUsingPlugin.identifier + PLUGIN_RESULT_SPLITTER_STR + result;
                                if (!listResults.contains(result)) {
                                    listResults.add(result);
                                }
                            }
                            try {
                                TimeUnit.MILLISECONDS.sleep(10);
                            } catch (InterruptedException ignored) {
                                // ignored
                            }
                        }
                    }
                    if (DatabaseService.getInstance().getStatus() != Constants.Enums.DatabaseStatus.NORMAL) {
                        //开启线程等待搜索完成
                        addSearchWaiter(isWaiting);
                        clearAllLabels();
                    }
                }
                try {
                    TimeUnit.MILLISECONDS.sleep(10);
                } catch (InterruptedException ignored) {
                    // ignored
                }
            }
        });
    }

    private void searchCaseToLowerAndRemoveConflict() {
        if (searchCase == null || searchCase.length == 0) {
            return;
        }
        ArrayList<String> list = new ArrayList<>();
        for (String each : searchCase) {
            list.add(each.toLowerCase());
        }
        if (list.indexOf("f") != -1 && list.indexOf("d") != -1) {
            list.remove("f");
            list.remove("d");
        }
        if (list.isEmpty()) {
            searchCase = null;
        } else {
            String[] tmp = new String[list.size()];
            list.toArray(tmp);
            searchCase = tmp;
        }
    }

    private void showSearchbar() {
        showSearchbar(false, false);
    }

    private void grabFocus() {
        int x = 0, y = 0;
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        int height = screenSize.height;
        int searchBarHeight = (int) (height * SEARCH_BAR_HEIGHT_RATIO);
        int labelHeight = searchBarHeight / 9;
        long start = System.currentTimeMillis();
        try {
            while (!isVisible() && System.currentTimeMillis() - start < 3000) {
                TimeUnit.MILLISECONDS.sleep(5);
            }
        } catch (InterruptedException ignored) {
            // ignored
        }
        if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
            x = searchBar.getX() + textField.getWidth() / 2;
            y = searchBar.getY() + textField.getHeight() / 2;
        } else if (showingMode == Constants.Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
            x = searchBar.getX() + textField.getWidth() / 2;
            y = searchBar.getY() + labelHeight * 8 + searchInfoLabel.getHeight() + textField.getHeight() / 2;
        }
        RobotUtil.INSTANCE.mouseClicked(x, y, 1, InputEvent.BUTTON1_DOWN_MASK);
    }

    /**
     * 显示窗口
     *
     * @param isGrabFocus 是否强制抓取焦点
     */
    private void showSearchbar(boolean isGrabFocus, boolean isSwitchToNormal) {
        EventManagement eventManagement = EventManagement.getInstance();
        try {
            SwingUtilities.invokeAndWait(() -> {
                if (!isVisible()) {
                    searchBar.setAutoRequestFocus(isGrabFocus);
                    setVisible(true);
                    textField.requestFocusInWindow();
                    textField.setCaretPosition(0);
                    searchInfoLabel.setText("");
                    searchInfoLabel.setIcon(null);
                    startTime = System.currentTimeMillis();
                    visibleStartTime = startTime;
                    eventManagement.putEvent(new SearchBarReadyEvent(showingMode.toString()));
                } else if (isSwitchToNormal) {
                    isSwitchToNormalManual.set(true);
                }
            });
        } catch (InterruptedException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
        listResults = new ArrayList<>();
        ThreadPoolUtil.getInstance().executeTask(() -> {
            if (isGrabFocus && !isSwitchToNormal) {
                try {
                    TimeUnit.MILLISECONDS.sleep(250);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
                grabFocus();
            }
        });
        if (isBorderThreadNotExist.compareAndSet(true, false)) {
            ThreadPoolUtil.getInstance().executeTask(this::setBorderOnVisible);
        }
        if (isTryToShowResultThreadNotExist.compareAndSet(true, false)) {
            tryToShowRecordsThread();
        }
        if (isLockMouseMotionThreadNotExist.compareAndSet(true, false)) {
            lockMouseMotionThread();
        }
    }

    /**
     * 在路径中添加省略号
     *
     * @param path path
     * @return 生成后的字符串
     */
    private String getContractPath(String path) {
        String[] split = RegexUtil.getPattern("\\\\", 0).split(path);
        StringBuilder tmpPath = new StringBuilder();
        int contractLimit = 35;
        for (String tmp : split) {
            if (tmp.length() > contractLimit) {
                tmpPath.append(tmp, 0, contractLimit).append("...").append("\\");
            } else {
                tmpPath.append(tmp).append("\\");
            }
        }
        return tmpPath.toString();
    }

    /**
     * 高亮显示
     *
     * @param html     待处理的html
     * @param keywords 高亮关键字
     * @return 处理后带html
     */
    private String highLight(String html, String[] keywords) {
        StringBuilder regexPatternBuilder = new StringBuilder();
        List<String> collect = Arrays.stream(keywords).sorted((o1, o2) -> o2.length() - o1.length()).toList();
        for (String keyword : collect) {
            if (!keyword.isBlank()) {
                if (".".equals(keyword)) {
                    keyword = "\\.";
                } else if (keyword.startsWith(File.separator)) {
                    continue;
                }
                regexPatternBuilder.append(keyword).append("|");
            }
        }
        if (PinyinUtil.isStringContainChinese(html)) {
            // 挑出所有的中文字符
            Map<String, String> chinesePinyinMap = PinyinUtil.getChinesePinyinMap(html);
            // 转换成拼音后和keywords匹配，如果发现匹配出成功，则添加到正则表达式中
            chinesePinyinMap.entrySet()
                    .stream()
                    .filter(pair -> {
                        for (String each : keywords) {
                            if (each.toLowerCase().indexOf(pair.getValue().toLowerCase()) != -1) {
                                return true;
                            }
                        }
                        return false;
                    })
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue))
                    .forEach((k, v) -> regexPatternBuilder.append(k).append("|"));
        }
        if (regexPatternBuilder.length() > 0) {
            String pattern = regexPatternBuilder.substring(0, regexPatternBuilder.length() - 1);
            Pattern compile = RegexUtil.getPattern(pattern, Pattern.CASE_INSENSITIVE);
            Matcher matcher = compile.matcher(html);
            html = matcher.replaceAll((matchResult) -> {
                String group = matchResult.group();
                String s = "#" + ColorUtil.parseColorHex(fontColorWithCoverage);
                return "<span style=\"color: " + s + ";\">" + group + "</span>";
            });
        }
        return html;
    }


    /**
     * 根据path或command生成显示html
     *
     * @param path                     path
     * @param command                  command
     * @param isResultStringContracted 结果是否压缩，在显示path的情况下可能由于文件路径太长而不显示
     * @return html
     */
    private String getHtml(String path, String command, boolean[] isResultStringContracted) {
        String colorHex = "#" + ColorUtil.parseColorHex(labelFontColor);
        String template = "<html><body style=\"color: " + colorHex + ";\">%s</body></html>";
        isResultStringContracted[0] = false;
        if (path == null) {
            // 命令模式
            String[] info = RegexUtil.semicolon.split(command);
            String commandPath = info[1];
            String commandName = info[0];
            var keywordToHighlight = new String[]{getSearchBarText().substring(1)};
            return String.format(template, "<div>" +
                    highLight(commandName, keywordToHighlight) + "<br>" +
                    "<div style=\"overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: smaller;\">" +
                    "&gt;&gt;" + commandPath +
                    "</div>" +
                    "</div>");
        }
        if (command == null) {
            // 普通模式
            String parentPath = FileUtil.getParentPath(path);
            String fileName = FileUtil.getFileName(path);
            parentPath = getContractPath(parentPath);
            int needWidth = label1.getFontMetrics(label1.getFont()).stringWidth(parentPath);
            if (label1.getWidth() >= needWidth) {
                return String.format(template, "<div>" +
                        highLight(fileName, keywords) + "<br>" +
                        "<div style=\"overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: smaller;\">" + parentPath +
                        "</div>" +
                        "</div>");
            } else {
                isResultStringContracted[0] = true;
                return String.format(template, "<div>" +
                        highLight(fileName, keywords) +
                        "</div>");
            }
        }
        return template.replace("%s", "");
    }

    /**
     * 在label上显示当前文件路径对应文件的信息
     *
     * @param path     文件路径
     * @param label    需要显示的label
     * @param isChosen 是否当前被选中
     */
    private void showResultOnLabel(String path, JLabel label, boolean isChosen) {
        if (path.startsWith("plugin")) {
            showPluginResultOnLabel(path, label, isChosen);
            return;
        }
        GetIconUtil getIconUtil = GetIconUtil.getInstance();
        String searchBarText = getSearchBarText();
        if (!searchBarText.isEmpty() && searchBarText.charAt(0) == '>') {
            var getPluginByIdentifierEvent = new GetPluginByIdentifierEvent(path);
            var eventManagement = EventManagement.getInstance();
            eventManagement.putEvent(getPluginByIdentifierEvent);
            eventManagement.waitForEvent(getPluginByIdentifierEvent);
            Optional<PluginService.PluginInfo> pluginInfoByName = getPluginByIdentifierEvent.getReturnValue();
            pluginInfoByName.ifPresent(pluginInfo -> {
                ImageIcon tmpIcon = pluginInfo.plugin.getPluginIcon();
                if (tmpIcon != null) {
                    ImageIcon pluginIcon = getIconUtil.changeIcon(tmpIcon, iconSideLength, iconSideLength);
                    label.setIcon(pluginIcon);
                }
            });
            label.setText(path);
        } else {
            //将文件的路径信息存储在label的名称中，在未被选中时只显示文件名，选中后才显示文件路径
            boolean[] isParentPathEmpty = new boolean[1];
            String allHtml = getHtml(path, null, isParentPathEmpty);
            if (isParentPathEmpty[0]) {
                String color = "#" + ColorUtil.parseColorHex(labelFontColor);
                label.setName("<html><body style=\"color: " + color + ";\">" + highLight(FileUtil.getFileName(path), keywords) + "<br>" +
                        "<div style=\"overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: smaller;\">" +
                        FileUtil.getParentPath(path) + "</div></body></html>");
            } else {
                label.setName(RESULT_LABEL_NAME_HOLDER);
            }
            var lastPath = labelShowingPathInfo.put(label, path);
            lastPath = lastPath == null ? "" : lastPath;
            if (!lastPath.equals(path)) {
                labelLastShowingPathInfo.put(label, lastPath);
            }
            label.setText(allHtml);
            if (label.getIcon() == null) {
                ImageIcon icon = getIconUtil.getBigIcon("blankIcon", iconSideLength, iconSideLength);
                label.setIcon(icon);
            }
        }
        if (isChosen) {
            setLabelChosen(label);
        } else {
            setLabelNotChosen(label);
        }
    }

    /**
     * 在label上显示插件返回的信息（由插件自己实现）
     *
     * @param result   结果
     * @param label    需要显示的label
     * @param isChosen 是否当前被选中
     */
    private void showPluginResultOnLabel(String result, JLabel label, boolean isChosen) {
        String[] resultWithPluginInfo = splitPluginResult(result);
        if (runningMode == RunningMode.PLUGIN_MODE) {
            currentUsingPlugin.showResultOnLabel(resultWithPluginInfo[2], label, isChosen);
        } else {
            var getPluginByIdentifierEvent = new GetPluginByIdentifierEvent(resultWithPluginInfo[1]);
            var eventManagement = EventManagement.getInstance();
            eventManagement.putEvent(getPluginByIdentifierEvent);
            eventManagement.waitForEvent(getPluginByIdentifierEvent);
            Optional<PluginService.PluginInfo> pluginInfoOptional = getPluginByIdentifierEvent.getReturnValue();
            pluginInfoOptional.ifPresent(pluginInfo -> pluginInfo.plugin.showResultOnLabel(resultWithPluginInfo[2], label, isChosen));
        }
        label.setName(RESULT_LABEL_NAME_HOLDER);
    }

    /**
     * 在label上显示命令信息
     *
     * @param command  命令
     * @param label    需要显示的label
     * @param isChosen 是否当前被选中
     */
    private void showCommandOnLabel(String command, JLabel label, boolean isChosen) {
        String[] info = RegexUtil.semicolon.split(command);
        if (info.length != 2) {
            return;
        }
        GetIconUtil getIconUtil = GetIconUtil.getInstance();
        String name = info[0];
        String iconKey = RegexUtil.colon.split(name)[1];
        String showStr = getHtml(null, command, new boolean[1]);
        label.setText(showStr);
        label.setName(RESULT_LABEL_NAME_HOLDER);
        ImageIcon commandIcon = getIconUtil.getCommandIcon(iconKey, iconSideLength, iconSideLength);
        if (commandIcon == null) {
            if (label.getIcon() == null) {
                ImageIcon icon = getIconUtil.getBigIcon("blankIcon", iconSideLength, iconSideLength);
                label.setIcon(icon);
            }
            String path = info[1];
            String lastPath = labelShowingPathInfo.put(label, path);
            lastPath = lastPath == null ? "" : lastPath;
            if (!lastPath.equals(path)) {
                labelLastShowingPathInfo.put(label, lastPath);
            }
        } else {
            labelShowingPathInfo.remove(label);
            labelLastShowingPathInfo.remove(label);
            label.setIcon(commandIcon);
        }
        if (isChosen) {
            setLabelChosen(label);
        } else {
            setLabelNotChosen(label);
        }
    }

    /**
     * 清空单个label的所有信息
     *
     * @param label 需要清空的label
     */
    private void clearALabel(JLabel label) {
        SwingUtilities.invokeLater(() -> {
            label.setBackground(null);
            label.setText(null);
            label.setName(null);
            label.setIcon(null);
            labelShowingPathInfo.remove(label);
            labelLastShowingPathInfo.remove(label);
        });
    }

    /**
     * 清空所有label
     */
    private void clearAllLabels() {
        clearALabel(label1);
        clearALabel(label2);
        clearALabel(label3);
        clearALabel(label4);
        clearALabel(label5);
        clearALabel(label6);
        clearALabel(label7);
        clearALabel(label8);
        repaint();
    }

    /**
     * 以管理员方式运行文件，失败则打开文件位置
     *
     * @param path 文件路径
     */
    private void openWithAdmin(String path) {
        EventManagement.getInstance().putEvent(new OpenFileEvent(OpenFileEvent.OpenStatus.WITH_ADMIN, path), event -> saveCache(path), null);
    }

    /**
     * 以普通权限运行文件，失败则打开文件位置
     *
     * @param path 文件路径
     */
    private void openWithoutAdmin(String path) {
        EventManagement.getInstance().putEvent(new OpenFileEvent(OpenFileEvent.OpenStatus.NORMAL_OPEN, path), event -> saveCache(path), null);
    }

    /**
     * 保存当前文件路径到数据库缓存
     *
     * @param content 文件路径
     */
    private void saveCache(String content) {
        AllConfigs allConfigs = AllConfigs.getInstance();
        EventManagement eventManagement = EventManagement.getInstance();
        if (DatabaseService.getInstance().getDatabaseCacheNum() < allConfigs.getConfigEntity().getCacheNumLimit()) {
            //检查缓存是否已存在
            eventManagement.putEvent(new IsCacheExistEvent(content), event -> {
                Optional<Boolean> ret = event.getReturnValue();
                ret.ifPresent(isCacheExist -> {
                    if (!isCacheExist) {
                        //不存在则添加
                        eventManagement.putEvent(new AddToCacheEvent(content));
                    }
                });
            }, null);
        }
    }

    /**
     * 设置窗口透明度
     *
     * @param trans 透明度
     */
    private void setTransparency(float trans) {
        searchBar.setOpacity(trans);
    }

    private void clearTextFieldText() {
        SwingUtilities.invokeLater(() -> textField.setText(""));
    }

    /**
     * 检测当前模式并重置状态
     */
    private void detectShowingModeAndClose() {
        if (showingMode == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
            closeSearchBar();
        } else if (showingMode == Constants.Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
            closeWithoutHideSearchBar();
        }
    }

    private void setVisible(boolean visible) {
        if (!visible) {
            if (!isPreviewMode.get()) {
                searchBar.setVisible(false);
                EventManagement.getInstance().putEvent(new SearchBarCloseEvent());
                GetIconUtil.getInstance().clearIconCache();
            }
        } else {
            searchBar.setVisible(true);
        }
    }

    /**
     * 重置所有状态并关闭窗口
     */
    private synchronized void closeSearchBar() {
        if (!isPreviewMode.get()) {
            if (isVisible()) {
                setVisible(false);
            }
            clearAllLabels();
            clearTextFieldText();
            resetAllStatus();
        }
        menu.setVisible(false);
        isSwitchToNormalManual.set(false);
    }

    /**
     * 重置所有状态但不关闭窗口
     */
    private synchronized void closeWithoutHideSearchBar() {
        clearAllLabels();
        clearTextFieldText();
        resetAllStatus();
        listResults = new ArrayList<>();
        menu.setVisible(false);
    }

    private void resetAllStatus() {
        startTime = System.currentTimeMillis();//结束搜索
        currentResultCount.set(0);
        currentLabelSelectedPosition.set(0);
        isUserPressed.set(false);
        isLockMouseMotion.set(false);
        isOpenLastFolderPressed.set(false);
        isRunAsAdminPressed.set(false);
        isCopyPathPressed.set(false);
        startSearchSignal.set(false);
        isMouseDraggedInWindow.set(false);
        currentUsingPlugin = null;
        isRoundRadiusSet.set(false);
        searchInfoLabel.setText("");
        searchInfoLabel.setIcon(null);
    }

    /**
     * 判断窗口是否可见
     *
     * @return true如果可见 否则false
     */
    private boolean isVisible() {
        return searchBarContentPane.isShowing();
    }

    private void setFontColorWithCoverage(int colorNum) {
        fontColorWithCoverage = new Color(colorNum);
    }

    private void setDefaultBackgroundColor(int colorNum) {
        backgroundColor = new Color(colorNum);
    }

    private void setLabelColor(int colorNum) {
        labelColor = new Color(colorNum);
    }

    private void setLabelFontColor(int colorNum) {
        labelFontColor = new Color(colorNum);
    }

    private void setSearchBarColor(int colorNum) {
        textField.setBackground(new Color(colorNum));
    }

    private void setSearchBarFontColor(int colorNum) {
        textField.setForeground(new Color(colorNum));
    }

    private void setBorderColor(Constants.Enums.BorderType borderType, int colorNum, float borderThickness) {
        initBorder(borderType, new Color(colorNum), borderThickness);
    }
}

