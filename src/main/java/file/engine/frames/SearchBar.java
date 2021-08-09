package file.engine.frames;


import file.engine.IsDebug;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.Enums;
import file.engine.constant.Constants;
import file.engine.dllInterface.FileMonitor;
import file.engine.dllInterface.GetHandle;
import file.engine.dllInterface.IsLocalDisk;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.database.AddToCacheEvent;
import file.engine.event.handler.impl.database.DeleteFromCacheEvent;
import file.engine.event.handler.impl.database.UpdateDatabaseEvent;
import file.engine.event.handler.impl.frame.searchBar.*;
import file.engine.event.handler.impl.frame.settingsFrame.AddCacheEvent;
import file.engine.event.handler.impl.frame.settingsFrame.IsCacheExistEvent;
import file.engine.event.handler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import file.engine.event.handler.impl.monitor.disk.StartMonitorDiskEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.services.DatabaseService;
import file.engine.services.plugin.system.Plugin;
import file.engine.services.plugin.system.PluginService;
import file.engine.utils.*;
import file.engine.utils.bit.Bit;
import file.engine.utils.file.CopyFileUtil;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.Transferable;
import java.awt.event.*;
import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;


@SuppressWarnings({"IndexOfReplaceableByContains", "ListIndexOfReplaceableByContains"})
public class SearchBar {
    private final AtomicBoolean isCacheAndPrioritySearched = new AtomicBoolean(false);
    private final AtomicBoolean isLockMouseMotion = new AtomicBoolean(false);
    private final AtomicBoolean isOpenLastFolderPressed = new AtomicBoolean(false);
    private final AtomicBoolean isRunAsAdminPressed = new AtomicBoolean(false);
    private final AtomicBoolean isCopyPathPressed = new AtomicBoolean(false);
    private final AtomicBoolean startSignal = new AtomicBoolean(false);
    private final AtomicBoolean isUserPressed = new AtomicBoolean(false);
    private final AtomicBoolean isDatabaseUpdated = new AtomicBoolean(false);
    private final AtomicBoolean isMouseDraggedInWindow = new AtomicBoolean(false);
    private final AtomicBoolean isNotSqlInitialized = new AtomicBoolean(true);
    private final AtomicBoolean isBorderThreadNotExist = new AtomicBoolean(true);
    private final AtomicBoolean isRepaintFrameThreadNotExist = new AtomicBoolean(true);
    private static final AtomicBoolean isPreviewMode = new AtomicBoolean(false);
    private final AtomicBoolean isTutorialMode = new AtomicBoolean(false);
    private Border fullBorder;
    private Border topBorder;
    private Border middleBorder;
    private Border bottomBorder;
    private Border pluginFullBorder;
    private Border pluginTopBorder;
    private Border pluginMiddleBorder;
    private Border pluginBottomBorder;
    private final JFrame searchBar;
    private final JLabel label1;
    private final JLabel label2;
    private final JLabel label3;
    private final JLabel label4;
    private final JLabel label5;
    private final JLabel label6;
    private final JLabel label7;
    private final JLabel label8;
    private final AtomicInteger currentResultCount;  //保存当前选中的结果是在listResults中的第几个 范围 0 - listResults.size()
    private final JTextField textField;
    private Color labelColor;
    private Color backgroundColor;
    private Color fontColorWithCoverage;
    private Color labelFontColor;
    private volatile long startTime = 0;
    private final AtomicBoolean isWaiting = new AtomicBoolean(false);
    private final Pattern semicolon;
    private final Pattern colon;
    private final Pattern slash;
    private final Pattern blank;
    private volatile Enums.RunningMode runningMode;
    private volatile Enums.ShowingSearchBarMode showingMode;
    private long mouseWheelTime = 0;
    private final int iconSideLength;
    private volatile long visibleStartTime = 0;  //记录窗口开始可见的事件，窗口默认最短可见时间0.5秒，防止窗口快速闪烁
    private volatile long firstResultStartShowingTime = 0;  //记录开始显示结果的时间，用于防止刚开始移动到鼠标导致误触
    private final Set<String> tempResults;  //在优先文件夹和数据库cache未搜索完时暂时保存结果，搜索完后会立即被转移到listResults
    private final ConcurrentLinkedQueue<String> tableQueue;  //保存哪些表需要被查
    private final CopyOnWriteArrayList<String> listResults;  //保存从数据库中找出符合条件的记录（文件路径）
    private final Set<TableNameWeightInfo> tableSet;    //保存从0-40数据库的表，使用频率和名字对应，使经常使用的表最快被搜索到
    private final ConcurrentLinkedQueue<Integer> priorityQueue;
    private volatile String[] searchCase;
    private volatile String searchText;
    private volatile String[] keywords;
    private final DatabaseService databaseService;
    private final AtomicInteger listResultsNum;  //保存当前listResults中有多少个结果
    private final AtomicInteger tempResultNum;  //保存当前tempResults中有多少个结果
    private final AtomicInteger currentLabelSelectedPosition;   //保存当前是哪个label被选中 范围 0 - 7
    private volatile Plugin currentUsingPlugin;
    private volatile String currentPluginIdentifier;
    private final JPopupMenu menu = new JPopupMenu();
    private final JMenuItem open;
    private final JMenuItem openAsAdmin;
    private final JMenuItem copyDir;
    private final JMenuItem openLast;

    private static final int MAX_RESULTS_COUNT = 200;

    private static volatile SearchBar instance = null;

    private static class TableNameWeightInfo {
        private final String tableName;
        private final AtomicLong weight;

        private TableNameWeightInfo(String tableName) {
            this.tableName = tableName;
            this.weight = new AtomicLong(0);
        }
    }

    private SearchBar() {
        listResults = new CopyOnWriteArrayList<>();
        tempResults = ConcurrentHashMap.newKeySet();
        tableQueue = new ConcurrentLinkedQueue<>();
        tableSet = ConcurrentHashMap.newKeySet();
        priorityQueue = new ConcurrentLinkedQueue<>();
        searchBar = new JFrame();
        currentResultCount = new AtomicInteger(0);
        listResultsNum = new AtomicInteger(0);
        tempResultNum = new AtomicInteger(0);
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        open = new JMenuItem(translateUtil.getTranslation("Open"));
        openAsAdmin = new JMenuItem(translateUtil.getTranslation("Open as administrator"));
        copyDir = new JMenuItem(translateUtil.getTranslation("Copy file path"));
        openLast = new JMenuItem(translateUtil.getTranslation("Open parent folder"));
        menu.add(open);
        menu.add(openAsAdmin);
        menu.add(copyDir);
        menu.add(openLast);

        runningMode = Enums.RunningMode.NORMAL_MODE;
        showingMode = Enums.ShowingSearchBarMode.NORMAL_SHOWING;
        currentLabelSelectedPosition = new AtomicInteger(0);
        semicolon = RegexUtil.semicolon;
        colon = RegexUtil.colon;
        slash = RegexUtil.slash;
        blank = RegexUtil.blank;
        JPanel panel = new JPanel();
        Color transparentColor = new Color(0, 0, 0, 0);

        databaseService = DatabaseService.getInstance();

        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
        int width = screenSize.width;
        int height = screenSize.height;
        int searchBarWidth = (int) (width * 0.3);
        int searchBarHeight = (int) (height * 0.4);
        int positionX = width / 2 - searchBarWidth / 2;
        int positionY = height / 2 - searchBarHeight / 3;

        AllConfigs allConfigs = AllConfigs.getInstance();
        labelColor = new Color(allConfigs.getLabelColor());
        fontColorWithCoverage = new Color(allConfigs.getLabelFontColorWithCoverage());
        backgroundColor = new Color(allConfigs.getDefaultBackgroundColor());
        labelFontColor = new Color(allConfigs.getLabelFontColor());
        initBorder(allConfigs.getBorderType(), new Color(allConfigs.getBorderColor()), allConfigs.getBorderThickness());

        //frame
        searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
        searchBar.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        searchBar.setUndecorated(true);
        searchBar.getRootPane().setWindowDecorationStyle(JRootPane.NONE);
        searchBar.setBackground(transparentColor);
        searchBar.setOpacity(allConfigs.getOpacity());
        searchBar.setContentPane(panel);
        searchBar.setType(JFrame.Type.UTILITY);
        searchBar.setAlwaysOnTop(true);
        //用于C++判断是否点击了当前窗口
        searchBar.setTitle("File-Engine-SearchBar");

        //labels
        Font labelFont = new Font(Font.SANS_SERIF, Font.BOLD, getLabelFontSizeBySearchBarHeight(searchBarHeight));
        label1 = new JLabel();
        label2 = new JLabel();
        label3 = new JLabel();
        label4 = new JLabel();
        label5 = new JLabel();
        label6 = new JLabel();
        label7 = new JLabel();
        label8 = new JLabel();

        int labelHeight = searchBarHeight / 9;
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

        //TextField
        textField = new JTextField(1000);
        textField.setSize(searchBarWidth, labelHeight);
        Font textFieldFont = new Font(Font.SANS_SERIF, Font.PLAIN, getTextFieldFontSizeBySearchBarHeight(searchBarHeight));
        textField.setFont(textFieldFont);
        textField.setForeground(Color.BLACK);
        textField.setHorizontalAlignment(JTextField.LEFT);
        textField.setBackground(Color.WHITE);
        textField.setLocation(0, 0);
        textField.setOpaque(true);

        //panel
        panel.setLayout(null);
        panel.setBackground(transparentColor);
        panel.add(textField);
        panel.add(label1);
        panel.add(label2);
        panel.add(label3);
        panel.add(label4);
        panel.add(label5);
        panel.add(label6);
        panel.add(label7);
        panel.add(label8);

        initTableMap();

        initPriorityQueue();

        initMenuItems();

        //开启所有线程
        initThreadPool();

        //添加textField搜索变更检测
        addTextFieldDocumentListener();

        //添加结果的鼠标事件响应
        addSearchBarMouseListener();

        //添加结果的鼠标滚轮响应
        addSearchBarMouseWheelListener();

        //添加结果的鼠标移动事件响应
        addSearchBarMouseMotionListener();

        //添加textfield对键盘的响应
        addTextFieldKeyListener();

        addTextFieldFocusListener();
    }

    private void initMenuItems() {
        open.addActionListener(e -> {
            if (isPreviewMode.get() || isTutorialMode.get()) {
                return;
            }
            if (listResultsNum.get() != 0) {
                if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                    setVisible(false);
                }
                String res = listResults.get(currentResultCount.get());
                if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                    openWithoutAdmin(res);
                } else {
                    String[] commandInfo = semicolon.split(res);
                    boolean isExecuted = runInternalCommand(colon.split(commandInfo[0])[1]);
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
            if (listResultsNum.get() != 0) {
                if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                    setVisible(false);
                }
                String res = listResults.get(currentResultCount.get());
                if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                    openWithAdmin(res);
                } else {
                    String[] commandInfo = semicolon.split(res);
                    boolean isExecuted = runInternalCommand(colon.split(commandInfo[0])[1]);
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
            if (listResultsNum.get() != 0) {
                if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                    setVisible(false);
                }
                String res = listResults.get(currentResultCount.get());
                if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                    copyToClipBoard(res, true);
                } else {
                    String[] commandInfo = semicolon.split(res);
                    boolean isExecuted = runInternalCommand(colon.split(commandInfo[0])[1]);
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
            if (listResultsNum.get() != 0) {
                if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                    setVisible(false);
                }
                String res = listResults.get(currentResultCount.get());
                if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                    openFolderByExplorer(res);
                } else {
                    String[] commandInfo = semicolon.split(res);
                    boolean isExecuted = runInternalCommand(colon.split(commandInfo[0])[1]);
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

    private void initBorder(Enums.BorderType borderType, Color borderColor, int borderThickness) {
        if (Enums.BorderType.AROUND == borderType) {
            topBorder = BorderFactory.createMatteBorder(borderThickness, borderThickness, 0, borderThickness, borderColor);
            middleBorder = BorderFactory.createMatteBorder(0, borderThickness, 0, borderThickness, borderColor);
            bottomBorder = BorderFactory.createMatteBorder(0, borderThickness, borderThickness, borderThickness, borderColor);
            fullBorder = BorderFactory.createMatteBorder(
                    borderThickness,
                    borderThickness,
                    borderThickness,
                    borderThickness,
                    borderColor);
        } else if (Enums.BorderType.EMPTY == borderType) {
            Border emptyBorder = BorderFactory.createEmptyBorder();
            topBorder = emptyBorder;
            middleBorder = emptyBorder;
            bottomBorder = emptyBorder;
            fullBorder = emptyBorder;
        } else {
            Border lineBorder = BorderFactory.createMatteBorder(
                    borderThickness,
                    borderThickness,
                    borderThickness,
                    borderThickness,
                    borderColor);
            topBorder = lineBorder;
            middleBorder = lineBorder;
            bottomBorder = lineBorder;
            fullBorder = lineBorder;
        }
        Color highContrast = Color.RED;
        pluginTopBorder = BorderFactory.createMatteBorder(2, 2, 0, 2, highContrast);
        pluginBottomBorder = BorderFactory.createMatteBorder(0, 2, 2, 2, highContrast);
        pluginFullBorder = BorderFactory.createMatteBorder(2, 2, 2, 2, highContrast);
        pluginMiddleBorder = BorderFactory.createMatteBorder(0, 2, 0, 2, highContrast);
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
                    if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING && allConfigs.isLoseFocusClose()) {
                        if (!isTutorialMode.get()) {
                            closeSearchBar();
                        }
                    } else if (showingMode == Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                        closeWithoutHideSearchBar();
                    }
                }
            }
        });
    }

    private void initTableMap() {
        for (int i = 0; i <= Constants.ALL_TABLE_NUM; i++) {
            tableSet.add(new TableNameWeightInfo("list" + i));
        }
    }

    private TableNameWeightInfo getInfoByName(String tableName) {
        for (TableNameWeightInfo each : tableSet) {
            if (each.tableName.equals(tableName)) {
                return each;
            }
        }
        return null;
    }

    private void updateTableWeight(String tableName, long weight) {
        TableNameWeightInfo origin = getInfoByName(tableName);
        if (origin == null) {
            return;
        }
        origin.weight.addAndGet(weight);
        if (IsDebug.isDebug()) {
            System.err.println("已更新" + tableName + "权重, 之前为" + origin + "***增加了" + weight);
        }
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
    private void setLabelSize(int width, int height, int positionY, JLabel label) {
        label.setSize(width, height);
        label.setLocation(0, positionY);
    }

    /**
     * 创建需要打开的文件的快捷方式
     *
     * @param fileOrFolderPath  文件路径
     * @param writeShortCutPath 保存快捷方式的位置
     * @throws Exception 创建错误
     */
    private void createShortCut(String fileOrFolderPath, String writeShortCutPath, boolean isNotifyUser) throws Exception {
        EventManagement eventManagement = EventManagement.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        String lower = fileOrFolderPath.toLowerCase();
        if (lower.endsWith(".lnk") || lower.endsWith(".url")) {
            //直接复制文件
            CopyFileUtil.copyFile(new File(fileOrFolderPath), new File(writeShortCutPath));
        } else {
            File shortcutGen = new File("user/shortcutGenerator.vbs");
            String shortcutGenPath = shortcutGen.getAbsolutePath();
            String start = "cmd.exe /c start " + shortcutGenPath.substring(0, 2);
            String end = "\"" + shortcutGenPath.substring(2) + "\"";
            String commandToGenLnk = start + end + " /target:" + "\"" + fileOrFolderPath + "\"" + " " + "/shortcut:" + "\"" + writeShortCutPath + "\"" + " /workingdir:" + "\"" + fileOrFolderPath.substring(0, fileOrFolderPath.lastIndexOf(File.separator)) + "\"";
            Runtime.getRuntime().exec("cmd.exe " + commandToGenLnk);
        }
        if (isNotifyUser) {
            eventManagement.putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Info"),
                    translateUtil.getTranslation("Shortcut created")));
        }
    }

    /**
     * 让搜索窗口响应鼠标双击事件以打开文件
     */
    private void addSearchBarMouseListener() {
        searchBar.addMouseListener(new MouseAdapter() {
            private final AllConfigs allConfigs = AllConfigs.getInstance();

            @Override
            public void mousePressed(MouseEvent e) {
                int count = e.getClickCount();
                if (count == 2) {
                    if (isPreviewMode.get() || isTutorialMode.get()) {
                        return;
                    }
                    if (listResultsNum.get() != 0) {
                        if (runningMode != Enums.RunningMode.PLUGIN_MODE) {
                            if (showingMode != Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                                if (isVisible()) {
                                    setVisible(false);
                                }
                            }
                            String res = listResults.get(currentResultCount.get());
                            if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                                if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                                    if (isOpenLastFolderPressed.get()) {
                                        //打开上级文件夹
                                        openFolderByExplorer(res);
                                    } else if (allConfigs.isDefaultAdmin() || isRunAsAdminPressed.get()) {
                                        openWithAdmin(res);
                                    } else if (isCopyPathPressed.get()) {
                                        copyToClipBoard(res, true);
                                    } else {
                                        openWithoutAdmin(res);
                                    }
                                }
                            } else if (runningMode == Enums.RunningMode.COMMAND_MODE) {
                                String[] commandInfo = semicolon.split(res);
                                boolean isExecuted = runInternalCommand(colon.split(commandInfo[0])[1]);
                                if (isExecuted) {
                                    return;
                                }
                                File open = new File(commandInfo[1]);
                                if (isOpenLastFolderPressed.get()) {
                                    //打开上级文件夹
                                    openFolderByExplorer(open.getAbsolutePath());
                                } else if (allConfigs.isDefaultAdmin() || isRunAsAdminPressed.get()) {
                                    openWithAdmin(open.getAbsolutePath());
                                } else if (isCopyPathPressed.get()) {
                                    copyToClipBoard(open.getAbsolutePath(), true);
                                } else {
                                    openWithoutAdmin(open.getAbsolutePath());
                                }
                            }
                        } else if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                            if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                                if (currentUsingPlugin != null) {
                                    if (listResultsNum.get() != 0) {
                                        currentUsingPlugin.mousePressed(e, listResults.get(currentResultCount.get()));
                                    }
                                }
                            }
                        }
                    }
                    detectShowingModeAndClose();
                }

                if (e.getButton() == MouseEvent.BUTTON3 && runningMode != Enums.RunningMode.PLUGIN_MODE) {
                    //右键被点击
                    double dpi = GetHandle.INSTANCE.getDpi();
                    menu.show(searchBar, (int) (e.getX() / dpi), (int) (e.getY() / dpi));
                } else if (e.getButton() == MouseEvent.BUTTON1) {
                    menu.setVisible(false);
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                if (isMouseDraggedInWindow.get()) {
                    isMouseDraggedInWindow.set(false);
                    if (IsDebug.isDebug()) {
                        Point point = java.awt.MouseInfo.getPointerInfo().getLocation();
                        System.out.println("鼠标释放");
                        System.out.println("鼠标X：" + point.x);
                        System.out.println("鼠标Y：" + point.y);
                    }
                    //创建快捷方式
                    try {
                        String writePath = GetHandle.INSTANCE.getExplorerPath();
                        if (writePath != null) {
                            if (!writePath.isEmpty()) {
                                String result = listResults.get(currentResultCount.get());
                                if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                                    //普通模式直接获取文件路径
                                    File f = new File(result);
                                    createShortCut(f.getAbsolutePath(), writePath + File.separator + f.getName(), AllConfigs.getInstance().isShowTipOnCreatingLnk());
                                } else if (runningMode == Enums.RunningMode.COMMAND_MODE) {
                                    String[] commandInfo = semicolon.split(result);
                                    //获取命令后的文件路径
                                    if (commandInfo == null || commandInfo.length <= 1) {
                                        return;
                                    }
                                    File f = new File(commandInfo[1]);
                                    if (f.exists()) {
                                        createShortCut(f.getAbsolutePath(),
                                                writePath + File.separator + f.getName(), AllConfigs.getInstance().isShowTipOnCreatingLnk());
                                    }
                                }
                            }
                        }
                    } catch (Exception exception) {
                        exception.printStackTrace();
                    }
                }
                if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                    if (currentUsingPlugin != null) {
                        if (listResultsNum.get() != 0) {
                            currentUsingPlugin.mouseReleased(e, listResults.get(currentResultCount.get()));
                        }
                    }
                }
            }
        });
    }


    //在explorer attach模式时操作鼠标和键盘以快速跳转到文件位置
    private void quickJump(String result) {
        int x, y;
        RobotUtil robotUtil = RobotUtil.getInstance();
        Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
        Transferable originalData = clipboard.getContents(null);
        if (isFile(result)) {
            result = getParentPath(result);
        }
        saveCache(result);
        copyToClipBoard(result, false);
        double dpi = GetHandle.INSTANCE.getDpi();
        x = (int) (GetHandle.INSTANCE.getToolBarX() / dpi);
        y = (int) (GetHandle.INSTANCE.getToolBarY() / dpi);
        robotUtil.mouseClicked(x, y, 1, InputEvent.BUTTON1_DOWN_MASK);
        robotUtil.keyTyped(KeyEvent.VK_CONTROL, KeyEvent.VK_V);
        robotUtil.keyTyped(KeyEvent.VK_ENTER);
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                //保证在执行粘贴操作时不会被提前恢复数据
                TimeUnit.MILLISECONDS.sleep(500);
                copyToClipBoard(originalData, false);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * 复制信息到系统剪贴板
     *
     * @param res 需要复制的信息
     */
    private void copyToClipBoard(String res, boolean isNotifyUser) {
        Transferable trans = new StringSelection(res);
        copyToClipBoard(trans, isNotifyUser);
    }

    private void copyToClipBoard(Transferable data, boolean isNotifyUser) {
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
        clipboard.setContents(data, null);
        if (isNotifyUser) {
            EventManagement.getInstance().putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Info"),
                    translateUtil.getTranslation("The result has been copied to the clipboard")));
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
            final int timeLimit = 50;
            long pressTime;
            boolean isFirstPress = true;
            final AllConfigs allConfigs = AllConfigs.getInstance();

            @Override
            public void keyPressed(KeyEvent arg0) {
                int key = arg0.getKeyCode();
                if (key == 8 && getSearchBarText().isEmpty()) {
                    //消除搜索框为空时按删除键发出的无效提示音
                    arg0.consume();
                    if (currentUsingPlugin != null) {
                        currentUsingPlugin = null;
                        String substring = ">" + currentPluginIdentifier.substring(0, currentPluginIdentifier.length() - 1);
                        SwingUtilities.invokeLater(() -> textField.setText(substring));
                    }
                }
                if (listResultsNum.get() != 0) {
                    if (38 == key) {
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

                                if (currentResultCount.get() >= listResultsNum.get()) {
                                    currentResultCount.set(listResultsNum.get() - 1);
                                }
                                if (currentResultCount.get() <= 0) {
                                    currentResultCount.set(0);
                                }
                                moveUpward(getCurrentLabelPos());
                            }
                        }
                    } else if (40 == key) {
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

                                    if (currentResultCount.get() >= listResultsNum.get()) {
                                        currentResultCount.set(listResultsNum.get() - 1);
                                    }
                                    if (currentResultCount.get() <= 0) {
                                        currentResultCount.set(0);
                                    }
                                    moveDownward(getCurrentLabelPos());
                                }
                            }
                        }
                    } else if (10 == key) {
                        if (isPreviewMode.get() || isTutorialMode.get()) {
                            return;
                        }
                        if (runningMode != Enums.RunningMode.PLUGIN_MODE) {
                            //enter被点击
                            clearAllLabels();
                            if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                                if (isVisible()) {
                                    setVisible(false);
                                }
                            }
                            if (listResultsNum.get() != 0) {
                                String res = listResults.get(currentResultCount.get());
                                if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                                    if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                                        if (isOpenLastFolderPressed.get()) {
                                            //打开上级文件夹
                                            openFolderByExplorer(res);
                                        } else if (allConfigs.isDefaultAdmin() || isRunAsAdminPressed.get()) {
                                            openWithAdmin(res);
                                        } else if (isCopyPathPressed.get()) {
                                            copyToClipBoard(res, true);
                                        } else {
                                            openWithoutAdmin(res);
                                        }
                                    } else if (showingMode == Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                                        if (isCopyPathPressed.get()) {
                                            copyToClipBoard(res, true);
                                        } else {
                                            quickJump(res);
                                        }
                                    }
                                } else if (runningMode == Enums.RunningMode.COMMAND_MODE) {
                                    String[] commandInfo = semicolon.split(res);
                                    boolean isExecuted = runInternalCommand(colon.split(commandInfo[0])[1]);
                                    if (isExecuted) {
                                        return;
                                    }
                                    File open = new File(commandInfo[1]);
                                    if (isOpenLastFolderPressed.get()) {
                                        //打开上级文件夹
                                        openFolderByExplorer(open.getAbsolutePath());
                                    } else if (allConfigs.isDefaultAdmin() || isRunAsAdminPressed.get()) {
                                        openWithAdmin(open.getAbsolutePath());
                                    } else if (isCopyPathPressed.get()) {
                                        copyToClipBoard(open.getAbsolutePath(), true);
                                    } else {
                                        openWithoutAdmin(open.getAbsolutePath());
                                    }
                                }
                            }
                            detectShowingModeAndClose();
                        }
                    } else if (allConfigs.getOpenLastFolderKeyCode() == key) {
                        //打开上级文件夹热键被点击
                        isOpenLastFolderPressed.set(true);
                    } else if (allConfigs.getRunAsAdminKeyCode() == key) {
                        //以管理员方式运行热键被点击
                        isRunAsAdminPressed.set(true);
                    } else if (allConfigs.getCopyPathKeyCode() == key) {
                        isCopyPathPressed.set(true);
                    }
                }
                if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                    if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                        if (key != 38 && key != 40) {
                            if (currentUsingPlugin != null) {
                                if (listResultsNum.get() != 0) {
                                    currentUsingPlugin.keyPressed(arg0, listResults.get(currentResultCount.get()));
                                }
                            }
                            if (key == 10) {
                                closeSearchBar();
                            }
                        }
                    }
                }
            }

            @Override
            public void keyReleased(KeyEvent arg0) {
                int key = arg0.getKeyCode();
                if (allConfigs.getOpenLastFolderKeyCode() == key) {
                    //复位按键状态
                    isOpenLastFolderPressed.set(false);
                } else if (allConfigs.getRunAsAdminKeyCode() == key) {
                    isRunAsAdminPressed.set(false);
                } else if (allConfigs.getCopyPathKeyCode() == key) {
                    isCopyPathPressed.set(false);
                }

                if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                    if (key != 38 && key != 40) {
                        if (currentUsingPlugin != null) {
                            if (listResultsNum.get() != 0) {
                                currentUsingPlugin.keyReleased(arg0, listResults.get(currentResultCount.get()));
                            }
                        }
                    }
                }
            }

            @Override
            public void keyTyped(KeyEvent arg0) {
                if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                    int key = arg0.getKeyCode();
                    if (key != 38 && key != 40) {
                        if (currentUsingPlugin != null) {
                            if (listResultsNum.get() != 0) {
                                currentUsingPlugin.keyTyped(arg0, listResults.get(currentResultCount.get()));
                            }
                        }
                    }
                }
            }
        });
    }

    private void openFolderByExplorerWithException(String dir) throws IOException {
        Runtime.getRuntime().exec("explorer.exe /select, \"" + dir + "\"");
    }

    private void openFolderByExplorer(String dir) {
        try {
            saveCache(dir);
            openFolderByExplorerWithException(dir);
        } catch (IOException e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(null, TranslateUtil.getInstance().getTranslation("Execute failed"));
        }
    }

    /**
     * 在command模式下，检测当前输入信息是否是软件已经定义的内部命令
     * clearbin update help version
     * return true only the internal command was executed. Otherwise false
     */
    private boolean runInternalCommand(String commandName) {
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        EventManagement eventManagement = EventManagement.getInstance();
        switch (commandName) {
            case "clearbin":
                detectShowingModeAndClose();
                if (JOptionPane.showConfirmDialog(null, translateUtil.getTranslation(
                        "Are you sure you want to empty the recycle bin")) == JOptionPane.OK_OPTION) {
                    try {
                        File[] roots = File.listRoots();
                        for (File root : roots) {
                            Runtime.getRuntime().exec("cmd.exe /c rd /s /q " + root.getAbsolutePath() + "$Recycle.Bin");
                        }
                        JOptionPane.showMessageDialog(null, translateUtil.getTranslation(
                                "Successfully empty the recycle bin"));
                    } catch (IOException e) {
                        JOptionPane.showMessageDialog(null, translateUtil.getTranslation(
                                "Failed to empty the recycle bin"));
                    }
                }
                return true;
            case "update":
                detectShowingModeAndClose();
                eventManagement.putEvent(new ShowTaskBarMessageEvent(
                        translateUtil.getTranslation("Info"),
                        translateUtil.getTranslation("Updating file index")));
                eventManagement.putEvent(new UpdateDatabaseEvent());
                startSignal.set(false);
                isNotSqlInitialized.set(false);
                return true;
            case "help":
                detectShowingModeAndClose();
                if (JOptionPane.showConfirmDialog(null, translateUtil.getTranslation("Whether to view help"))
                        == JOptionPane.OK_OPTION) {
                    isTutorialMode.set(true);
                    CachedThreadPoolUtil.getInstance().executeTask(() -> {
                        showTutorial();
                        isTutorialMode.set(false);
                    });
                }
                return true;
            case "version":
                detectShowingModeAndClose();
                JOptionPane.showMessageDialog(null, translateUtil.getTranslation(
                        "Current Version:") + AllConfigs.version);
                return true;
            default:
                return false;
        }
    }

    private void showTutorial() {
        if (isPreviewMode.get()) {
            return;
        }
        int count = 0;
        final int maxWaiting = 10;
        AtomicBoolean isCanceled = new AtomicBoolean(false);
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        //检查数据库是否正常
        if (databaseService.getStatus() != Enums.DatabaseStatus.NORMAL) {
            JFrame frame = new JFrame();
            frame.setUndecorated(true);
            frame.getRootPane().setWindowDecorationStyle(JRootPane.FRAME);
            LoadingPanel glasspane = new LoadingPanel(translateUtil.getTranslation("Please wait up to 10 seconds"));
            glasspane.setSize(600, 400);
            frame.setGlassPane(glasspane);
            glasspane.start();//开始动画加载效果
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
                while (databaseService.getStatus() != Enums.DatabaseStatus.NORMAL && count < maxWaiting) {
                    count++;
                    TimeUnit.SECONDS.sleep(1);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                glasspane.stop();
                frame.setVisible(false);
            }
        }
        if (isCanceled.get()) {
            return;
        }
        if (count == maxWaiting) {
            JOptionPane.showMessageDialog(null, translateUtil.getTranslation("Waiting overtime"));
            return;
        }
        EventManagement eventManagement = EventManagement.getInstance();
        showSearchbar(false);
        JOptionPane.showMessageDialog(searchBar, translateUtil.getTranslation("Welcome to the tutorial of File-Engine") + "\n" +
                translateUtil.getTranslation("The default Ctrl + Alt + K calls out the search bar, which can be changed in the settings.") +
                translateUtil.getTranslation("You can enter the keywords you want to search here"));
        JOptionPane.showMessageDialog(searchBar, translateUtil.getTranslation("Let's see an example"));
        textField.setText("test");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("When you enter \"test\" in the search bar") + ",\n" +
                        translateUtil.getTranslation("files with \"test\" in the name will be displayed below the search bar"));
        textField.setText("test;file");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("If you know multiple keywords of a file") + "\n" +
                        translateUtil.getTranslation("(for example, the file name contains both \"file\" and \"test\")") + ",\n" +
                        translateUtil.getTranslation("you can separate them with \";\" (semicolon) to search together as keywords."));
        textField.setText("/test");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("When entering \"/test\" in the search bar") + ", " +
                        translateUtil.getTranslation("the file containing \"test\" in the path will be displayed below the search bar"));
        textField.setText("");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("Add \":\" + suffix after the keyword to achieve a more precise search") + "\n" +
                        translateUtil.getTranslation("The program has the following four suffixes") + "\n" +
                        ":d     :f     :full     :case" + "\n" +
                        translateUtil.getTranslation("not case sensitive"));
        textField.setText("test:d");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("\":d\" is the suffix for searching only folders"));
        textField.setText("test:f");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("\":f\" is the suffix to search only for files"));
        textField.setText("test:full");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("\":full\" means full word matching, but case insensitive"));
        textField.setText("test:case");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("\":case\" means case sensitive"));
        textField.setText("test:d;full");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("You can also combine different suffixes to use") + "\n" +
                        translateUtil.getTranslation("you can separate them with \";\" (semicolon) to search together as keywords."));
        textField.setText("test;/file:d;case");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("Different keywords are separated by \";\" (semicolon), suffix and keywords are separated by \":\" (colon)"));
        //判断是否为中文
        if ("简体中文".equals(translateUtil.getLanguage())) {
            textField.setText("pinyin");
            JOptionPane.showMessageDialog(searchBar, "你可以使用拼音来代替汉字");
        }
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("Click \"Enter\" to open the file directly") + "\n" +
                        translateUtil.getTranslation("Click \"Ctrl + Enter\" to open the folder where the file is located") + "\n" +
                        translateUtil.getTranslation("Click \"Shift + Enter\" to open the file as an administrator (use with caution)") + "\n" +
                        translateUtil.getTranslation("Click \"Alt+ Enter\" to copy the file path") + "\n\n" +
                        translateUtil.getTranslation("You can modify these hotkeys in the settings"));
        textField.setText(":");
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("Enter \":\" (colon) at the front of the search box to enter the command mode") + "\n" +
                        translateUtil.getTranslation("There are built-in commands, you can also add custom commands in the settings"));
        JOptionPane.showMessageDialog(searchBar,
                translateUtil.getTranslation("If you find that some files cannot be searched, you can enter \":update\" in the search bar to rebuild the index."));
        closeSearchBar();
        eventManagement.putEvent(new ShowSettingsFrameEvent());
        JOptionPane.showMessageDialog(null,
                translateUtil.getTranslation("This is the settings window") + "\n" +
                        translateUtil.getTranslation("You can modify many settings here") + "\n" +
                        translateUtil.getTranslation("Including the color of the window, the hot key to call out the search box, the transparency of the window, custom commands and so on."));
        if (JOptionPane.showConfirmDialog(null,
                translateUtil.getTranslation("End of the tutorial") + "\n" +
                        translateUtil.getTranslation("You can enter \":help\" in the search bar at any time to enter the tutorial") + "\n" +
                        translateUtil.getTranslation("There are more detailed tutorials on the Github wiki. Would you like to check it out?"))
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
                if (!(name == null || name.isEmpty() || "filled".equals(name))) {
                    String currentText = label.getText();
                    //indexOf效率更高
                    if (currentText == null || currentText.indexOf(":\\") == -1) {
                        //当前显示的不是路径
                        label.setName(currentText);
                        label.setText(name);
                    }
                }
                label.setBackground(labelColor);
//                label.setForeground(fontColorWithCoverage);
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
                if (!(name == null || name.isEmpty() || "filled".equals(name))) {
                    String currentText = label.getText();
                    //indexOf效率更高
                    if (currentText == null || currentText.indexOf(":\\") != -1) {
                        //当前显示的不是名称
                        label.setName(currentText);
                        label.setText(name);
                    }
                }
                label.setBackground(backgroundColor);
                label.setForeground(labelFontColor);
            });
        }
    }

    /**
     * 检测鼠标在窗口的位置，并设置鼠标所在位置的label为选中
     */
    private void addSearchBarMouseMotionListener() {
        AtomicBoolean shouldSaveMousePos = new AtomicBoolean(false);
        final int minMouseMoveDistance = label1.getHeight() / 6;
        //添加一个线程不断更新鼠标保存时间
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.isNotMainExit()) {
                    shouldSaveMousePos.set(true);
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        searchBar.addMouseMotionListener(new MouseAdapter() {
            double absoluteDistance;
            int lastPositionX = 0;
            int lastPositionY = 0;

            @Override
            public void mouseDragged(MouseEvent e) {
                Enums.RunningMode mode = runningMode;
                isMouseDraggedInWindow.set(
                        mode == Enums.RunningMode.NORMAL_MODE ||
                                mode == Enums.RunningMode.COMMAND_MODE
                );
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
                    if (lastPositionX == 0 || lastPositionY == 0) {
                        lastPositionX = currentX;
                        lastPositionY = currentY;
                    }
                    //计算鼠标当前位置到上次位置的直线距离
                    absoluteDistance = Math.sqrt(Math.pow((currentX - lastPositionX), 2) + Math.pow((currentY - lastPositionY), 2));
                    if (shouldSaveMousePos.get()) {
                        //超过50毫秒，保存一次鼠标位置
                        shouldSaveMousePos.set(false);
                        lastPositionX = currentX;
                        lastPositionY = currentY;
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
                            if (mouseOnWhichLabel < listResultsNum.get()) {
                                int ret;
                                if (position < mouseOnWhichLabel) {
                                    ret = mouseOnWhichLabel - position;
                                } else {
                                    ret = -(position - mouseOnWhichLabel);
                                }
                                currentResultCount.getAndAdd(ret);
                                currentLabelSelectedPosition.getAndAdd(ret);
                                switch (mouseOnWhichLabel) {
                                    case 0:
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
                                        break;
                                    case 1:
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
                                        break;
                                    case 2:
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
                                        break;
                                    case 3:
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
                                        break;
                                    case 4:
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
                                        break;
                                    case 5:
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
                                        break;
                                    case 6:
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
                                        break;
                                    case 7:
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
                                        break;
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
            case 0:
                if (isLabelNotEmpty(label2)) {
                    isNextLabelValid = true;
                }
                break;
            case 1:
                if (isLabelNotEmpty(label3)) {
                    isNextLabelValid = true;
                }
                break;
            case 2:
                if (isLabelNotEmpty(label4)) {
                    isNextLabelValid = true;
                }
                break;
            case 3:
                if (isLabelNotEmpty(label5)) {
                    isNextLabelValid = true;
                }
                break;
            case 4:
                if (isLabelNotEmpty(label6)) {
                    isNextLabelValid = true;
                }
                break;
            case 5:
                if (isLabelNotEmpty(label7)) {
                    isNextLabelValid = true;
                }
                break;
            case 6:
                if (isLabelNotEmpty(label8)) {
                    isNextLabelValid = true;
                }
                break;
            default:
                if (listResultsNum.get() > 8) {
                    return true;
                }
        }
        return isNextLabelValid;
    }

    private boolean isLabelEmpty(JLabel label) {
        boolean isEmpty = true;
        String text;
        String name;
        if (label != null) {
            text = label.getText();
            name = label.getText();
            if (text != null && name != null) {
                isEmpty = text.isEmpty() && name.isEmpty();
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

                        if (currentResultCount.get() >= listResultsNum.get()) {
                            currentResultCount.set(listResultsNum.get() - 1);
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

                    if (currentResultCount.get() >= listResultsNum.get()) {
                        currentResultCount.set(listResultsNum.get() - 1);
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
                int size = listResultsNum.get();
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
                size = listResultsNum.get();
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
                size = listResultsNum.get();
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
                size = listResultsNum.get();
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
                size = listResultsNum.get();
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
                size = listResultsNum.get();
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
                if (runningMode == Enums.RunningMode.NORMAL_MODE) {
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
                } else if (runningMode == Enums.RunningMode.COMMAND_MODE) {
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
                } else if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
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
                break;
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
            case 0:
                size = listResultsNum.get();
                if (runningMode == Enums.RunningMode.NORMAL_MODE) {
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
                } else if (runningMode == Enums.RunningMode.COMMAND_MODE) {
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
                } else if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
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
                break;
            case 1:
                size = listResultsNum.get();
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
                break;
            case 2:
                size = listResultsNum.get();
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
                break;
            case 3:
                size = listResultsNum.get();
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
                break;
            case 4:
                size = listResultsNum.get();
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
                break;
            case 5:
                size = listResultsNum.get();
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
                break;
            case 6:
                size = listResultsNum.get();
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
                break;
            case 7:
                setLabelNotChosen(label1);
                setLabelNotChosen(label2);
                setLabelNotChosen(label3);
                setLabelNotChosen(label4);
                setLabelNotChosen(label5);
                setLabelNotChosen(label6);
                setLabelChosen(label7);
                setLabelNotChosen(label8);
                break;
        }
    }

    private void clearListAndTempAndReset() {
        listResults.clear();
        tempResults.clear();
        listResultsNum.set(0);
        tempResultNum.set(0);
    }

    //只在重新输入需要初始化所有设置时使用
    private void clearAllAndResetAll() {
        clearAllLabels();
        clearListAndTempAndReset();
        tableQueue.clear();
        firstResultStartShowingTime = 0;
        currentResultCount.set(0);
        currentLabelSelectedPosition.set(0);
        isCacheAndPrioritySearched.set(false);
    }

    /**
     * 设置当前运行模式
     *
     * @return 是否发送startTime以及开始信号
     */
    private boolean setRunningMode() {
        if (currentUsingPlugin != null) {
            return true;
        }
        String text = getSearchBarText();
        if (text == null || text.isEmpty()) {
            runningMode = Enums.RunningMode.NORMAL_MODE;
        } else {
            char first = text.charAt(0);
            if (first == ':') {
                runningMode = Enums.RunningMode.COMMAND_MODE;
            } else if (first == '>') {
                String subText = text.substring(1);
                String[] s = blank.split(subText);
                if (text.charAt(text.length() - 1) == ' ') {
                    currentUsingPlugin = PluginService.getInstance().getPluginInfoByIdentifier(s[0]).plugin;
                    if (currentUsingPlugin != null) {
                        runningMode = Enums.RunningMode.PLUGIN_MODE;
                        currentPluginIdentifier = s[0];
                        SwingUtilities.invokeLater(() -> textField.setText(""));
                    }
                    return false;
                }
            } else {
                runningMode = Enums.RunningMode.NORMAL_MODE;
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
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (IsDebug.isDebug()) {
            System.out.println();
            System.err.println("正在切换字体");
            System.out.println();
        }
        Font labelFont = label1.getFont();
        Font newFont = translateUtil.getFitFont(labelFont.getStyle(), labelFont.getSize(), testStr);
        textField.setFont(translateUtil.getFitFont(origin.getStyle(), origin.getSize(), testStr));
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

            @Override
            public void insertUpdate(DocumentEvent e) {
                changeFontOnDisplayFailed();
                clearAllAndResetAll();
                isSendSignal = setRunningMode();
                if (isSendSignal) {
                    startTime = System.currentTimeMillis();
                    startSignal.set(true);
                    isNotSqlInitialized.set(true);
                }
                if (runningMode == Enums.RunningMode.PLUGIN_MODE && currentUsingPlugin != null) {
                    currentUsingPlugin.textChanged(getSearchBarText());
                    currentUsingPlugin.clearResultQueue();
                }
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                changeFontOnDisplayFailed();
                clearAllAndResetAll();
                isSendSignal = setRunningMode();
                if (getSearchBarText().isEmpty()) {
                    listResultsNum.set(0);
                    tempResultNum.set(0);
                    currentResultCount.set(0);
                    startTime = System.currentTimeMillis();
                    startSignal.set(false);
                    isNotSqlInitialized.set(false);
                } else {
                    if (isSendSignal) {
                        startTime = System.currentTimeMillis();
                        startSignal.set(true);
                        isNotSqlInitialized.set(true);
                    }
                    if (runningMode == Enums.RunningMode.PLUGIN_MODE && currentUsingPlugin != null) {
                        currentUsingPlugin.textChanged(getSearchBarText());
                        currentUsingPlugin.clearResultQueue();
                    }
                }
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                startTime = System.currentTimeMillis();
                startSignal.set(false);
                isNotSqlInitialized.set(false);
            }
        });
    }

    private boolean isAdmin() {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("cmd.exe");
            Process process = processBuilder.start();
            PrintStream printStream = new PrintStream(process.getOutputStream(), true);
            Scanner scanner = new Scanner(process.getInputStream());
            printStream.println("@echo off");
            printStream.println(">nul 2>&1 \"%SYSTEMROOT%\\system32\\cacls.exe\" \"%SYSTEMROOT%\\system32\\config\\system\"");
            printStream.println("echo %errorlevel%");

            boolean printedErrorLevel = false;
            while (true) {
                String nextLine = scanner.nextLine();
                if (printedErrorLevel) {
                    int errorLevel = Integer.parseInt(nextLine);
                    scanner.close();
                    return errorLevel == 0;
                } else if ("echo %errorlevel%".equals(nextLine)) {
                    printedErrorLevel = true;
                }
            }
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * 开始监控磁盘文件变化
     */
    private void startMonitorDisk() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            TranslateUtil translateUtil = TranslateUtil.getInstance();
            File[] roots = File.listRoots();
            if (isAdmin()) {
                FileMonitor.INSTANCE.set_output(new File("tmp").getAbsolutePath());
                for (File root : roots) {
                    boolean isLocal = IsLocalDisk.INSTANCE.isLocalDisk(root.getAbsolutePath());
                    if (isLocal) {
                        FileMonitor.INSTANCE.monitor(root.getAbsolutePath());
                    }
                }
            } else {
                eventManagement.putEvent(new ShowTaskBarMessageEvent(
                        translateUtil.getTranslation("Warning"),
                        translateUtil.getTranslation("Not administrator, file monitoring function is turned off")));
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
    private void addSearchWaiter() {
        if (!isWaiting.get()) {
            isWaiting.set(true);
            CachedThreadPoolUtil.getInstance().executeTask(() -> {
                try {
                    while (isWaiting.get()) {
                        if (databaseService.getStatus() == Enums.DatabaseStatus.NORMAL) {
                            startTime = System.currentTimeMillis() - 500;
                            startSignal.set(true);
                            isNotSqlInitialized.set(true);
                            isWaiting.set(false);
                            return;
                        }
                        TimeUnit.MILLISECONDS.sleep(20);
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            });
        }
    }

    /**
     * 初始化线程池
     */
    private void initThreadPool() {
        lockMouseMotionThread();

        tryToShowRecordsThread();

        sendSignalAndShowCommandThread();

        switchSearchBarShowingMode();
    }

    private boolean isDark(int rgbHex) {
        Color color = new Color(rgbHex);
        return isDark(color.getRed(), color.getGreen(), color.getBlue());
    }

    /**
     * 根据RGB值判断 深色与浅色
     *
     * @return true if color is dark
     */
    private boolean isDark(int r, int g, int b) {
        return !(r * 0.299 + g * 0.578 + b * 0.114 >= 192);
    }

    @EventRegister(registerClass = ShowSearchBarEvent.class)
    private static void showSearchBarEvent(Event event) {
        ShowSearchBarEvent showSearchBarTask = (ShowSearchBarEvent) event;
        getInstance().showSearchbar(showSearchBarTask.isGrabFocus);
    }

    @EventRegister(registerClass = StartMonitorDiskEvent.class)
    private static void startMonitorDiskEvent(Event event) {
        getInstance().startMonitorDisk();
    }

    @EventRegister(registerClass = HideSearchBarEvent.class)
    private static void hideSearchBarEvent(Event event) {
        getInstance().closeSearchBar();
    }

    @EventRegister(registerClass = SetSearchBarTransparencyEvent.class)
    private static void setSearchBarTransparencyEvent(Event event) {
        SetSearchBarTransparencyEvent task1 = (SetSearchBarTransparencyEvent) event;
        getInstance().setTransparency(task1.trans);
    }

    @EventRegister(registerClass = SetBorderEvent.class)
    private static void setBorderEvent(Event event) {
        SetBorderEvent setBorderEvent = (SetBorderEvent) event;
        getInstance().setBorderColor(setBorderEvent.borderType, setBorderEvent.borderColor, setBorderEvent.borderThickness);
    }

    @EventRegister(registerClass = SetSearchBarColorEvent.class)
    private static void setSearchBarColorEvent(Event event) {
        SetSearchBarColorEvent setSearchBarColorTask = (SetSearchBarColorEvent) event;
        SearchBar searchBarInstance = getInstance();
        searchBarInstance.setSearchBarColor(setSearchBarColorTask.color);
        if (searchBarInstance.isDark(setSearchBarColorTask.color)) {
            searchBarInstance.textField.setCaretColor(Color.WHITE);
        } else {
            searchBarInstance.textField.setCaretColor(Color.BLACK);
        }
    }

    @EventRegister(registerClass = SetSearchBarLabelColorEvent.class)
    private static void setSearchBarLabelColorEvent(Event event) {
        SetSearchBarLabelColorEvent setSearchBarLabelColorTask = (SetSearchBarLabelColorEvent) event;
        getInstance().setLabelColor(setSearchBarLabelColorTask.color);
    }

    @EventRegister(registerClass = SetSearchBarDefaultBackgroundEvent.class)
    private static void setSearchBarDefaultBackgroundEvent(Event event) {
        SetSearchBarDefaultBackgroundEvent setSearchBarDefaultBackgroundTask = (SetSearchBarDefaultBackgroundEvent) event;
        getInstance().setDefaultBackgroundColor(setSearchBarDefaultBackgroundTask.color);
    }

    @EventRegister(registerClass = SetSearchBarFontColorWithCoverageEvent.class)
    private static void setSearchBarFontColorWithCoverageEvent(Event event) {
        SetSearchBarFontColorWithCoverageEvent task1 = (SetSearchBarFontColorWithCoverageEvent) event;
        getInstance().setFontColorWithCoverage(task1.color);
    }

    @EventRegister(registerClass = SetSearchBarLabelFontColorEvent.class)
    private static void setSearchBarLabelFontColorEvent(Event event) {
        SetSearchBarLabelFontColorEvent setSearchBarLabelFontColorTask = (SetSearchBarLabelFontColorEvent) event;
        getInstance().setLabelFontColor(setSearchBarLabelFontColorTask.color);
    }

    @EventRegister(registerClass = SetSearchBarFontColorEvent.class)
    private static void setSearchBarFontColorEvent(Event event) {
        SetSearchBarFontColorEvent setSearchBarFontColorTask = (SetSearchBarFontColorEvent) event;
        getInstance().setSearchBarFontColor(setSearchBarFontColorTask.color);
    }

    static class IsStartTimeSet {
        static AtomicBoolean isStartTimeSet = new AtomicBoolean(false);
    }

    @EventRegister(registerClass = PreviewSearchBarEvent.class)
    private static void previewSearchBarEvent(Event event) {
        if (isPreviewMode.get()) {
            EventManagement eventManagement = EventManagement.getInstance();
            PreviewSearchBarEvent preview = (PreviewSearchBarEvent) event;
            SearchBar searchBar = getInstance();
            eventManagement.putEvent(new SetBorderEvent(preview.borderType, preview.borderColor, preview.borderThickness));
            eventManagement.putEvent(new SetSearchBarColorEvent(preview.searchBarColor));
            eventManagement.putEvent(new SetSearchBarDefaultBackgroundEvent(preview.defaultBackgroundColor));
            eventManagement.putEvent(new SetSearchBarFontColorEvent(preview.searchBarFontColor));
            eventManagement.putEvent(new SetSearchBarFontColorWithCoverageEvent(preview.chosenLabelFontColor));
            eventManagement.putEvent(new SetSearchBarLabelColorEvent(preview.chosenLabelColor));
            eventManagement.putEvent(new SetSearchBarLabelFontColorEvent(preview.unchosenLabelFontColor));
            eventManagement.putEvent(new ShowSearchBarEvent(false));
            if (searchBar.getSearchBarText() == null || searchBar.getSearchBarText().isEmpty()) {
                CachedThreadPoolUtil.getInstance().executeTask(() -> {
                    try {
                        TimeUnit.MILLISECONDS.sleep(50);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    searchBar.textField.setText("a");
                    if (!IsStartTimeSet.isStartTimeSet.get()) {
                        IsStartTimeSet.isStartTimeSet.set(true);
                        searchBar.startTime = System.currentTimeMillis();
                        searchBar.startSignal.set(true);
                        searchBar.isNotSqlInitialized.set(true);
                    }
                });
            }
        }
    }

    @EventRegister(registerClass = StartPreviewEvent.class)
    private static void startPreviewEvent(Event event) {
        isPreviewMode.set(true);
    }

    @EventRegister(registerClass = StopPreviewEvent.class)
    private static void stopPreviewEvent(Event event) {
        isPreviewMode.set(false);
        IsStartTimeSet.isStartTimeSet.set(false);
    }

    @EventRegister(registerClass = IsSearchBarVisibleEvent.class)
    private static void isSearchBarVisibleEvent(Event event) {
        event.setReturnValue(getInstance().isVisible());
    }

    @EventRegister(registerClass = GetShowingModeEvent.class)
    private static void getShowingModeEvent(Event event) {
        event.setReturnValue(getInstance().showingMode);
    }

    @EventListener(registerClass = RestartEvent.class)
    private static void restartEvent() {
        FileMonitor.INSTANCE.stop_monitor();
    }

    /**
     * 自动切换显示模式线程
     */
    private void switchSearchBarShowingMode() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                GetHandle.INSTANCE.start();
                AllConfigs allConfigs = AllConfigs.getInstance();
                while (eventManagement.isNotMainExit()) {
                    if (showingMode == Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                        getExplorerSizeAndChangeSearchBarSizeExplorerMode();
                    } else {
                        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
                        int width = screenSize.width;
                        int height = screenSize.height;
                        int searchBarWidth = (int) (width * 0.3);
                        int searchBarHeight = (int) (height * 0.4);
                        int positionX, positionY;
                        if (isPreviewMode.get()) {
                            positionX = 50;
                            positionY = 50;
                        } else {
                            positionX = width / 2 - searchBarWidth / 2;
                            positionY = height / 2 - searchBarHeight / 3;
                        }
                        changeSearchBarSizeAndPos(positionX, positionY, searchBarWidth, searchBarHeight);
                        setTextFieldAtTop(searchBarHeight);
                    }
                    boolean isChangeToAttach = GetHandle.INSTANCE.changeToAttach();
                    boolean attachExplorer = allConfigs.isAttachExplorer();
                    if (isChangeToAttach && attachExplorer) {
                        switchToExplorerAttachMode();
                    } else {
                        if (showingMode != Enums.ShowingSearchBarMode.NORMAL_SHOWING && GetHandle.INSTANCE.changeToNormal()) {
                            switchToNormalMode();
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                GetHandle.INSTANCE.stop();
            }
        });
    }

    /**
     * 获取explorer窗口大小，并修改显示模式和大小
     */
    private void getExplorerSizeAndChangeSearchBarSizeExplorerMode() {
        double dpi = GetHandle.INSTANCE.getDpi();
        long explorerWidth = (long) (GetHandle.INSTANCE.getExplorerWidth() / dpi);
        long explorerHeight = (long) (GetHandle.INSTANCE.getExplorerHeight() / dpi);
        long explorerX = (long) (GetHandle.INSTANCE.getExplorerX() / dpi);
        long explorerY = (long) (GetHandle.INSTANCE.getExplorerY() / dpi);
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        int searchBarWidth = (int) (explorerWidth * 0.3);
        int searchBarHeight = (int) (screenSize.height * 0.4);

        int labelHeight = searchBarHeight / 9;
        //explorer窗口大于20像素才开始显示，防止误判其他系统窗口
        if (labelHeight > 20) {
            int positionX;
            int positionY;
            boolean isGrabFocus;
            if (GetHandle.INSTANCE.isDialogWindow()) {
                positionX = (int) (explorerX + (explorerWidth / 2 - searchBarWidth / 2));
                positionY = (int) (explorerY + explorerHeight - searchBarHeight + labelHeight);
                isGrabFocus = true;
            } else {
                positionX = (int) (explorerX + explorerWidth - searchBarWidth - 25);
                positionY = (int) (explorerY + explorerHeight - searchBarHeight - labelHeight);
                isGrabFocus = false;
            }
            //设置窗口大小
            changeSearchBarSizeAndPos(positionX, positionY, searchBarWidth, searchBarHeight, labelHeight);
            setLabelAtTop(searchBarHeight);
            showSearchbar(isGrabFocus);
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
            textField.setLocation(0, labelHeight * 8);
            int offset = 8 - listResultsNum.get();
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
        SwingUtilities.invokeLater(() -> {
            textField.setLocation(0, 0);
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
    private void changeSearchBarSizeAndPos(int positionX, int positionY, int searchBarWidth, int searchBarHeight, int labelHeight) {
        if (positionX != searchBar.getX()
                || positionY != searchBar.getY()
                || searchBarWidth != searchBar.getWidth()
                || searchBarHeight != searchBar.getHeight()) {
            SwingUtilities.invokeLater(() -> {
                //设置窗口大小
                searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
                //设置label大小
                int firstLabelY = label1.getY();
                setLabelSize(searchBarWidth, labelHeight, firstLabelY, label1);
                setLabelSize(searchBarWidth, labelHeight, firstLabelY + labelHeight, label2);
                setLabelSize(searchBarWidth, labelHeight, firstLabelY + labelHeight * 2, label3);
                setLabelSize(searchBarWidth, labelHeight, firstLabelY + labelHeight * 3, label4);
                setLabelSize(searchBarWidth, labelHeight, firstLabelY + labelHeight * 4, label5);
                setLabelSize(searchBarWidth, labelHeight, firstLabelY + labelHeight * 5, label6);
                setLabelSize(searchBarWidth, labelHeight, firstLabelY + labelHeight * 6, label7);
                setLabelSize(searchBarWidth, labelHeight, firstLabelY + labelHeight * 7, label8);
                //设置textField大小
                textField.setSize(searchBarWidth, labelHeight);
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
    private void changeSearchBarSizeAndPos(int positionX, int positionY, int searchBarWidth, int searchBarHeight) {
        int labelHeight = searchBarHeight / 9;
        changeSearchBarSizeAndPos(positionX, positionY, searchBarWidth, searchBarHeight, labelHeight);
    }

    private void switchToExplorerAttachMode() throws InterruptedException {
        int searchBarHeight = (int) (GetHandle.INSTANCE.getExplorerHeight() * 0.75);
        int labelHeight = searchBarHeight / 9;
        if (labelHeight > 35) {
            if (showingMode != Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                //设置字体
                Font textFieldFont = new Font(null, Font.PLAIN, getTextFieldFontSizeBySearchBarHeight(searchBarHeight));
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
                showingMode = Enums.ShowingSearchBarMode.EXPLORER_ATTACH;
                searchBar.setOpacity(1);
                TimeUnit.MILLISECONDS.sleep(150);
            }
        }
    }

    private int getTextFieldFontSizeBySearchBarHeight(int searchBarHeight) {
        return (int) ((searchBarHeight * 0.4) / 96 * 72) / 4;
    }

    private int getLabelFontSizeBySearchBarHeight(int searchBarHeight) {
        return (int) (((searchBarHeight * 0.2) / 96 * 72) / 4.5);
    }

    private void switchToNormalMode() throws InterruptedException {
        closeSearchBar();
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
        int height = screenSize.height;
        int searchBarHeight = (int) (height * 0.5);
        //设置字体
        Font labelFont = new Font(null, Font.BOLD, getLabelFontSizeBySearchBarHeight(searchBarHeight));
        Font textFieldFont = new Font(null, Font.PLAIN, getTextFieldFontSizeBySearchBarHeight(searchBarHeight));
        textField.setFont(textFieldFont);
        label1.setFont(labelFont);
        label2.setFont(labelFont);
        label3.setFont(labelFont);
        label4.setFont(labelFont);
        label5.setFont(labelFont);
        label6.setFont(labelFont);
        label7.setFont(labelFont);
        label8.setFont(labelFont);
        showingMode = Enums.ShowingSearchBarMode.NORMAL_SHOWING;
        searchBar.setOpacity(AllConfigs.getInstance().getOpacity());
        TimeUnit.MILLISECONDS.sleep(150);
    }

    private static boolean isNotContains(Collection<String> list, String record) {
        if (list.contains(record)) {
            return false;
        } else {
            synchronized (SearchBar.class) {
                if (list.contains(record)) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * 在鼠标滚轮往下滑的过程中，不检测鼠标指针的移动事件
     */
    private void lockMouseMotionThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            //锁住MouseMotion检测，阻止同时发出两个动作
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.isNotMainExit()) {
                    if (System.currentTimeMillis() - mouseWheelTime > 500) {
                        isLockMouseMotion.set(false);
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    private void tryToShowRecordsWhenHasLabelEmpty() {
        if (currentResultCount.get() < listResultsNum.get()) {
            if (
                    isLabelEmpty(label2) ||
                            isLabelEmpty(label3) ||
                            isLabelEmpty(label4) ||
                            isLabelEmpty(label5) ||
                            isLabelEmpty(label6) ||
                            isLabelEmpty(label7) ||
                            isLabelEmpty(label8)
            ) {
                //设置窗口上的文字和图片显示，键盘模式
                int pos = getCurrentLabelPos();
                var ref = new Object() {
                    boolean isLabel1Chosen = false;
                    boolean isLabel2Chosen = false;
                    boolean isLabel3Chosen = false;
                    boolean isLabel4Chosen = false;
                    boolean isLabel5Chosen = false;
                    boolean isLabel6Chosen = false;
                    boolean isLabel7Chosen = false;
                    boolean isLabel8Chosen = false;
                };
                switch (pos) {
                    case 0:
                        ref.isLabel1Chosen = true;
                        break;
                    case 1:
                        ref.isLabel2Chosen = true;
                        break;
                    case 2:
                        ref.isLabel3Chosen = true;
                        break;
                    case 3:
                        ref.isLabel4Chosen = true;
                        break;
                    case 4:
                        ref.isLabel5Chosen = true;
                        break;
                    case 5:
                        ref.isLabel6Chosen = true;
                        break;
                    case 6:
                        ref.isLabel7Chosen = true;
                        break;
                    case 7:
                        ref.isLabel8Chosen = true;
                        break;
                }
                SwingUtilities.invokeLater(() -> showResults(
                        ref.isLabel1Chosen, ref.isLabel2Chosen, ref.isLabel3Chosen, ref.isLabel4Chosen,
                        ref.isLabel5Chosen, ref.isLabel6Chosen, ref.isLabel7Chosen, ref.isLabel8Chosen
                ));
            }
        }
    }

    /**
     * 不断尝试显示结果
     */
    private void tryToShowRecordsThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            //显示结果线程
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                while (eventManagement.isNotMainExit()) {
                    if (!listResults.isEmpty()) {
                        //在结果不足8个的时候不断尝试显示
                        tryToShowRecordsWhenHasLabelEmpty();
                        String text = getSearchBarText();
                        if (text.isEmpty()) {
                            clearAllLabels();
                            clearListAndTempAndReset();
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

                        if (!listResults.isEmpty() && firstResultStartShowingTime == 0) {
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
                    TimeUnit.MILLISECONDS.sleep(250);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    private void repaint() {
        if (isPreviewMode.get()) {
            SwingUtilities.invokeLater(() -> SwingUtilities.updateComponentTreeUI(searchBar));
        }
        SwingUtilities.invokeLater(searchBar::repaint);
    }

    /**
     * 更新画面线程
     */
    private void repaintFrameThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                while (isVisible()) {
                    repaint();
                    TimeUnit.MILLISECONDS.sleep(250);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                isRepaintFrameThreadNotExist.set(true);
            }
        });
    }

    /**
     * 生成未格式化的sql
     * 第一个map中key保存未格式化的sql，value保存表名称，第二个map为搜索结果的暂时存储容器
     *
     * @return map
     */
    private LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> getNonFormattedSqlFromTableQueue() {
        if (isDatabaseUpdated.get()) {
            isDatabaseUpdated.set(false);
            initPriorityQueue();
        }
        LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>> sqlColumnMap = new LinkedHashMap<>();
        if (priorityQueue.isEmpty()) {
            return sqlColumnMap;
        }
        priorityQueue.forEach(i -> {
            LinkedHashMap<String, String> eachPriorityMap = new LinkedHashMap<>();
            tableQueue.forEach(each -> {
                String sql = "SELECT %s FROM " + each + " WHERE priority=" + i;
                eachPriorityMap.put(sql, each);
            });
            sqlColumnMap.put(eachPriorityMap, new ConcurrentSkipListSet<>());
        });
        tableQueue.clear();
        return sqlColumnMap;
    }

    private void addMergeThread(AtomicBoolean isCreated) {
        isCreated.set(true);
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                long time = System.currentTimeMillis();
                while (isVisible()) {
                    if (startTime > time) {
                        return;
                    }
                    if (isCacheAndPrioritySearched.get()) {
                        tempResults.forEach(each -> {
                            tempResultNum.decrementAndGet();
                            if (isNotContains(listResults, each)) {
                                listResults.add(each);
                                listResultsNum.incrementAndGet();
                            }
                        });
                    }
                    TimeUnit.MILLISECONDS.sleep(5);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                isCreated.set(false);
            }
        });
    }

    @EventListener(registerClass = UpdateDatabaseEvent.class)
    private static void updateDatabaseEvent() {
        getInstance().isDatabaseUpdated.set(true);
    }

    /**
     * 添加sql语句，并开始搜索
     */
    private void addSqlCommands() {
        tableQueue.clear();
        LinkedList<TableNameWeightInfo> tmpCommandList = new LinkedList<>(tableSet);
        //将tableSet通过权重排序
        tmpCommandList.sort((o1, o2) -> Long.compare(o2.weight.get(), o1.weight.get()));
        tmpCommandList.forEach(each -> {
            if (IsDebug.isDebug()) {
                System.out.println("已添加表" + each.tableName + "----权重" + each.weight.get());
            }
            tableQueue.add(each.tableName);
        });
        AtomicBoolean isMergeTempResultThreadCreated = new AtomicBoolean(false);
        CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
        if (!isMergeTempResultThreadCreated.get()) {
            addMergeThread(isMergeTempResultThreadCreated);
        }
        //每个priority用一个线程
        //按照优先级排列，key是sql和表名的对应，value是容器
        LinkedHashMap<LinkedHashMap<String, String>, ConcurrentSkipListSet<String>>
                nonFormattedSql = getNonFormattedSqlFromTableQueue();
        AtomicBoolean isResultsFull = new AtomicBoolean(false);
        Bit taskStatus = new Bit(new byte[]{0});
        Bit allTaskStatus = new Bit(new byte[]{0});
        Bit number = new Bit(new byte[]{1});

        LinkedHashMap<String, ConcurrentSkipListSet<String>> containerMap = new LinkedHashMap<>();

        ConcurrentLinkedQueue<Runnable> taskQueue = new ConcurrentLinkedQueue<>();

        //添加搜索任务
        nonFormattedSql.forEach((commandsMap, container) -> {
            for (String eachDisk : RegexUtil.comma.split(AllConfigs.getInstance().getDisks())) {
                number.shiftLeft();
                Bit currentTaskNum = new Bit(number);
                allTaskStatus.set(allTaskStatus.or(currentTaskNum));
                containerMap.put(currentTaskNum.toString(), container);
                taskQueue.add(() -> {
                    long time = System.currentTimeMillis();
                    try {
                        Set<String> sqls = commandsMap.keySet();
                        DatabaseService databaseService = DatabaseService.getInstance();
                        sqls.forEach(each -> {
                            String formattedSql;
                            if (runningMode == Enums.RunningMode.NORMAL_MODE && databaseService.getStatus() == Enums.DatabaseStatus.NORMAL) {
                                formattedSql = String.format(each, "PATH");
                                int matchedNum = searchAndAddToTempResults(System.currentTimeMillis(),
                                        formattedSql,
                                        isResultsFull,
                                        container,
                                        String.valueOf(eachDisk.charAt(0)));
                                long weight = Math.min(matchedNum, 5);
                                if (weight != 0L) {
                                    updateTableWeight(commandsMap.get(each), weight);
                                }
                                if (isResultsFull.get() || time < startTime) {
                                    throw new RuntimeException("stopped");
                                }
                            }
                        });
                    } finally {
                        //执行完后将对应的线程flag设为1
                        taskStatus.set(taskStatus.or(currentTaskNum));
                    }
                });
            }
        });

        //添加消费者线程
        int processors = Runtime.getRuntime().availableProcessors();
        processors = Math.min(processors, 4);
        for (int i = 0; i < processors; i++) {
            cachedThreadPoolUtil.executeTask(() -> {
                Runnable todo;
                while ((todo = taskQueue.poll()) != null) {
                    todo.run();
                }
            });
        }

        cachedThreadPoolUtil.executeTask(() -> {
            final long startSearchTime = System.currentTimeMillis();
            while (isVisible()) {
                try {
                    Bit tmpTaskStatus = new Bit(new byte[]{0});
                    //线程状态的记录从第二个位开始，所以初始值为2
                    Bit start = new Bit(new byte[]{1, 0});
                    int loopCount = 1;
                    int failCount = 0;
                    Bit zero = new Bit(new byte[]{0});
                    Bit one = new Bit(new byte[]{1});
                    ConcurrentSkipListSet<String> results;
                    while (tmpTaskStatus.length() != allTaskStatus.length() || !tmpTaskStatus.equals(taskStatus) || taskStatus.or(zero).equals(zero)) {
                        TimeUnit.MILLISECONDS.sleep(1);
                        if (startTime > startSearchTime) {
                            break;
                        }
                        results = containerMap.get(start.toString());
                        if (results != null) {
                            tempResults.addAll(results);
                            tempResultNum.set(tempResults.size());
                        }
                        //当线程完成，taskStatus中的位会被设置为1
                        //这时，将taskStatus和start做与运算，然后移到第一位，如果为1，则线程已完成搜索
                        Bit and = taskStatus.and(start);
                        if (((and).shiftRight(loopCount)).equals(one) || failCount > 300) {
                            // 阻塞超过100次(搜索时间超过0.3s)则跳过
                            failCount = 0;
                            results = containerMap.get(start.toString());
                            if (results != null) {
                                tempResults.addAll(results);
                                tempResultNum.set(tempResults.size());
                            }
                            tmpTaskStatus = tmpTaskStatus.or(start);
                            start.shiftLeft();
                            loopCount++;
                        } else {
                            failCount++;
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                try {
                    TimeUnit.MILLISECONDS.sleep(1);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });
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
            case 1:
                // 顶部
                component.setBorder(topBorder);
                break;
            case 2:
                // 底部
                component.setBorder(bottomBorder);
                break;
            case 3:
                // 左右
                component.setBorder(middleBorder);
                break;
            case 4:
                // 全部
                component.setBorder(fullBorder);
                break;
        }
    }

    private void chooseAndSetBorder(JComponent component, int type) {
        if (currentUsingPlugin != null && runningMode == Enums.RunningMode.PLUGIN_MODE) {
            setBorder0(component, type, pluginTopBorder, pluginBottomBorder, pluginMiddleBorder, pluginFullBorder);
        } else {
            setBorder0(component, type, topBorder, bottomBorder, middleBorder, fullBorder);
        }
    }

    private void setBorderOnVisible() {
        try {
            while (isVisible()) {
                String text = getSearchBarText();
                if (text == null || text.isEmpty()) {
                    clearAllLabelBorder();
                    chooseAndSetBorder(textField, 4);
                } else {
                    if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                        chooseAndSetBorder(textField, 1);
                    } else {
                        chooseAndSetBorder(textField, 2);
                    }
                    int resultNum = listResultsNum.get();
                    if (resultNum == 0 || resultNum == 1) {
                        if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
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
                        if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
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
                        if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
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
                        if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
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
                        if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
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
                        if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
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
                        if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
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
                        if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
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
                TimeUnit.MILLISECONDS.sleep(150);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            clearAllLabelBorder();
            isBorderThreadNotExist.set(true);
        }
    }

    private void sendSignalAndShowCommandThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            //缓存和常用文件夹搜索线程
            //每一次输入会更新一次startTime，该线程记录endTime
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                TranslateUtil translateUtil = TranslateUtil.getInstance();
                AllConfigs allConfigs = AllConfigs.getInstance();
                String[] strings;
                int length;
                String text;
                if (allConfigs.isFirstRun()) {
                    runInternalCommand("help");
                }
                while (eventManagement.isNotMainExit()) {
                    long endTime = System.currentTimeMillis();
                    text = getSearchBarText();
                    if ((endTime - startTime > 200) && isNotSqlInitialized.get() && startSignal.get()) {
                        if (!getSearchBarText().isEmpty()) {
                            isNotSqlInitialized.set(false);
                            if (databaseService.getStatus() == Enums.DatabaseStatus.NORMAL && runningMode == Enums.RunningMode.NORMAL_MODE) {
                                strings = colon.split(text);
                                length = strings.length;
                                if (length == 2) {
                                    searchCase = semicolon.split(strings[1]);
                                    searchText = strings[0];
                                } else {
                                    searchText = strings[0];
                                    searchCase = null;
                                }
                                keywords = semicolon.split(searchText);
                                searchCaseToLowerAndRemoveConflict();
                                addSqlCommands();
                            }
                        }
                    }

                    if ((endTime - startTime > 300) && startSignal.get()) {
                        startSignal.set(false); //开始搜索 计时停止
                        currentResultCount.set(0);
                        currentLabelSelectedPosition.set(0);
                        clearAllLabels();
                        if (!getSearchBarText().isEmpty()) {
                            setLabelChosen(label1);
                        }
                        if (databaseService.getStatus() == Enums.DatabaseStatus.NORMAL) {
                            if (runningMode == Enums.RunningMode.COMMAND_MODE) {
                                //去掉冒号
                                boolean isExecuted = runInternalCommand(text.substring(1).toLowerCase());
                                if (!isExecuted) {
                                    LinkedHashSet<String> cmdSet = allConfigs.getCmdSet();
                                    cmdSet.add(":clearbin;" + translateUtil.getTranslation("Clear the recycle bin"));
                                    cmdSet.add(":update;" + translateUtil.getTranslation("Update file index"));
                                    cmdSet.add(":help;" + translateUtil.getTranslation("View help"));
                                    cmdSet.add(":version;" + translateUtil.getTranslation("View Version"));
                                    String finalText = text;
                                    cmdSet.forEach(i -> {
                                        if (i.startsWith(finalText)) {
                                            listResultsNum.incrementAndGet();
                                            String result = translateUtil.getTranslation("Run command") + i;
                                            listResults.add(result);
                                        }
                                        String[] cmdInfo = semicolon.split(i);
                                        if (cmdInfo[0].equals(finalText)) {
                                            detectShowingModeAndClose();
                                            openWithoutAdmin(cmdInfo[1]);
                                        }
                                    });
                                    showResults(true, false, false, false,
                                            false, false, false, false);
                                }
                            } else if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                                //对搜索关键字赋值
                                searchPriorityFolder();
                                searchCache();
                                isCacheAndPrioritySearched.set(true);
                            } else if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                                String result;
                                while (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                                    if (currentUsingPlugin != null && (result = currentUsingPlugin.pollFromResultQueue()) != null) {
                                        if (isResultNotRepeat(result)) {
                                            listResults.add(result);
                                            listResultsNum.incrementAndGet();
                                        }
                                    }
                                    TimeUnit.MILLISECONDS.sleep(10);
                                }
                            }

                            showResults(true, false, false, false,
                                    false, false, false, false);

                        } else if (databaseService.getStatus() == Enums.DatabaseStatus.VACUUM) {
                            setLabelChosen(label1);
                            eventManagement.putEvent(new ShowTaskBarMessageEvent(translateUtil.getTranslation("Info"),
                                    translateUtil.getTranslation("Organizing database")));
                        } else if (databaseService.getStatus() == Enums.DatabaseStatus.MANUAL_UPDATE) {
                            setLabelChosen(label1);
                            eventManagement.putEvent(new ShowTaskBarMessageEvent(translateUtil.getTranslation("Info"),
                                    translateUtil.getTranslation("Updating file index") + "..."));
                        }

                        if (databaseService.getStatus() != Enums.DatabaseStatus.NORMAL) {
                            //开启线程等待搜索完成
                            addSearchWaiter();
                            clearAllLabels();
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(25);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
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

    private boolean isExist(String path) {
        return Files.exists(Path.of(path));
    }


    /**
     * 检查文件路径是否匹配所有输入规则
     *
     * @param path 文件路径
     * @return true如果满足所有条件 否则false
     */
    private boolean check(String path) {
        if (notMatched(path, true)) {
            return false;
        }
        if (searchCase == null || searchCase.length == 0) {
            return true;
        }
        for (String eachCase : searchCase) {
            switch (eachCase) {
                case "f":
                    if (!Files.isRegularFile(Path.of(path))) {
                        return false;
                    }
                    break;
                case "d":
                    if (!Files.isDirectory(Path.of(path))) {
                        return false;
                    }
                    break;
                case "full":
                    if (!searchText.equalsIgnoreCase(getFileName(path))) {
                        return false;
                    }
                    break;
                case "case":
                    if (notMatched(path, false)) {
                        return false;
                    }
            }
        }
        //所有规则均已匹配
        return true;
    }

    /**
     * 检查当前文件路径是否已被加入到listResults中
     *
     * @param result 文件路径
     * @return true如果还未被加入
     */
    private boolean isResultNotRepeat(String result) {
        return isNotContains(tempResults, result) && isNotContains(listResults, result);
    }

    /**
     * 检查文件路径是否匹配然后加入到列表
     *
     * @param path 文件路径
     */
    private void matchOnCacheAndPriorityFolder(String path, boolean isResultFromCache) {
        checkIsMatchedAndAddToList(path, false, isResultFromCache, null);
    }

    /**
     * * 检查文件路径是否匹配然后加入到列表
     *
     * @param path              文件路径
     * @param isPutToTemp       是否放到临时容器，在搜索优先文件夹和cache时为false，其他为true
     * @param isResultFromCache 是否来自缓存
     * @return true如果匹配成功
     */
    private boolean checkIsMatchedAndAddToList(String path, boolean isPutToTemp, boolean isResultFromCache, ConcurrentSkipListSet<String> container) {
        boolean ret = false;
        if (check(path)) {
            if (isExist(path)) {
                //字符串匹配通过
                ret = true;
                if (isPutToTemp) {
                    if (isNotContains(tempResults, path)) {
                        if (container == null) {
                            tempResults.add(path);
                            tempResultNum.incrementAndGet();
                        } else {
                            container.add(path);
                        }
                    }
                } else {
                    if (isNotContains(listResults, path)) {
                        if (container == null) {
                            listResultsNum.incrementAndGet();
                            listResults.add(path);
                        } else {
                            container.add(path);
                        }
                    }
                }
            } else {
                if (isResultFromCache) {
                    EventManagement.getInstance().putEvent(new DeleteFromCacheEvent(path));
                }
            }
        }
        return ret;
    }

    private void initPriorityQueue() {
        priorityQueue.clear();
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement("SELECT PRIORITY FROM priority order by priority desc;", "cache")) {
            ResultSet resultSet = pStmt.executeQuery();
            while (resultSet.next()) {
                priorityQueue.add(resultSet.getInt("PRIORITY"));
            }
        } catch (SQLException throwables) {
            throwables.printStackTrace();
        }
    }

    /**
     * 搜索数据酷并加入到tempQueue中
     *
     * @param time 开始搜索时间，用于检测用于重新输入匹配信息后快速停止
     * @param sql  sql
     */
    private int searchAndAddToTempResults(long time, String sql, AtomicBoolean isResultsFull, ConcurrentSkipListSet<String> container, String disk) {
        int count = 0;
        //结果太多则不再进行搜索
        if (isResultsFull.get() || listResultsNum.get() + tempResultNum.get() > MAX_RESULTS_COUNT || startTime > time || !isVisible()) {
            isResultsFull.set(true);
            return count;
        }
        try (PreparedStatement stmt = SQLiteUtil.getPreparedStatement(sql, disk);
             ResultSet resultSet = stmt.executeQuery()) {
            String each;
            while (resultSet.next()) {
                //结果太多则不再进行搜索
                //用户重新输入了信息
                if (isResultsFull.get() || listResultsNum.get() + tempResultNum.get() > MAX_RESULTS_COUNT || startTime > time || !isVisible()) {
                    tableQueue.clear();
                    return count;
                }
                each = resultSet.getString("PATH");
                if (databaseService.getStatus() == Enums.DatabaseStatus.NORMAL) {
                    if (checkIsMatchedAndAddToList(each, true, false, container)) {
                        count++;
                    }
                }
            }
        } catch (SQLException throwables) {
            System.err.println("error sql : " + sql);
            throwables.printStackTrace();
        }
        return count;
    }

    /**
     * 显示窗口
     *
     * @param isGrabFocus 是否强制抓取焦点
     */
    private void showSearchbar(boolean isGrabFocus) {
        SwingUtilities.invokeLater(() -> {
            if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                if (isGrabFocus) {
                    GetHandle.INSTANCE.bringSearchBarToTop();
                }
            }
            if (!isVisible()) {
                if (isGrabFocus) {
                    GetHandle.INSTANCE.bringSearchBarToTop();
                }
                setVisible(true);
                textField.requestFocusInWindow();
                textField.setCaretPosition(0);
                startTime = System.currentTimeMillis();
                visibleStartTime = startTime;
            }
            if (isBorderThreadNotExist.get()) {
                isBorderThreadNotExist.set(false);
                CachedThreadPoolUtil.getInstance().executeTask(this::setBorderOnVisible);
            }
            if (isRepaintFrameThreadNotExist.get()) {
                isRepaintFrameThreadNotExist.set(false);
                repaintFrameThread();
            }
        });
    }

    /**
     * @param label JLabel
     * @return 计算出的每个label可显示的最大字符数量
     */
    private int getMaxShowCharsNum(JLabel label) {
        int fontSize = (int) ((label.getFont().getSize() / 96.0f * 72) / 2);
        return Math.max(label.getWidth() / fontSize - 40, 20);
    }

    /**
     * 在路径中添加省略号
     *
     * @param path           path
     * @param fileName       fileName
     * @param maxShowCharNum 最多显示的字符数量
     * @param addOffset      需要增加的偏移，如前方已经有字符串占位
     * @return 生成后的字符串
     */
    private String getContractPath(String path, String fileName, int maxShowCharNum, int addOffset) {
        String[] split = RegexUtil.reverseSlash.split(path);
        StringBuilder tmpPath = new StringBuilder();
        StringBuilder lastTmpPath = new StringBuilder();
        int contractLimit = 35;
        for (int i = 0; i < split.length / 2; i++) {
            if (tmpPath.length() + lastTmpPath.length() + contractLimit * 2 + addOffset + fileName.length() + "...".length() < maxShowCharNum) {
                tmpPath.append(split[i]).append("\\");
                if (split[i].length() > contractLimit) {
                    tmpPath.append(split[i], 0, contractLimit).append("\\");
                }
                if (split[split.length - i - 1].length() > contractLimit) {
                    lastTmpPath.append("\\").append(split[split.length - i - 1], 0, contractLimit);
                }
            } else if (tmpPath.length() + lastTmpPath.length() + contractLimit + fileName.length() + "...".length() + addOffset < maxShowCharNum) {
                if (split[i].length() > contractLimit && split[split.length - i - 1].length() > contractLimit ||
                        split[i].length() > contractLimit && split[split.length - i - 1].length() <= contractLimit) {
                    tmpPath.append(split[i], 0, contractLimit).append("\\");
                } else if (split[split.length - i - 1].length() > contractLimit) {
                    lastTmpPath.append("\\").append(split[split.length - i - 1], 0, contractLimit);
                }
            } else {
                if (split[i].length() > contractLimit) {
                    tmpPath.append(split[i], 0, contractLimit).append("\\");
                }
                if (split[split.length - i - 1].length() > contractLimit) {
                    lastTmpPath.append("\\").append(split[split.length - i - 1], 0, contractLimit);
                }
                break;
            }
        }
        if (tmpPath.length() != 0 || lastTmpPath.length() != 0) {
            tmpPath.append("...").append(lastTmpPath);
            return tmpPath.toString();
        } else {
            return "";
        }
    }

    /**
     * 高亮显示
     *
     * @param html     待处理的html
     * @param keywords 高亮关键字
     * @return 处理后带html
     */
    private String highLight(String html, String[] keywords) {
        StringBuilder builder = new StringBuilder();
        for (String keyword : keywords) {
            builder.append(keyword).append("|");
        }
        // 挑出所有的中文字符
        Map<String, String> chinesePinyinMap = PinyinUtil.getChinesePinyinMap(html);
        // 转换成拼音后和keywords匹配，如果发现匹配出成功，则添加到正则表达式中
        chinesePinyinMap.entrySet()
                .stream()
                .filter(pair -> Arrays.stream(keywords)
                        .anyMatch(each -> each.toLowerCase(Locale.ROOT).indexOf(pair.getValue().toLowerCase(Locale.ROOT)) != -1))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue))
                .forEach((k, v) -> builder.append(k).append("|"));
        String pattern = builder.substring(0, builder.length() - 1);
        Pattern compile = RegexUtil.getPatter(pattern, Pattern.CASE_INSENSITIVE);
        Matcher matcher = compile.matcher(html);
        html = matcher.replaceAll((matchResult) -> {
            String group = matchResult.group();
            String s = "#" + ColorUtils.parseColorHex(fontColorWithCoverage);
            return "<span style=\"color: " + s + ";\">" + group + "</span>";
        });
        return html;
    }


    /**
     * 根据path或command生成显示html
     *
     * @param path    path
     * @param command command
     * @return html
     */
    private String getHtml(String path, String command, boolean[] isParentPathEmpty) {
        String template = "<html><body>%s</body></html>";
        isParentPathEmpty[0] = false;
        if (path == null) {
            // 命令模式
            String[] info = semicolon.split(command);
            String commandPath = info[1];
            String commandName = info[0];
            int maxShowCharNum = getMaxShowCharsNum(label1);
            if (commandPath.length() + ">>".length() > maxShowCharNum) {
                String show = getContractPath(commandPath, "", maxShowCharNum, 0);
                if (show.isEmpty()) {
                    int subNum = Math.max(0, maxShowCharNum - 10);
                    subNum = Math.min(commandPath.length(), subNum);
                    show = commandPath.substring(0, subNum) + "...";
                    commandPath = show;
                }
            }
            return String.format(template,
                    "<div>" +
                            highLight(commandName, new String[]{getSearchBarText()}) +
                            "<br><font size=\"-2\">" + "&gt;&gt;" + commandPath +
                            "</div>");
        } else if (command == null) {
            // 普通模式
            int maxShowCharNum = getMaxShowCharsNum(label1);
            String parentPath = getParentPath(path);
            String fileName = getFileName(path);
            int blankNUm = 20;
            int charNumbers = fileName.length() + parentPath.length() + 20;
            if (charNumbers > maxShowCharNum) {
                parentPath = getContractPath(parentPath, fileName, maxShowCharNum, 10);
                isParentPathEmpty[0] = parentPath.isEmpty();
            } else {
                blankNUm = Math.max(maxShowCharNum - fileName.length() - parentPath.length() - 20, 20);
            }
            return String.format(template,
                    "<div>" +
                            highLight(fileName, keywords) +
                            "<font size=\"-2\">" +
                            getBlank(blankNUm) + parentPath +
                            "</font>" +
                            "</div>");
        }
        return template.replace("%s", "");
    }

    private String getBlank(int num) {
        return "&nbsp;".repeat(Math.max(0, num));
    }

    /**
     * 在label上显示当前文件路径对应文件的信息
     *
     * @param path     文件路径
     * @param label    需要显示的label
     * @param isChosen 是否当前被选中
     */
    private void showResultOnLabel(String path, JLabel label, boolean isChosen) {
        //将文件的路径信息存储在label的名称中，在未被选中时只显示文件名，选中后才显示文件路径
        boolean[] isEmpty = new boolean[1];
        String allHtml = getHtml(path, null, isEmpty);
        if (isEmpty[0]) {
            int maxShowCharsNum = getMaxShowCharsNum(label1);
            boolean isContract = path.length() > maxShowCharsNum;
            int subNum = Math.max(0, maxShowCharsNum - "...".length() - 20);
            subNum = Math.min(path.length(), subNum);
            String showPath = isContract ? path.substring(0, subNum) : path;
            String add = isContract ? "..." : "";
            label.setName("<html><body>" + highLight(getFileName(path), keywords) + getBlank(20) + "<font size=\"-2\">" + showPath + add + "</font></body></html>");
        } else {
            label.setName("filled");
        }
        label.setText(allHtml);
        ImageIcon icon = GetIconUtil.getInstance().getBigIcon(path, iconSideLength, iconSideLength);
        label.setIcon(icon);
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
        currentUsingPlugin.showResultOnLabel(result, label, isChosen);
    }

    /**
     * 在label上显示命令信息
     *
     * @param command  命令
     * @param label    需要显示的label
     * @param isChosen 是否当前被选中
     */
    private void showCommandOnLabel(String command, JLabel label, boolean isChosen) {
        GetIconUtil getIconUtil = GetIconUtil.getInstance();
        String[] info = semicolon.split(command);
        String path = info[1];
        String name = info[0];
        String showStr = getHtml(null, command, new boolean[1]);
        label.setText(showStr);
        label.setName("filled");
        ImageIcon imageIcon = getIconUtil.getCommandIcon(colon.split(name)[1], iconSideLength, iconSideLength);
        imageIcon = imageIcon == null ? getIconUtil.getBigIcon(path, iconSideLength, iconSideLength) : imageIcon;
        label.setIcon(imageIcon);
        if (isChosen) {
            setLabelChosen(label);
        } else {
            setLabelNotChosen(label);
        }
    }

    /**
     * 用于控制8个label显示信息
     *
     * @param isLabel1Chosen label1是否被选中
     * @param isLabel2Chosen label2是否被选中
     * @param isLabel3Chosen label3是否被选中
     * @param isLabel4Chosen label4是否被选中
     * @param isLabel5Chosen label5是否被选中
     * @param isLabel6Chosen label6是否被选中
     * @param isLabel7Chosen label7是否被选中
     * @param isLabel8Chosen label8是否被选中
     */
    private void showResults(boolean isLabel1Chosen, boolean isLabel2Chosen, boolean isLabel3Chosen, boolean isLabel4Chosen,
                             boolean isLabel5Chosen, boolean isLabel6Chosen, boolean isLabel7Chosen, boolean isLabel8Chosen) {
        int size;
        if (runningMode == Enums.RunningMode.NORMAL_MODE) {
            try {
                String path;
                if (!listResults.isEmpty()) {
                    path = listResults.get(0);
                    showResultOnLabel(path, label1, isLabel1Chosen);
                }
                size = listResults.size();
                if (size > 1) {
                    path = listResults.get(1);
                    showResultOnLabel(path, label2, isLabel2Chosen);
                }
                if (size > 2) {
                    path = listResults.get(2);
                    showResultOnLabel(path, label3, isLabel3Chosen);
                }
                if (size > 3) {
                    path = listResults.get(3);
                    showResultOnLabel(path, label4, isLabel4Chosen);
                }
                if (size > 4) {
                    path = listResults.get(4);
                    showResultOnLabel(path, label5, isLabel5Chosen);
                }
                if (size > 5) {
                    path = listResults.get(5);
                    showResultOnLabel(path, label6, isLabel6Chosen);
                }
                if (size > 6) {
                    path = listResults.get(6);
                    showResultOnLabel(path, label7, isLabel7Chosen);
                }
                if (size > 7) {
                    path = listResults.get(7);
                    showResultOnLabel(path, label8, isLabel8Chosen);
                }
            } catch (IndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        } else if (runningMode == Enums.RunningMode.COMMAND_MODE) {
            try {
                String command;
                if (!listResults.isEmpty()) {
                    command = listResults.get(0);
                    showCommandOnLabel(command, label1, isLabel1Chosen);
                }
                size = listResults.size();
                if (size > 1) {
                    command = listResults.get(1);
                    showCommandOnLabel(command, label2, isLabel2Chosen);
                }
                if (size > 2) {
                    command = listResults.get(2);
                    showCommandOnLabel(command, label3, isLabel3Chosen);
                }
                if (size > 3) {
                    command = listResults.get(3);
                    showCommandOnLabel(command, label4, isLabel4Chosen);
                }
                if (size > 4) {
                    command = listResults.get(4);
                    showCommandOnLabel(command, label5, isLabel5Chosen);
                }
                if (size > 5) {
                    command = listResults.get(5);
                    showCommandOnLabel(command, label6, isLabel6Chosen);
                }
                if (size > 6) {
                    command = listResults.get(6);
                    showCommandOnLabel(command, label7, isLabel7Chosen);
                }
                if (size > 7) {
                    command = listResults.get(7);
                    showCommandOnLabel(command, label8, isLabel8Chosen);
                }
            } catch (IndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        } else if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
            try {
                String command;
                if (!listResults.isEmpty()) {
                    command = listResults.get(0);
                    showPluginResultOnLabel(command, label1, isLabel1Chosen);
                }
                size = listResults.size();
                if (size > 1) {
                    command = listResults.get(1);
                    showPluginResultOnLabel(command, label2, isLabel2Chosen);
                }
                if (size > 2) {
                    command = listResults.get(2);
                    showPluginResultOnLabel(command, label3, isLabel3Chosen);
                }
                if (size > 3) {
                    command = listResults.get(3);
                    showPluginResultOnLabel(command, label4, isLabel4Chosen);
                }
                if (size > 4) {
                    command = listResults.get(4);
                    showPluginResultOnLabel(command, label5, isLabel5Chosen);
                }
                if (size > 5) {
                    command = listResults.get(5);
                    showPluginResultOnLabel(command, label6, isLabel6Chosen);
                }
                if (size > 6) {
                    command = listResults.get(6);
                    showPluginResultOnLabel(command, label7, isLabel7Chosen);
                }
                if (size > 7) {
                    command = listResults.get(7);
                    showPluginResultOnLabel(command, label8, isLabel8Chosen);
                }
            } catch (IndexOutOfBoundsException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 清空单个label的所有信息
     *
     * @param label 需要清空的label
     */
    private void clearALabel(JLabel label) {
        label.setBackground(null);
        label.setText(null);
        label.setName(null);
        label.setIcon(null);
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
    }

    /**
     * 以管理员方式运行文件，失败则打开文件位置
     *
     * @param path 文件路径
     */
    private void openWithAdmin(String path) {
        saveCache(path);
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        File file = new File(path);
        if (file.exists()) {
            try {
                String command = file.getAbsolutePath();
                String start = "cmd.exe /c start " + command.substring(0, 2);
                String end = "\"" + command.substring(2) + "\"";
                Runtime.getRuntime().exec(start + end, null, file.getParentFile());
            } catch (IOException e) {
                //打开上级文件夹
                try {
                    openFolderByExplorerWithException(file.getAbsolutePath());
                } catch (IOException e1) {
                    JOptionPane.showMessageDialog(null, translateUtil.getTranslation("Execute failed"));
                    e.printStackTrace();
                }
            }
        } else {
            JOptionPane.showMessageDialog(null, translateUtil.getTranslation("File not exist"));
        }
    }

    /**
     * 在windows的temp目录(或者该软件的tmp目录，如果路径中没有空格)中生成bat以及用于隐藏bat的vbs脚本
     *
     * @param command    要运行的cmd命令
     * @param filePath   文件位置（必须传入文件夹）
     * @param workingDir 应用打开后的工作目录
     * @return vbs的路径
     */
    private String generateBatAndVbsFile(String command, String filePath, String workingDir) {
        char disk = workingDir.charAt(0);
        String start = workingDir.substring(0, 2);
        String end = workingDir.substring(2);
        File batFilePath = new File(filePath, "openBat_File_Engine.bat");
        File vbsFilePath = new File(filePath, "openVbs_File_Engine.vbs");
        try (BufferedWriter batW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(batFilePath), System.getProperty("sun.jnu.encoding")));
             BufferedWriter vbsW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(vbsFilePath), System.getProperty("sun.jnu.encoding")))) {
            //生成bat
            batW.write(disk + ":");
            batW.newLine();
            batW.write("cd " + start + "\"" + end + "\"");
            batW.newLine();
            batW.write(command);
            //生成vbs
            vbsW.write("set ws=createobject(\"wscript.shell\")");
            vbsW.newLine();
            vbsW.write("ws.run \"" + batFilePath + "\", 0");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return vbsFilePath.getAbsolutePath();
    }

    /**
     * 以普通权限运行文件，失败则打开文件位置
     *
     * @param path 文件路径
     */
    private void openWithoutAdmin(String path) {
        saveCache(path);
        File file = new File(path);
        String pathLower = path.toLowerCase();
        Desktop desktop;
        if (file.exists()) {
            try {
                if (pathLower.endsWith(".url")) {
                    if (Desktop.isDesktopSupported()) {
                        desktop = Desktop.getDesktop();
                        desktop.open(new File(path));
                    }
                } else if (pathLower.endsWith(".lnk")) {
                    Runtime.getRuntime().exec("explorer.exe \"" + path + "\"");
                } else {
                    String command;
                    if (file.isFile()) {
                        command = "start " + path.substring(0, 2) + "\"" + path.substring(2) + "\"";
                        String tmpDir = new File("").getAbsolutePath().indexOf(" ") != -1 ?
                                System.getProperty("java.io.tmpdir") : new File("tmp").getAbsolutePath();
                        String vbsFilePath = generateBatAndVbsFile(command, tmpDir, getParentPath(path));
                        Runtime.getRuntime().exec("explorer.exe " + vbsFilePath.substring(0, 2) + "\"" + vbsFilePath.substring(2) + "\"");
                    } else {
                        Runtime.getRuntime().exec("explorer.exe \"" + path + "\"");
                    }
                }
            } catch (Exception e) {
                //打开上级文件夹
                e.printStackTrace();
                openFolderByExplorer(path);
            }
        } else {
            JOptionPane.showMessageDialog(null, TranslateUtil.getInstance().getTranslation("File not exist"));
        }
    }

    private String getFileName(String path) {
        if (path != null) {
            int index = path.lastIndexOf(File.separator);
            return path.substring(index + 1);
        }
        return "";
    }

    /**
     * 保存当前文件路径到数据库缓存
     *
     * @param content 文件路径
     */
    private void saveCache(String content) {
        AllConfigs allConfigs = AllConfigs.getInstance();
        IsCacheExistEvent isCacheExistEvent = new IsCacheExistEvent(content);
        EventManagement eventManagement = EventManagement.getInstance();
        if (DatabaseService.getInstance().getCacheNum() < allConfigs.getCacheNumLimit()) {
            //检查缓存是否已存在
            eventManagement.putEvent(isCacheExistEvent);
            if (!eventManagement.waitForEvent(isCacheExistEvent)) {
                if (!(Boolean) isCacheExistEvent.getReturnValue()) {
                    //不存在则添加
                    eventManagement.putEvent(new AddToCacheEvent(content));
                    eventManagement.putEvent(new AddCacheEvent(content));
                }
            }
        }
    }

    /**
     * 从缓存中搜索结果并将匹配的放入listResults
     */
    private void searchCache() {
        try (PreparedStatement statement = SQLiteUtil.getPreparedStatement("SELECT PATH FROM cache;", "cache");
             ResultSet resultSet = statement.executeQuery()) {
            while (resultSet.next()) {
                String eachCache = resultSet.getString("PATH");
                matchOnCacheAndPriorityFolder(eachCache, true);
            }
        } catch (SQLException throwables) {
            throwables.printStackTrace();
        }
    }

    /**
     * 判断文件路径是否满足当前匹配结果（该方法由check（String）方法使用），检查文件路径使用check（String）方法。
     *
     * @param path         文件路径
     * @param isIgnoreCase 是否忽略大小谢
     * @return true如果匹配成功
     * @see #check(String);
     */
    private boolean notMatched(String path, boolean isIgnoreCase) {
        String matcherStrFromFilePath;
        boolean isPath;
        for (String eachKeyword : keywords) {
            if (eachKeyword == null || eachKeyword.isEmpty()) {
                continue;
            }
            if (eachKeyword.startsWith("/") || eachKeyword.startsWith(File.separator)) {
                //匹配路径
                isPath = true;
                Matcher matcher = slash.matcher(eachKeyword);
                eachKeyword = matcher.replaceAll(Matcher.quoteReplacement(File.separator));
                //获取父路径
                matcherStrFromFilePath = getParentPath(path);
            } else {
                //获取名字
                isPath = false;
                matcherStrFromFilePath = getFileName(path);
            }
            //转换大小写
            if (isIgnoreCase) {
                matcherStrFromFilePath = matcherStrFromFilePath.toLowerCase();
                eachKeyword = eachKeyword.toLowerCase();
            }
            //开始匹配
            if (matcherStrFromFilePath.indexOf(eachKeyword) == -1) {
                if (isPath) {
                    return true;
                } else {
                    if (PinyinUtil.isContainChinese(matcherStrFromFilePath)) {
                        if (PinyinUtil.toPinyin(matcherStrFromFilePath, "").indexOf(eachKeyword) == -1) {
                            return true;
                        }
                    } else {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * 搜索优先文件夹
     */
    private void searchPriorityFolder() {
        File path = new File(AllConfigs.getInstance().getPriorityFolder());
        boolean isPriorityFolderExist = path.exists();
        AtomicInteger taskNum = new AtomicInteger(0);
        if (isPriorityFolderExist) {
            ConcurrentLinkedQueue<String> listRemain = new ConcurrentLinkedQueue<>();
            File[] files = path.listFiles();
            if (null == files || files.length == 0) {
                return;
            }
            Arrays.stream(files).forEach(each -> {
                matchOnCacheAndPriorityFolder(each.getAbsolutePath(), false);
                if (each.isDirectory()) {
                    listRemain.add(each.getAbsolutePath());
                }
            });

            int cpuCores = Runtime.getRuntime().availableProcessors();
            final int threadCount = Math.min(cpuCores, 8);
            CachedThreadPoolUtil threadPoolUtil = CachedThreadPoolUtil.getInstance();
            for (int i = 0; i < threadCount; ++i) {
                threadPoolUtil.executeTask(() -> {
                    long startSearchTime = System.currentTimeMillis();
                    while (!listRemain.isEmpty()) {
                        String remain = listRemain.poll();
                        if (remain == null || remain.isEmpty()) {
                            continue;
                        }
                        File[] allFiles = new File(remain).listFiles();
                        if (allFiles == null || allFiles.length == 0) {
                            continue;
                        }

                        Arrays.stream(allFiles).forEach(each -> {
                            matchOnCacheAndPriorityFolder(each.getAbsolutePath(), false);
                            if (startTime > startSearchTime) {
                                listRemain.clear();
                                throw new RuntimeException("stopped");
                            }
                            if (each.isDirectory()) {
                                listRemain.add(each.getAbsolutePath());
                            }
                        });
                    }
                    taskNum.incrementAndGet();
                });
            }
            //等待所有线程完成
            try {
                int count = 0;
                EventManagement eventManagement = EventManagement.getInstance();
                while (taskNum.get() != threadCount) {
                    TimeUnit.MILLISECONDS.sleep(1);
                    count++;
                    if (count >= 2000 || (!eventManagement.isNotMainExit())) {
                        break;
                    }
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
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
        textField.setText("");
    }

    /**
     * 检测当前模式并重置状态
     */
    private void detectShowingModeAndClose() {
        if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
            closeSearchBar();
        } else if (showingMode == Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
            closeWithoutHideSearchBar();
        }
    }

    private void setVisible(boolean b) {
        if (!b) {
            if (!isPreviewMode.get()) {
                searchBar.setVisible(false);
            }
        } else {
            searchBar.setVisible(true);
        }
    }

    /**
     * 重置所有状态并关闭窗口
     */
    private void closeSearchBar() {
        SwingUtilities.invokeLater(() -> {
            if (!isPreviewMode.get()) {
                if (isVisible()) {
                    setVisible(false);
                }
                clearAllLabels();
                clearTextFieldText();
                resetAllStatus();
            }
            menu.setVisible(false);
        });
    }

    /**
     * 重置所有状态但不关闭窗口
     */
    private void closeWithoutHideSearchBar() {
        SwingUtilities.invokeLater(() -> {
            clearAllLabels();
            clearTextFieldText();
            resetAllStatus();
            menu.setVisible(false);
        });
    }

    private void resetAllStatus() {
        startTime = System.currentTimeMillis();//结束搜索
        currentResultCount.set(0);
        currentLabelSelectedPosition.set(0);
        clearListAndTempAndReset();
        tableQueue.clear();
        isUserPressed.set(false);
        isLockMouseMotion.set(false);
        isOpenLastFolderPressed.set(false);
        isRunAsAdminPressed.set(false);
        isCopyPathPressed.set(false);
        startSignal.set(false);
        isCacheAndPrioritySearched.set(false);
        isWaiting.set(false);
        isMouseDraggedInWindow.set(false);
        currentUsingPlugin = null;
    }

    /**
     * 判断窗口是否可见
     *
     * @return true如果可见 否则false
     */
    public boolean isVisible() {
        if (searchBar == null) {
            return false;
        }
        return searchBar.isVisible();
    }

    private String getParentPath(String path) {
        File f = new File(path);
        return f.getParentFile().getAbsolutePath();
    }

    private boolean isFile(String text) {
        File file = new File(text);
        return file.isFile();
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

    private void setBorderColor(Enums.BorderType borderType, int colorNum, int borderThickness) {
        initBorder(borderType, new Color(colorNum), borderThickness);
        textField.setBorder(fullBorder);
    }
}

