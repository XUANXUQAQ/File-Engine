package FileEngine.frames;


import FileEngine.IsDebug;
import FileEngine.configs.AllConfigs;
import FileEngine.configs.Enums;
import FileEngine.dllInterface.FileMonitor;
import FileEngine.dllInterface.GetHandle;
import FileEngine.dllInterface.IsLocalDisk;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventHandler;
import FileEngine.eventHandler.impl.database.AddToCacheEvent;
import FileEngine.eventHandler.impl.database.DeleteFromCacheEvent;
import FileEngine.eventHandler.impl.database.UpdateDatabaseEvent;
import FileEngine.eventHandler.impl.frame.searchBar.*;
import FileEngine.eventHandler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import FileEngine.eventHandler.impl.monitorDisk.StartMonitorDiskEvent;
import FileEngine.eventHandler.impl.monitorDisk.StopMonitorDiskEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarMessageEvent;
import FileEngine.utils.*;
import FileEngine.utils.database.DatabaseUtil;
import FileEngine.utils.database.SQLiteUtil;
import FileEngine.utils.moveFiles.CopyFileUtil;
import FileEngine.utils.pinyin.PinyinUtil;
import FileEngine.utils.pluginSystem.Plugin;
import FileEngine.utils.pluginSystem.PluginUtil;

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
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class SearchBar {
    private final AtomicBoolean isCacheAndPrioritySearched = new AtomicBoolean(false);
    private final AtomicBoolean isLockMouseMotion = new AtomicBoolean(false);
    private final AtomicBoolean isOpenLastFolderPressed = new AtomicBoolean(false);
    private final AtomicBoolean isRunAsAdminPressed = new AtomicBoolean(false);
    private final AtomicBoolean isCopyPathPressed = new AtomicBoolean(false);
    private final AtomicBoolean startSignal = new AtomicBoolean(false);
    private final AtomicBoolean isUserPressed = new AtomicBoolean(false);
    private final AtomicBoolean isMouseDraggedInWindow = new AtomicBoolean(false);
    private static final AtomicBoolean isPreviewMode = new AtomicBoolean(false);
    private final AtomicBoolean isTutorialMode = new AtomicBoolean(false);
    private Border border;
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
    private final CopyOnWriteArrayList<String> tempResults;  //在优先文件夹和数据库cache未搜索完时暂时保存结果，搜索完后会立即被转移到listResults
    private final ConcurrentLinkedQueue<String> commandQueue;  //保存需要被执行的sql语句
    private final CopyOnWriteArrayList<String> listResults;  //保存从数据库中找出符合条件的记录（文件路径）
    private final Set<TableNameWeightInfo> tableSet;    //保存从0-40数据库的表，使用频率和名字对应，使经常使用的表最快被搜索到
    private volatile String[] searchCase;
    private volatile String searchText;
    private volatile String[] keywords;
    private final DatabaseUtil databaseUtil;
    private final AtomicInteger listResultsNum;  //保存当前listResults中有多少个结果
    private final AtomicInteger tempResultNum;  //保存当前tempResults中有多少个结果
    private final AtomicInteger currentLabelSelectedPosition;   //保存当前是哪个label被选中 范围 0 - 7
    private volatile Plugin currentUsingPlugin;

    private static final int MAX_RESULTS_COUNT = 200;

    private static volatile SearchBar instance = null;

    private static class TableNameWeightInfo {
        private final String tableName;
        private long weight;

        private TableNameWeightInfo(String tableName, long weight) {
            this.tableName = tableName;
            this.weight = weight;
        }
    }

    private SearchBar() {
        listResults = new CopyOnWriteArrayList<>();
        tempResults = new CopyOnWriteArrayList<>();
        commandQueue = new ConcurrentLinkedQueue<>();
        tableSet = ConcurrentHashMap.newKeySet();
        searchBar = new JFrame();
        currentResultCount = new AtomicInteger(0);
        listResultsNum = new AtomicInteger(0);
        tempResultNum = new AtomicInteger(0);
        runningMode = Enums.RunningMode.NORMAL_MODE;
        showingMode = Enums.ShowingSearchBarMode.NORMAL_SHOWING;
        currentLabelSelectedPosition = new AtomicInteger(0);
        semicolon = RegexUtil.semicolon;
        colon = RegexUtil.colon;
        slash = RegexUtil.slash;
        blank = RegexUtil.blank;
        JPanel panel = new JPanel();
        Color transparentColor = new Color(0, 0, 0, 0);

        databaseUtil = DatabaseUtil.getInstance();

        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
        int width = screenSize.width;
        int height = screenSize.height;
        int searchBarWidth = (int) (width * 0.4);
        int searchBarHeight = (int) (height * 0.5);
        int positionX = width / 2 - searchBarWidth / 2;
        int positionY = height / 2 - searchBarHeight / 2;

        AllConfigs allConfigs = AllConfigs.getInstance();
        labelColor = new Color(allConfigs.getLabelColor());
        fontColorWithCoverage = new Color(allConfigs.getLabelFontColorWithCoverage());
        backgroundColor = new Color(allConfigs.getDefaultBackgroundColor());
        labelFontColor = new Color(allConfigs.getLabelFontColor());
        border = BorderFactory.createLineBorder(new Color(allConfigs.getBorderColor()));

        //frame
        searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
        searchBar.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        searchBar.setUndecorated(true);
        searchBar.getRootPane().setWindowDecorationStyle(JRootPane.NONE);
        searchBar.setBackground(transparentColor);
        searchBar.setOpacity(allConfigs.getTransparency());
        searchBar.setContentPane(panel);
        searchBar.setType(JFrame.Type.UTILITY);
        searchBar.setAlwaysOnTop(true);
        //用于C++判断是否点击了当前窗口
        searchBar.setTitle("File-Engine-SearchBar");

        //labels
        Font labelFont = new Font(Font.SANS_SERIF, Font.BOLD, (int) ((searchBarHeight * 0.2) / 96 * 72) / 4);
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

        URL icon = TaskBar.class.getResource("/icons/taskbar_32x32.png");
        Image image = new ImageIcon(icon).getImage();
        searchBar.setIconImage(image);

        //TextField
        textField = new JTextField(1000);
        textField.setSize(searchBarWidth - 6, labelHeight - 5);
        Font textFieldFont = new Font(Font.SANS_SERIF, Font.PLAIN, (int) ((searchBarHeight * 0.4) / 96 * 72) / 4);
        textField.setFont(textFieldFont);
        textField.setBorder(border);
        textField.setForeground(Color.BLACK);
        textField.setHorizontalAlignment(JTextField.LEFT);
        textField.setBackground(Color.WHITE);
        textField.setLocation(3, 0);
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

        //开启所有线程
        initThreadPool();

        //添加textField搜索变更检测
        addTextFieldDocumentListener();

        //添加结果的鼠标双击事件响应
        addSearchBarMouseListener();

        //添加结果的鼠标滚轮响应
        addSearchBarMouseWheelListener();

        //添加结果的鼠标移动事件响应
        addSearchBarMouseMotionListener();

        //添加textfield对键盘的响应
        addTextFieldKeyListener();

        addTextFieldFocusListener();
    }

    public static SearchBar getInstance() {
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
                if (System.currentTimeMillis() - visibleStartTime < 1000 && showingMode == Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                    //在explorer attach模式下 1s内窗口就获取到了焦点
                    searchBar.transferFocusBackward();
                }
            }

            @Override
            public void focusLost(FocusEvent e) {
                if (System.currentTimeMillis() - visibleStartTime > 500) {
                    if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING && allConfigs.isLoseFocusClose()) {
                        if (!(isPreviewMode.get() || isTutorialMode.get())) {
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
        for (int i = 0; i <= 40; i++) {
            tableSet.add(new TableNameWeightInfo("list" + i, 0L));
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
        origin.weight += weight;
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
     * @param width     宽
     * @param height    高
     * @param positionY Y坐标
     * @param label     需要修改大小的label
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
        EventUtil eventUtil = EventUtil.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        String lower = fileOrFolderPath.toLowerCase();
        if (lower.endsWith(".lnk") || lower.endsWith(".url")) {
            //直接复制文件
            CopyFileUtil.copyFile(new File(fileOrFolderPath), new File(writeShortCutPath));
        }else {
            File shortcutGen = new File("user/shortcutGenerator.vbs");
            String shortcutGenPath = shortcutGen.getAbsolutePath();
            String start = "cmd.exe /c start " + shortcutGenPath.substring(0, 2);
            String end = "\"" + shortcutGenPath.substring(2) + "\"";
            String commandToGenLnk = start + end + " /target:" + "\"" + fileOrFolderPath + "\"" + " " + "/shortcut:" + "\"" + writeShortCutPath + "\"" + " /workingdir:" + "\"" + fileOrFolderPath.substring(0, fileOrFolderPath.lastIndexOf(File.separator)) + "\"";
            Runtime.getRuntime().exec("cmd.exe " + commandToGenLnk);
        }
        if (isNotifyUser) {
            eventUtil.putEvent(new ShowTaskBarMessageEvent(
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
                                    saveCache(res);
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
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                if (isMouseDraggedInWindow.get()) {
                    isMouseDraggedInWindow.set(false);
                    GetHandle.INSTANCE.setExplorerPath();
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
        copyToClipBoard(result, false);
        x = GetHandle.INSTANCE.getToolbarClickX();
        y = GetHandle.INSTANCE.getToolbarClickY();
        robotUtil.mouseClicked(x, y, 1, InputEvent.BUTTON1_DOWN_MASK);
        robotUtil.keyTyped(KeyEvent.VK_CONTROL, KeyEvent.VK_V);
        robotUtil.keyTyped(KeyEvent.VK_ENTER);
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                //保证在执行粘贴操作时不会被提前恢复数据
                TimeUnit.MILLISECONDS.sleep(500);
                copyToClipBoard(originalData, false);
            } catch (InterruptedException ignored) {
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
            EventUtil.getInstance().putEvent(new ShowTaskBarMessageEvent(
                    translateUtil.getTranslation("Info"),
                    translateUtil.getTranslation("The result has been copied to the clipboard")));
        }
    }

    private String getTextFieldText() {
        return textField.getText();
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
                if (key == 8 && getTextFieldText().isEmpty()) {
                    //消除搜索框为空时按删除键发出的无效提示音
                    arg0.consume();
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

                            if (!getTextFieldText().isEmpty()) {
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
                            if (isNextLabelValid) {
                                if (!getTextFieldText().isEmpty()) {
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
                                        saveCache(res);
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
        EventUtil eventUtil = EventUtil.getInstance();
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
                eventUtil.putEvent(new ShowTaskBarMessageEvent(
                        translateUtil.getTranslation("Info"),
                        translateUtil.getTranslation("Updating file index")));
                  eventUtil.putEvent(new UpdateDatabaseEvent());
                startSignal.set(false);
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
        int count = 0;
        final int maxWaiting = 10;
        AtomicBoolean isCanceled = new AtomicBoolean(false);
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        if (databaseUtil.getStatus() != Enums.DatabaseStatus.NORMAL) {
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
                while (databaseUtil.getStatus() != Enums.DatabaseStatus.NORMAL && count < maxWaiting) {
                    count++;
                    TimeUnit.SECONDS.sleep(1);
                }
            } catch (InterruptedException ignored) {
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
        EventUtil eventUtil = EventUtil.getInstance();
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
            JOptionPane.showMessageDialog(searchBar, "你可以使用拼音来代替汉字，但拼音和英文单词之间需要用\";\"(分号)隔开（作为不同的关键字）");
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
        eventUtil.putEvent(new ShowSettingsFrameEvent());
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
                label.setBackground(labelColor);
                label.setForeground(fontColorWithCoverage);
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
                EventUtil eventUtil = EventUtil.getInstance();
                while (eventUtil.isNotMainExit()) {
                    shouldSaveMousePos.set(true);
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            }catch (InterruptedException ignored) {
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
                //判断鼠标位置
                int labelPosition = label1.getY();
                int labelPosition2 = labelPosition * 2;
                int labelPosition3 = labelPosition * 3;
                int labelPosition4 = labelPosition * 4;
                int labelPosition5 = labelPosition * 5;
                int labelPosition6 = labelPosition * 6;
                int labelPosition7 = labelPosition * 7;
                int labelPosition8 = labelPosition * 8;
                int labelPosition9 = labelPosition * 9;

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
                    if (!getTextFieldText().isEmpty()) {
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
                if (!getTextFieldText().isEmpty()) {
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
                        if (IsDebug.isDebug()) {
                            e.printStackTrace();
                        }
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
                        if (IsDebug.isDebug()) {
                            e.printStackTrace();
                        }
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
                        if (IsDebug.isDebug()) {
                            e.printStackTrace();
                        }
                    }
                }
                break;
        }
    }

    private void moveUpward(int position) {
        currentLabelSelectedPosition.decrementAndGet();
        if (currentLabelSelectedPosition.get() < 0) {
            currentLabelSelectedPosition.set(0);
        }
        int size;
        switch (position) {
            case 0:
                if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                    //到达了最上端，刷新显示
                    try {
                        String path = listResults.get(currentResultCount.get());
                        showResultOnLabel(path, label1, true);

                        path = listResults.get(currentResultCount.get() + 1);
                        showResultOnLabel(path, label2, false);

                        path = listResults.get(currentResultCount.get() + 2);
                        showResultOnLabel(path, label3, false);

                        path = listResults.get(currentResultCount.get() + 3);
                        showResultOnLabel(path, label4, false);

                        path = listResults.get(currentResultCount.get() + 4);
                        showResultOnLabel(path, label5, false);

                        path = listResults.get(currentResultCount.get() + 5);
                        showResultOnLabel(path, label6, false);

                        path = listResults.get(currentResultCount.get() + 6);
                        showResultOnLabel(path, label7, false);

                        path = listResults.get(currentResultCount.get() + 7);
                        showResultOnLabel(path, label8, false);
                    } catch (ArrayIndexOutOfBoundsException e) {
                        if (IsDebug.isDebug()) {
                            e.printStackTrace();
                        }
                    }
                } else if (runningMode == Enums.RunningMode.COMMAND_MODE) {
                    //到达了最上端，刷新显示
                    try {
                        String command = listResults.get(currentResultCount.get());
                        showCommandOnLabel(command, label1, true);

                        command = listResults.get(currentResultCount.get() + 1);
                        showCommandOnLabel(command, label2, false);

                        command = listResults.get(currentResultCount.get() + 2);
                        showCommandOnLabel(command, label3, false);

                        command = listResults.get(currentResultCount.get() + 3);
                        showCommandOnLabel(command, label4, false);

                        command = listResults.get(currentResultCount.get() + 4);
                        showCommandOnLabel(command, label5, false);

                        command = listResults.get(currentResultCount.get() + 5);
                        showCommandOnLabel(command, label6, false);

                        command = listResults.get(currentResultCount.get() + 6);
                        showCommandOnLabel(command, label7, false);

                        command = listResults.get(currentResultCount.get() + 7);
                        showCommandOnLabel(command, label8, false);
                    } catch (ArrayIndexOutOfBoundsException e) {
                        if (IsDebug.isDebug()) {
                            e.printStackTrace();
                        }
                    }
                } else if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                    try {
                        String command = listResults.get(currentResultCount.get());
                        showPluginResultOnLabel(command, label1, true);

                        command = listResults.get(currentResultCount.get() + 1);
                        showPluginResultOnLabel(command, label2, false);

                        command = listResults.get(currentResultCount.get() + 2);
                        showPluginResultOnLabel(command, label3, false);

                        command = listResults.get(currentResultCount.get() + 3);
                        showPluginResultOnLabel(command, label4, false);

                        command = listResults.get(currentResultCount.get() + 4);
                        showPluginResultOnLabel(command, label5, false);

                        command = listResults.get(currentResultCount.get() + 5);
                        showPluginResultOnLabel(command, label6, false);

                        command = listResults.get(currentResultCount.get() + 6);
                        showPluginResultOnLabel(command, label7, false);

                        command = listResults.get(currentResultCount.get() + 7);
                        showPluginResultOnLabel(command, label8, false);
                    } catch (ArrayIndexOutOfBoundsException e) {
                        if (IsDebug.isDebug()) {
                            e.printStackTrace();
                        }
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
        commandQueue.clear();
        firstResultStartShowingTime = 0;
        currentResultCount.set(0);
        currentLabelSelectedPosition.set(0);
        isCacheAndPrioritySearched.set(false);
    }

    //设置当前运行模式
    private void setRunningMode() {
        String text = getTextFieldText();
        final StringBuilder strb = new StringBuilder();
        if (text == null || text.isEmpty()) {
            runningMode = Enums.RunningMode.NORMAL_MODE;
        } else {
            char first = text.charAt(0);
            if (first == ':') {
                runningMode = Enums.RunningMode.COMMAND_MODE;
            } else if (first == '>') {
                runningMode = Enums.RunningMode.PLUGIN_MODE;
                String subText = text.substring(1);
                String[] s = blank.split(subText);
                currentUsingPlugin = PluginUtil.getInstance().getPluginInfoByIdentifier(s[0]).plugin;
                int length = s.length;
                if (currentUsingPlugin != null && length > 1) {
                    for (int i = 1; i < length - 1; ++i) {
                        strb.append(s[i]).append(" ");
                    }
                    strb.append(s[length - 1]);
                    currentUsingPlugin.textChanged(strb.toString());
                    currentUsingPlugin.clearResultQueue();
                    strb.delete(0, strb.length());
                }
            } else {
                runningMode = Enums.RunningMode.NORMAL_MODE;
            }
        }
    }

    private void changeFontOnDisplayFailed() {
        String testStr = getTextFieldText();
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
            @Override
            public void insertUpdate(DocumentEvent e) {
                changeFontOnDisplayFailed();
                clearAllAndResetAll();
                setRunningMode();
                startTime = System.currentTimeMillis();
                startSignal.set(true);
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                changeFontOnDisplayFailed();
                clearAllAndResetAll();
                setRunningMode();
                if (getTextFieldText().isEmpty()) {
                    listResultsNum.set(0);
                    tempResultNum.set(0);
                    currentResultCount.set(0);
                    startTime = System.currentTimeMillis();
                    startSignal.set(false);
                } else {
                    startTime = System.currentTimeMillis();
                    startSignal.set(true);
                }
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                startTime = System.currentTimeMillis();
                startSignal.set(false);
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

    private void startMonitorDisk() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventUtil eventUtil = EventUtil.getInstance();
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
                eventUtil.putEvent(new ShowTaskBarMessageEvent(
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

    private void addSearchWaiter() {
        if (!isWaiting.get()) {
            isWaiting.set(true);
            CachedThreadPoolUtil.getInstance().executeTask(() -> {
                try {
                    while (isWaiting.get()) {
                        if (databaseUtil.getStatus() == Enums.DatabaseStatus.NORMAL) {
                            startTime = System.currentTimeMillis() - 500;
                            startSignal.set(true);
                            isWaiting.set(false);
                            break;
                        }
                        TimeUnit.MILLISECONDS.sleep(20);
                    }
                } catch (InterruptedException ignored) {
                }
            });
        }
    }

    private void initThreadPool() {
        lockMouseMotionThread();

        tryToShowRecordsThread();

        repaintFrameThread();

        sendSignalAndShowCommandThread();

        switchSearchBarShowingMode();

        changeSearchBarSize();
    }

    public static void registerEventHandler() {
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.register(ShowSearchBarEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                ShowSearchBarEvent showSearchBarTask = (ShowSearchBarEvent) event;
                getInstance().showSearchbar(showSearchBarTask.isGrabFocus);
            }
        });

        eventUtil.register(StartMonitorDiskEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().startMonitorDisk();
            }
        });

        eventUtil.register(StopMonitorDiskEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                FileMonitor.INSTANCE.stop_monitor();
            }
        });

        eventUtil.register(HideSearchBarEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().closeSearchBar();
            }
        });

        eventUtil.register(SetSearchBarTransparencyEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                SetSearchBarTransparencyEvent task1 = (SetSearchBarTransparencyEvent) event;
                getInstance().setTransparency(task1.trans);
            }
        });

        eventUtil.register(SetBorderColorEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                SetBorderColorEvent setBorderColorTask = (SetBorderColorEvent) event;
                getInstance().setBorderColor(setBorderColorTask.color);
            }
        });

        eventUtil.register(SetSearchBarColorEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                SetSearchBarColorEvent setSearchBarColorTask = (SetSearchBarColorEvent) event;
                getInstance().setSearchBarColor(setSearchBarColorTask.color);
            }
        });

        eventUtil.register(SetSearchBarLabelColorEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                SetSearchBarLabelColorEvent setSearchBarLabelColorTask = (SetSearchBarLabelColorEvent) event;
                getInstance().setLabelColor(setSearchBarLabelColorTask.color);
            }
        });

        eventUtil.register(SetSearchBarDefaultBackgroundEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                SetSearchBarDefaultBackgroundEvent setSearchBarDefaultBackgroundTask = (SetSearchBarDefaultBackgroundEvent) event;
                getInstance().setDefaultBackgroundColor(setSearchBarDefaultBackgroundTask.color);
            }
        });

        eventUtil.register(SetSearchBarFontColorWithCoverageEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                SetSearchBarFontColorWithCoverageEvent task1 = (SetSearchBarFontColorWithCoverageEvent) event;
                getInstance().setFontColorWithCoverage(task1.color);
            }
        });

        eventUtil.register(SetSearchBarLabelFontColorEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                SetSearchBarLabelFontColorEvent setSearchBarLabelFontColorTask = (SetSearchBarLabelFontColorEvent) event;
                getInstance().setLabelFontColor(setSearchBarLabelFontColorTask.color);
            }
        });

        eventUtil.register(SetSearchBarFontColorEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                SetSearchBarFontColorEvent setSearchBarFontColorTask = (SetSearchBarFontColorEvent) event;
                getInstance().setSearchBarFontColor(setSearchBarFontColorTask.color);
            }
        });

        eventUtil.register(PreviewSearchBarEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                PreviewSearchBarEvent preview = (PreviewSearchBarEvent) event;
                eventUtil.putEvent(new SetPreviewOrNormalMode(true));
                eventUtil.putEvent(new SetBorderColorEvent(preview.borderColor));
                eventUtil.putEvent(new SetSearchBarColorEvent(preview.searchBarColor));
                eventUtil.putEvent(new SetSearchBarDefaultBackgroundEvent(preview.defaultBackgroundColor));
                eventUtil.putEvent(new SetSearchBarFontColorEvent(preview.searchBarFontColor));
                eventUtil.putEvent(new SetSearchBarFontColorWithCoverageEvent(preview.chosenLabelFontColor));
                eventUtil.putEvent(new SetSearchBarLabelColorEvent(preview.chosenLabelColor));
                eventUtil.putEvent(new SetSearchBarLabelFontColorEvent(preview.unchosenLabelFontColor));
                eventUtil.putEvent(new ShowSearchBarEvent(false));
                getInstance().textField.setText("a");
            }
        });

        eventUtil.register(SetPreviewOrNormalMode.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                SetPreviewOrNormalMode mode = (SetPreviewOrNormalMode) event;
                isPreviewMode.set(mode.isPreview);
            }
        });
    }

    private void switchSearchBarShowingMode() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventUtil eventUtil = EventUtil.getInstance();
                GetHandle.INSTANCE.start();
                while (eventUtil.isNotMainExit()) {
                    if (GetHandle.INSTANCE.isExplorerAtTop()) {
                        switchToExplorerAttachMode();
                    } else {
                        if (GetHandle.INSTANCE.isDialogNotExist() || GetHandle.INSTANCE.isExplorerAndSearchbarNotFocused()) {
                            switchToNormalMode();
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(150);
                }
            } catch (InterruptedException ignored) {
            } finally {
                GetHandle.INSTANCE.stop();
            }
        });
    }

    private void getExplorerSizeAndChangeSearchBarSizeExplorerMode() {
        int searchBarHeight = (int) (GetHandle.INSTANCE.getExplorerHeight() * 0.75);
        int labelHeight = searchBarHeight / 9;
        if (labelHeight > 20) {
            int searchBarWidth = (int) (GetHandle.INSTANCE.getExplorerWidth() / 3);
            int positionX = (int) (GetHandle.INSTANCE.getExplorerX());
            int positionY = (int) (GetHandle.INSTANCE.getExplorerY() - labelHeight - 5);
            //设置窗口大小
            changeSearchBarSizeAndPos(positionX, positionY, searchBarWidth, searchBarHeight, labelHeight);
        }
    }

    private void changeSearchBarSize() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventUtil eventUtil = EventUtil.getInstance();
                while (eventUtil.isNotMainExit()) {
                    if (isPreviewMode.get()) {
                        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
                        int width = screenSize.width;
                        int height = screenSize.height;
                        int positionX = 50;
                        int positionY = 50;
                        int searchBarWidth = width / 4;
                        int searchBarHeight = (int) (height * 0.5);
                        changeSearchBarSizeAndPos(positionX, positionY, searchBarWidth, searchBarHeight);
                    } else {
                        if (showingMode == Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                            getExplorerSizeAndChangeSearchBarSizeExplorerMode();
                        } else if (showingMode == Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                            Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
                            int width = screenSize.width;
                            int height = screenSize.height;
                            int searchBarWidth = (int) (width * 0.4);
                            int searchBarHeight = (int) (height * 0.5);
                            int positionX = width / 2 - searchBarWidth / 2;
                            int positionY = height / 2 - searchBarHeight / 2;
                            changeSearchBarSizeAndPos(positionX, positionY, searchBarWidth, searchBarHeight);
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(5);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void changeSearchBarSizeAndPos(int positionX, int positionY, int searchBarWidth, int searchBarHeight, int labelHeight) {
        if (positionX != searchBar.getX()
                || positionY != searchBar.getY()
                || searchBarWidth != searchBar.getWidth()
                || searchBarHeight != searchBar.getHeight()) {
            //设置窗口大小
            searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
            //设置label大小
            setLabelSize(searchBarWidth, labelHeight, labelHeight, label1);
            setLabelSize(searchBarWidth, labelHeight, labelHeight * 2, label2);
            setLabelSize(searchBarWidth, labelHeight, labelHeight * 3, label3);
            setLabelSize(searchBarWidth, labelHeight, labelHeight * 4, label4);
            setLabelSize(searchBarWidth, labelHeight, labelHeight * 5, label5);
            setLabelSize(searchBarWidth, labelHeight, labelHeight * 6, label6);
            setLabelSize(searchBarWidth, labelHeight, labelHeight * 7, label7);
            setLabelSize(searchBarWidth, labelHeight, labelHeight * 8, label8);
            //设置textField大小
            textField.setSize(searchBarWidth - 6, labelHeight - 5);
            textField.setLocation(3, 0);
        }
    }

    private void changeSearchBarSizeAndPos(int positionX, int positionY, int searchBarWidth, int searchBarHeight) {
        int labelHeight = searchBarHeight / 9;
        changeSearchBarSizeAndPos(positionX, positionY, searchBarWidth, searchBarHeight, labelHeight);
    }

    private void switchToExplorerAttachMode() {
        int searchBarHeight = (int) (GetHandle.INSTANCE.getExplorerHeight() * 0.75);
        int labelHeight = searchBarHeight / 9;
        if (labelHeight > 35) {
            if (showingMode != Enums.ShowingSearchBarMode.EXPLORER_ATTACH) {
                //设置字体
                Font textFieldFont = new Font(null, Font.PLAIN, (int) ((searchBarHeight * 0.4) / 96 * 72) / 4);
                textField.setFont(textFieldFont);
                Font labelFont = new Font(null, Font.BOLD, (int) ((searchBarHeight * 0.2) / 96 * 72) / 4);
                label1.setFont(labelFont);
                label2.setFont(labelFont);
                label3.setFont(labelFont);
                label4.setFont(labelFont);
                label5.setFont(labelFont);
                label6.setFont(labelFont);
                label7.setFont(labelFont);
                label8.setFont(labelFont);
                showingMode = Enums.ShowingSearchBarMode.EXPLORER_ATTACH;
                getExplorerSizeAndChangeSearchBarSizeExplorerMode();
                if (!isVisible()) {
                    showSearchbar(false);
                }
            }
        }
    }

    private void switchToNormalMode() {
        if (showingMode != Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
            closeSearchBar();
            Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
            int height = screenSize.height;
            int searchBarHeight = (int) (height * 0.5);
            //设置字体
            Font labelFont = new Font(null, Font.BOLD, (int) ((searchBarHeight * 0.2) / 96 * 72) / 4);
            Font textFieldFont = new Font(null, Font.PLAIN, (int) ((searchBarHeight * 0.4) / 96 * 72) / 4);
            textField.setFont(textFieldFont);
            label1.setFont(labelFont);
            label2.setFont(labelFont);
            label3.setFont(labelFont);
            label4.setFont(labelFont);
            label5.setFont(labelFont);
            label6.setFont(labelFont);
            label7.setFont(labelFont);
            label8.setFont(labelFont);
            showingMode= Enums.ShowingSearchBarMode.NORMAL_SHOWING;
        }
    }

    private boolean isNotContains(Collection<String> list, String record) {
        if (list.contains(record)) {
            return false;
        } else {
            synchronized (this) {
                if (list.contains(record)) {
                    return false;
                }
            }
            return true;
        }
    }

    private void lockMouseMotionThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            //锁住MouseMotion检测，阻止同时发出两个动作
            try {
                EventUtil eventUtil = EventUtil.getInstance();
                while (eventUtil.isNotMainExit()) {
                    if (System.currentTimeMillis() - mouseWheelTime > 500) {
                        isLockMouseMotion.set(false);
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void tryToShowRecordsWhenHasLabelEmpty() {
        boolean isLabel1Chosen = false;
        boolean isLabel2Chosen = false;
        boolean isLabel3Chosen = false;
        boolean isLabel4Chosen = false;
        boolean isLabel5Chosen = false;
        boolean isLabel6Chosen = false;
        boolean isLabel7Chosen = false;
        boolean isLabel8Chosen = false;
        if (currentResultCount.get() < listResultsNum.get()) {
            int pos = getCurrentLabelPos();
            switch (pos) {
                case 0:
                    isLabel1Chosen = true;
                    break;
                case 1:
                    isLabel2Chosen = true;
                    break;
                case 2:
                    isLabel3Chosen = true;
                    break;
                case 3:
                    isLabel4Chosen = true;
                    break;
                case 4:
                    isLabel5Chosen = true;
                    break;
                case 5:
                    isLabel6Chosen = true;
                    break;
                case 6:
                    isLabel7Chosen = true;
                    break;
                case 7:
                    isLabel8Chosen = true;
                    break;
            }
            //在结果不足8个的时候不断尝试显示
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
                boolean finalIsLabel1Chosen = isLabel1Chosen;
                boolean finalIsLabel2Chosen = isLabel2Chosen;
                boolean finalIsLabel3Chosen = isLabel3Chosen;
                boolean finalIsLabel4Chosen = isLabel4Chosen;
                boolean finalIsLabel5Chosen = isLabel5Chosen;
                boolean finalIsLabel6Chosen = isLabel6Chosen;
                boolean finalIsLabel7Chosen = isLabel7Chosen;
                boolean finalIsLabel8Chosen = isLabel8Chosen;
                SwingUtilities.invokeLater(() -> showResults(
                        finalIsLabel1Chosen, finalIsLabel2Chosen, finalIsLabel3Chosen, finalIsLabel4Chosen,
                        finalIsLabel5Chosen, finalIsLabel6Chosen, finalIsLabel7Chosen, finalIsLabel8Chosen
                ));
            }
        }
    }

    private void tryToShowRecordsThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            //显示结果线程
            try {
                EventUtil eventUtil = EventUtil.getInstance();
                while (eventUtil.isNotMainExit()) {
                    tryToShowRecordsWhenHasLabelEmpty();
                    String text = getTextFieldText();
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
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void repaintFrameThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventUtil eventUtil = EventUtil.getInstance();
                while (eventUtil.isNotMainExit()) {
                    if (isVisible()) {
                        SwingUtilities.invokeLater(() -> {
                            if (isPreviewMode.get()) {
                                SwingUtilities.updateComponentTreeUI(searchBar);
                            } else {
                                searchBar.repaint();
                            }
                        });
                    }
                    TimeUnit.MILLISECONDS.sleep(250);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void addSqlCommands() {
        LinkedList<TableNameWeightInfo> tmpCommandList = new LinkedList<>(tableSet);
        tmpCommandList.sort((o1, o2) -> Long.compare(o2.weight, o1.weight));
        for (TableNameWeightInfo each : tmpCommandList) {
            if (IsDebug.isDebug()) {
                System.out.println("已添加表" + each.tableName + "----权重" + each.weight);
            }
            commandQueue.add(each.tableName);
        }
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            String column;
            while (!commandQueue.isEmpty()) {
                column = commandQueue.poll();
                if (
                        runningMode == Enums.RunningMode.NORMAL_MODE &&
                        DatabaseUtil.getInstance().getStatus() == Enums.DatabaseStatus.NORMAL &&
                        column != null
                ) {
                    int matchedNum = searchAndAddToTempResults(System.currentTimeMillis(), getMaxPriority(), column);
                    long weight = Math.min(matchedNum, 5);
                    if (weight != 0L) {
                        updateTableWeight(column, weight);
                    }
                } else {
                    commandQueue.add(column);
                }
            }
            //去重
            listResults.addAll(new LinkedHashSet<>(tempResults));
            tempResults.clear();
            listResultsNum.addAndGet(tempResultNum.get());
            tempResultNum.set(0);
        });
    }

    private void sendSignalAndShowCommandThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            //缓存和常用文件夹搜索线程
            //停顿时间0.5s，每一次输入会更新一次startTime，该线程记录endTime
            try {
                EventUtil eventUtil = EventUtil.getInstance();
                TranslateUtil translateUtil = TranslateUtil.getInstance();
                AllConfigs allConfigs = AllConfigs.getInstance();
                if (allConfigs.isFirstRun()) {
                    runInternalCommand("help");
                }
                while (eventUtil.isNotMainExit()) {
                    long endTime = System.currentTimeMillis();
                    if ((endTime - startTime > 250) && startSignal.get()) {
                        startSignal.set(false); //开始搜索 计时停止
                        currentResultCount.set(0);
                        currentLabelSelectedPosition.set(0);
                        clearAllLabels();
                        if (!getTextFieldText().isEmpty()) {
                            setLabelChosen(label1);
                        }
                        clearListAndTempAndReset();
                        String text = getTextFieldText();
                        if (databaseUtil.getStatus() == Enums.DatabaseStatus.NORMAL) {
                            if (runningMode == Enums.RunningMode.COMMAND_MODE) {
                                //去掉冒号
                                boolean isExecuted = runInternalCommand(text.substring(1).toLowerCase());
                                if (!isExecuted) {
                                    LinkedHashSet<String> cmdSet = new LinkedHashSet<>(allConfigs.getCmdSet());
                                    cmdSet.add(":clearbin;" + translateUtil.getTranslation("Clear the recycle bin"));
                                    cmdSet.add(":update;" + translateUtil.getTranslation("Update file index"));
                                    cmdSet.add(":help;" + translateUtil.getTranslation("View help"));
                                    cmdSet.add(":version;" + translateUtil.getTranslation("View Version"));
                                    for (String i : cmdSet) {
                                        if (i.startsWith(text)) {
                                            listResultsNum.incrementAndGet();
                                            String result = translateUtil.getTranslation("Run command") + i;
                                            listResults.add(result);
                                        }
                                        String[] cmdInfo = semicolon.split(i);
                                        if (cmdInfo[0].equals(text)) {
                                            detectShowingModeAndClose();
                                            openWithoutAdmin(cmdInfo[1]);
                                        }
                                    }
                                    showResults(true, false, false, false,
                                            false, false, false, false);
                                }
                            } else if (runningMode == Enums.RunningMode.NORMAL_MODE) {
                                //对搜索关键字赋值
                                String[] strings;
                                int length;
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
                                isCacheAndPrioritySearched.set(false);
                                searchPriorityFolder();
                                searchCache();
                                isCacheAndPrioritySearched.set(true);
                            } else if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                                String result;
                                if (currentUsingPlugin != null) {
                                    while (runningMode == Enums.RunningMode.PLUGIN_MODE) {
                                        if (currentUsingPlugin != null) {
                                            if ((result = currentUsingPlugin.pollFromResultQueue()) != null) {
                                                if (isResultNotRepeat(result)) {
                                                    listResults.add(result);
                                                    listResultsNum.incrementAndGet();
                                                }
                                            }
                                        }
                                        TimeUnit.MILLISECONDS.sleep(10);
                                    }
                                }
                            }

                            showResults(true, false, false, false,
                                    false, false, false, false);

                        } else if (databaseUtil.getStatus() == Enums.DatabaseStatus.VACUUM) {
                            setLabelChosen(label1);
                            eventUtil.putEvent(new ShowTaskBarMessageEvent(translateUtil.getTranslation("Info"),
                                    translateUtil.getTranslation("Organizing database")));
                        } else if (databaseUtil.getStatus() == Enums.DatabaseStatus.MANUAL_UPDATE) {
                            setLabelChosen(label1);
                            eventUtil.putEvent(new ShowTaskBarMessageEvent(translateUtil.getTranslation("Info"),
                                    translateUtil.getTranslation("Updating file index") + "..."));
                        }

                        if (databaseUtil.getStatus() != Enums.DatabaseStatus.NORMAL) {
                            //开启线程等待搜索完成
                            addSearchWaiter();
                            clearAllLabels();
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException ignored) {
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
        if (list.contains("f") && list.contains("d")) {
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
        File f = new File(path);
        return f.exists();
    }


    /**
     * 检查文件路径是否匹配所有输入规则
     *
     * @param path 文件路径
     * @return true如果满足所有条件 否则false
     */
    private boolean check(String path) {
        if (searchCase == null || searchCase.length == 0) {
            return isMatched(path, true);
        } else {
            for (String eachCase : searchCase) {
                switch (eachCase) {
                    case "f":
                        if (!isMatched(path, true) || !isFile(path)) {
                            return false;
                        }
                        break;
                    case "d":
                        if (!isMatched(path, true) || !isDirectory(path)) {
                            return false;
                        }
                        break;
                    case "full":
                        if (!isMatched(path, true) || !getFileName(path).equalsIgnoreCase(searchText)) {
                            return false;
                        }
                        break;
                    case "case":
                        if (!isMatched(path, false)) {
                            return false;
                        }
                }
            }
            //所有规则均已匹配
            return true;
        }
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
     * @param path        文件路径
     */
    private void matchOnCacheAndPriorityFolder(String path, boolean isResultFromCache) {
        checkIsMatchedAndAddToList(path, 0, 0, false, isResultFromCache);
    }

    /**
     * * 检查文件路径是否匹配然后加入到列表
     * @param path  文件路径
     * @param priority 根据后缀得出的优先级（如果isPutToTemp为false，该参数会被忽略）
     * @param maxPriority 最大优先级
     * @param isPutToTemp 是否放到临时容器，在搜索优先文件夹和cache时为false，其他为true
     * @param isResultFromCache 是否来自缓存
     * @return true如果匹配成功
     */
    private boolean checkIsMatchedAndAddToList(String path, int priority, int maxPriority, boolean isPutToTemp, boolean isResultFromCache) {
        boolean ret = false;
        int offset = Math.max(maxPriority - priority, 0);
        if (check(path)) {
            if (isExist(path)) {
                //字符串匹配通过
                ret  = true;
                if (isPutToTemp) {
                    if (priority == maxPriority) {
                        tempResults.add(path);
                    } else {
                        try {
                            tempResults.add(offset, path);
                        } catch (IndexOutOfBoundsException e) {
                            tempResults.add(path);
                        }
                    }
                    tempResultNum.incrementAndGet();
                } else {
                    if (isNotContains(listResults, path)) {
                        listResultsNum.incrementAndGet();
                        listResults.add(path);
                    }
                }
            } else {
                if (isResultFromCache) {
                    EventUtil.getInstance().putEvent(new DeleteFromCacheEvent(path));
                }
            }
        }
        return ret;
    }

    private int getMaxPriority() {
        int max = 0;
        try (PreparedStatement pStmt = SQLiteUtil.getPreparedStatement("SELECT * FROM priority;")) {
            ResultSet resultSet = pStmt.executeQuery();
            while (resultSet.next()) {
                int priority = resultSet.getInt("PRIORITY");
                if (priority > max) {
                    max = priority;
                }
            }
        } catch (SQLException throwables) {
            throwables.printStackTrace();
        }
        return max;
    }

    /**
     * 搜索数据酷并加入到tempQueue中
     *
     * @param time   开始搜索时间，用于检测用于重新输入匹配信息后快速停止
     * @param eachColumn 数据库表
     */
    private int searchAndAddToTempResults(long time, int maxPriority, String eachColumn) {
        int count = 0;
        //结果太多则不再进行搜索
        if (listResultsNum.get() + tempResultNum.get() > MAX_RESULTS_COUNT || startTime > time) {
            commandQueue.clear();
            return count;
        }
        String sql;
        //为label添加结果
        sql = "SELECT PRIORITY, PATH FROM " + eachColumn + " ORDER BY PRIORITY desc;";

        try (PreparedStatement stmt = SQLiteUtil.getPreparedStatement(sql);
             ResultSet resultSet = stmt.executeQuery()) {

            String each;
            while (resultSet.next()) {
                //结果太多则不再进行搜索
                //用户重新输入了信息
                if (listResultsNum.get() + tempResultNum.get() > MAX_RESULTS_COUNT || startTime > time) {
                    commandQueue.clear();
                    return count;
                }
                each = resultSet.getString("PATH");
                int priority = resultSet.getInt("PRIORITY");
                if (databaseUtil.getStatus() == Enums.DatabaseStatus.NORMAL) {
                    if (checkIsMatchedAndAddToList(each, priority, maxPriority, true, false)) {
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
            searchBar.setAutoRequestFocus(isGrabFocus);
            setVisible(true);
            if (isGrabFocus) {
                //使用鼠标点击窗口以获得焦点
                int X = searchBar.getX() + (textField.getWidth() / 2);
                int Y = searchBar.getY() + (textField.getHeight() / 2);
                RobotUtil.getInstance().mouseClicked(X, Y, 1, InputEvent.BUTTON1_DOWN_MASK);
            }
            textField.setCaretPosition(0);
            startTime = System.currentTimeMillis();
            visibleStartTime = startTime;
        });
    }

    /**
     * 在label上显示当前文件路径对应文件的信息
     *
     * @param path     文件路径
     * @param label    需要显示的label
     * @param isChosen 是否当前被选中
     */
    private void showResultOnLabel(String path, JLabel label, boolean isChosen) {
        String name = getFileName(path);
        if (name.length() >= 32) {
            name = name.substring(0, 32) + "...";
        }
        label.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>" + getParentPath(path) + "</body></html>");
        ImageIcon icon = GetIconUtil.getInstance().getBigIcon(path, iconSideLength, iconSideLength);
        label.setIcon(icon);
        label.setBorder(border);
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
        ImageIcon imageIcon = getIconUtil.getCommandIcon(colon.split(name)[1], iconSideLength, iconSideLength);
        if (imageIcon == null) {
            imageIcon = getIconUtil.getBigIcon(path, iconSideLength, iconSideLength);
        }
        label.setIcon(imageIcon);
        label.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>" + path + "</font></body></html>");
        if (isChosen) {
            setLabelChosen(label);
        } else {
            setLabelNotChosen(label);
        }
        label.setBorder(border);
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
        if (runningMode == Enums.RunningMode.NORMAL_MODE) {
            try {
                String path = listResults.get(0);
                showResultOnLabel(path, label1, isLabel1Chosen);

                path = listResults.get(1);
                showResultOnLabel(path, label2, isLabel2Chosen);

                path = listResults.get(2);
                showResultOnLabel(path, label3, isLabel3Chosen);

                path = listResults.get(3);
                showResultOnLabel(path, label4, isLabel4Chosen);

                path = listResults.get(4);
                showResultOnLabel(path, label5, isLabel5Chosen);

                path = listResults.get(5);
                showResultOnLabel(path, label6, isLabel6Chosen);

                path = listResults.get(6);
                showResultOnLabel(path, label7, isLabel7Chosen);

                path = listResults.get(7);
                showResultOnLabel(path, label8, isLabel8Chosen);
            } catch (IndexOutOfBoundsException ignored) {
            }
        } else if (runningMode == Enums.RunningMode.COMMAND_MODE) {
            try {
                String command = listResults.get(0);
                showCommandOnLabel(command, label1, isLabel1Chosen);

                command = listResults.get(1);
                showCommandOnLabel(command, label2, isLabel2Chosen);

                command = listResults.get(2);
                showCommandOnLabel(command, label3, isLabel3Chosen);

                command = listResults.get(3);
                showCommandOnLabel(command, label4, isLabel4Chosen);

                command = listResults.get(4);
                showCommandOnLabel(command, label5, isLabel5Chosen);

                command = listResults.get(5);
                showCommandOnLabel(command, label6, isLabel6Chosen);

                command = listResults.get(6);
                showCommandOnLabel(command, label7, isLabel7Chosen);

                command = listResults.get(7);
                showCommandOnLabel(command, label8, isLabel8Chosen);
            } catch (IndexOutOfBoundsException ignored) {
            }
        } else if (runningMode == Enums.RunningMode.PLUGIN_MODE) {
            try {
                String command = listResults.get(0);
                showPluginResultOnLabel(command, label1, isLabel1Chosen);

                command = listResults.get(1);
                showPluginResultOnLabel(command, label2, isLabel2Chosen);

                command = listResults.get(2);
                showPluginResultOnLabel(command, label3, isLabel3Chosen);

                command = listResults.get(3);
                showPluginResultOnLabel(command, label4, isLabel4Chosen);

                command = listResults.get(4);
                showPluginResultOnLabel(command, label5, isLabel5Chosen);

                command = listResults.get(5);
                showPluginResultOnLabel(command, label6, isLabel6Chosen);

                command = listResults.get(6);
                showPluginResultOnLabel(command, label7, isLabel7Chosen);

                command = listResults.get(7);
                showPluginResultOnLabel(command, label8, isLabel8Chosen);
            } catch (IndexOutOfBoundsException ignored) {
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
        label.setIcon(null);
        label.setBorder(null);
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
     * @param command 要运行的cmd命令
     * @param filePath 文件位置（必须传入文件夹）
     * @param workingDir 应用打开后的工作目录
     * @return vbs的路径
     */
    private String generateBatAndVbsFile(String command, String filePath, String workingDir) {
        char disk = workingDir.charAt(0);
        String start = workingDir.substring(0,2);
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
                        String tmpDir = new File("").getAbsolutePath().contains(" ") ?
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
        SettingsFrame settingsFrame = SettingsFrame.getInstance();
        EventUtil eventUtil = EventUtil.getInstance();
        if (allConfigs.getCacheNum() < allConfigs.getCacheNumLimit()) {
            if (!settingsFrame.isCacheExist(content)) {
                eventUtil.putEvent(new AddToCacheEvent(content));
                settingsFrame.addCache(content);
                allConfigs.incrementCacheNum();
            }
        }
    }

    /**
     * 从缓存中搜索结果并将匹配的放入listResults
     */
    private void searchCache() {
        try (PreparedStatement statement = SQLiteUtil.getPreparedStatement("SELECT PATH FROM cache;");
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
    private boolean isMatched(String path, boolean isIgnoreCase) {
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
            if (!matcherStrFromFilePath.contains(eachKeyword)) {
                if (isPath) {
                    return false;
                } else {
                    if (PinyinUtil.isContainChinese(matcherStrFromFilePath)) {
                        if (!PinyinUtil.toPinyin(matcherStrFromFilePath, "").contains(eachKeyword)) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }
        }
        return true;
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
            for (File each : files) {
                matchOnCacheAndPriorityFolder(each.getAbsolutePath(), false);
                if (each.isDirectory()) {
                    listRemain.add(each.getAbsolutePath());
                }
            }

            int cpuCores = Runtime.getRuntime().availableProcessors();
            final int threadCount = Math.min(cpuCores, 8);
            for (int i = 0; i < threadCount; ++i) {
                CachedThreadPoolUtil.getInstance().executeTask(() -> {
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
                        for (File each : allFiles) {
                            matchOnCacheAndPriorityFolder(each.getAbsolutePath(), false);
                            if (startTime > startSearchTime) {
                                listRemain.clear();
                                break;
                            }
                            if (each.isDirectory()) {
                                listRemain.add(each.getAbsolutePath());
                            }
                        }
                    }
                    taskNum.incrementAndGet();
                });
            }
            //等待所有线程完成
            try {
                int count = 0;
                EventUtil eventUtil = EventUtil.getInstance();
                while (taskNum.get() != threadCount) {
                    TimeUnit.MILLISECONDS.sleep(1);
                    count++;
                    if (count >= 2000 || (!eventUtil.isNotMainExit())) {
                        break;
                    }
                }
            } catch (InterruptedException ignored) {
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
        searchBar.setVisible(b);
        GetHandle.INSTANCE.setSearchBarUsingStatus(b);
    }

    /**
     * 重置所有状态并关闭窗口
     */
    private void closeSearchBar() {
        SwingUtilities.invokeLater(() -> {
            if (isVisible()) {
                clearAllLabels();
                setVisible(false);
            }
            clearTextFieldText();
            resetAllStatus();
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
        });
    }

    private void resetAllStatus() {
        startTime = System.currentTimeMillis();//结束搜索
        currentResultCount.set(0);
        currentLabelSelectedPosition.set(0);
        clearListAndTempAndReset();
        commandQueue.clear();
        isUserPressed.set(false);
        isLockMouseMotion.set(false);
        isOpenLastFolderPressed.set(false);
        isRunAsAdminPressed.set(false);
        isCopyPathPressed.set(false);
        startSignal.set(false);
        isCacheAndPrioritySearched.set(false);
        isWaiting.set(false);
        isMouseDraggedInWindow.set(false);
        EventUtil.getInstance().putEvent(new SetPreviewOrNormalMode(false));
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

    private boolean isDirectory(String text) {
        File file = new File(text);
        return file.isDirectory();
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

    public Enums.ShowingSearchBarMode getShowingMode() {
        return showingMode;
    }

    private void setSearchBarFontColor(int colorNum) {
        textField.setForeground(new Color(colorNum));
    }

    private void setBorderColor(int colorNum) {
        border = BorderFactory.createLineBorder(new Color(colorNum));
        textField.setBorder(border);
    }
}

