package FileEngine.frames;


import FileEngine.SQLiteConfig.SQLiteUtil;
import FileEngine.configs.AllConfigs;
import FileEngine.dllInterface.FileMonitor;
import FileEngine.dllInterface.GetAscII;
import FileEngine.dllInterface.GetHandle;
import FileEngine.dllInterface.IsLocalDisk;
import FileEngine.getIcon.GetIconUtil;
import FileEngine.pluginSystem.Plugin;
import FileEngine.pluginSystem.PluginUtil;
import FileEngine.robotUtil.RobotUtil;
import FileEngine.search.SearchUtil;
import FileEngine.threadPool.CachedThreadPool;
import FileEngine.translate.TranslateUtil;

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
import java.nio.charset.StandardCharsets;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;


public class SearchBar {
    private static volatile boolean isCacheAndPrioritySearched = false;
    private static volatile boolean isStartSearchLocal = false;
    private static volatile boolean isLockMouseMotion = false;
    private static volatile boolean isOpenLastFolderPressed = false;
    private static volatile boolean isUsing = false;
    private static volatile boolean isRunAsAdminPressed = false;
    private static volatile boolean isCopyPathPressed = false;
    private static volatile boolean startSignal = false;
    private static volatile boolean isUserPressed = false;
    private final Border border;
    private final JFrame searchBar;
    private final JLabel label1;
    private final JLabel label2;
    private final JLabel label3;
    private final JLabel label4;
    private final JLabel label5;
    private final JLabel label6;
    private final JLabel label7;
    private final JLabel label8;
    private final AtomicInteger labelCount;
    private final JTextField textField;
    private Color labelColor;
    private Color backgroundColor;
    private Color fontColorWithCoverage;
    private Color fontColor;
    private volatile long startTime = 0;
    private volatile boolean isWaiting = false;
    private final Pattern semicolon;
    private final Pattern colon;
    private final Pattern blank;
    private final AtomicInteger runningMode;
    private final AtomicInteger showingMode;
    private final AtomicInteger cacheNum;
    private long mouseWheelTime = 0;
    private final int iconSideLength;
    private long visibleStartTime = 0;  //记录窗口开始可见的事件，窗口默认最短可见时间0.5秒，防止窗口快速闪烁
    private final ConcurrentLinkedQueue<String> tempResults;  //在优先文件夹和数据库cache未搜索完时暂时保存结果，搜索完后会立即被转移到listResults
    private final ConcurrentLinkedQueue<String> commandQueue;  //保存需要被执行的sql语句
    private final CopyOnWriteArrayList<String> listResults;  //保存从数据库中找出符合条件的记录（文件路径）
    private final Set<String> listResultsCopy;
    private volatile String[] searchCase;
    private volatile String searchText;
    private volatile String[] keywords;
    private final SearchUtil search;
    private final TaskBar taskBar;
    private final AtomicInteger resultCount;  //保存当前选中的结果是在listResults中的第几个 范围 0 - listResults.size()
    private final AtomicInteger currentLabelSelectedPosition;   //保存当前是哪个label被选中 范围 0 - 7
    private volatile Plugin currentUsingPlugin;

    private static class SearchBarBuilder {
        private static final SearchBar instance = new SearchBar();
    }

    private SearchBar() {
        listResults = new CopyOnWriteArrayList<>();
        tempResults = new ConcurrentLinkedQueue<>();
        commandQueue = new ConcurrentLinkedQueue<>();
        listResultsCopy = ConcurrentHashMap.newKeySet();
        border = BorderFactory.createLineBorder(new Color(73, 162, 255, 255));
        searchBar = new JFrame();
        labelCount = new AtomicInteger(0);
        resultCount = new AtomicInteger(0);
        runningMode = new AtomicInteger(AllConfigs.RunningMode.NORMAL_MODE);
        cacheNum = new AtomicInteger(0);
        showingMode = new AtomicInteger(AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING);
        currentLabelSelectedPosition = new AtomicInteger(0);
        semicolon = Pattern.compile(";");
        colon = Pattern.compile(":");
        blank = Pattern.compile(" ");
        JPanel panel = new JPanel();

        search = SearchUtil.getInstance();
        taskBar = TaskBar.getInstance();

        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
        int width = screenSize.width;
        int height = screenSize.height;
        int searchBarWidth = (int) (width * 0.4);
        int searchBarHeight = (int) (height * 0.5);
        int positionX = width / 2 - searchBarWidth / 2;
        int positionY = height / 2 - searchBarHeight / 2;


        labelColor = new Color(AllConfigs.getLabelColor());
        fontColorWithCoverage = new Color(AllConfigs.getFontColorWithCoverage());
        backgroundColor = new Color(AllConfigs.getDefaultBackgroundColor());
        fontColor = new Color(AllConfigs.getFontColor());

        //frame
        searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
        searchBar.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        searchBar.setUndecorated(true);
        searchBar.getRootPane().setWindowDecorationStyle(JRootPane.NONE);
        searchBar.setBackground(null);
        searchBar.setOpacity(AllConfigs.getTransparency());
        searchBar.setContentPane(panel);
        searchBar.setType(JFrame.Type.UTILITY);
        searchBar.setAlwaysOnTop(true);
        //用于C++判断是否点击了当前窗口
        searchBar.setTitle("File-Engine-SearchBar");

        //labels
        Font labelFont = new Font(null, Font.BOLD, (int) ((searchBarHeight * 0.2) / 96 * 72) / 4);
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
        Color transparentColor = new Color(0, 0, 0, 0);
        searchBar.setBackground(transparentColor);


        //TextField
        textField = new JTextField(1000);
        textField.setSize(searchBarWidth - 6, labelHeight - 5);
        Font textFieldFont = new Font(null, Font.PLAIN, (int) ((searchBarHeight * 0.4) / 96 * 72) / 4);
        textField.setFont(textFieldFont);
        textField.setBorder(border);
        textField.setForeground(Color.BLACK);
        textField.setHorizontalAlignment(JTextField.LEFT);
        textField.setBackground(Color.WHITE);
        textField.setLocation(3, 0);
        textField.setOpaque(true);
        textField.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent e) {
            }

            @Override
            public void focusLost(FocusEvent e) {
                if (System.currentTimeMillis() - visibleStartTime > 500) {
                    if (showingMode.get() != AllConfigs.ShowingSearchBarMode.EXPLORER_ATTACH && AllConfigs.isLoseFocusClose()) {
                        closeSearchBar();
                    } else if (showingMode.get() == AllConfigs.ShowingSearchBarMode.EXPLORER_ATTACH) {
                        closeWithoutHideSearchBar();
                    }
                }
            }
        });

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

        initCacheNum();

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

        //添加textfield对键盘上下键的响应
        addTextFieldKeyListener();
    }

    public static SearchBar getInstance() {
        return SearchBarBuilder.instance;
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
        label.setForeground(fontColor);
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
     * 让搜索窗口响应鼠标双击事件以打开文件
     */
    private void addSearchBarMouseListener() {
        searchBar.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                int count = e.getClickCount();
                if (count == 2) {
                    if (!(resultCount.get() == 0)) {
                        if (runningMode.get() != AllConfigs.RunningMode.PLUGIN_MODE) {
                            if (showingMode.get() != AllConfigs.ShowingSearchBarMode.EXPLORER_ATTACH) {
                                if (isVisible()) {
                                    setVisible(false);
                                }
                            }
                            String res = listResults.get(labelCount.get());
                            if (runningMode.get() == AllConfigs.RunningMode.NORMAL_MODE) {
                                if (showingMode.get() == AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING) {
                                    if (isOpenLastFolderPressed) {
                                        //打开上级文件夹
                                        File open = new File(res);
                                        try {
                                            Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                        } catch (IOException e1) {
                                            e1.printStackTrace();
                                        }
                                    } else if (AllConfigs.isDefaultAdmin() || isRunAsAdminPressed) {
                                        openWithAdmin(res);
                                    } else if (isCopyPathPressed) {
                                        copyToClipBoard(res);
                                    } else {
                                        if (res.endsWith(".bat") || res.endsWith(".cmd")) {
                                            openWithAdmin(res);
                                        } else {
                                            openWithoutAdmin(res);
                                        }
                                    }
                                    saveCache(res);
                                }
                            } else if (runningMode.get() == AllConfigs.RunningMode.COMMAND_MODE) {
                                String[] commandInfo = semicolon.split(res);
                                boolean isExecuted = false;
                                try {
                                    isExecuted = runInternalCommand(colon.split(commandInfo[0])[1]);
                                } catch (IOException | URISyntaxException ignored) {
                                }
                                if (isExecuted) {
                                    return;
                                }
                                File open = new File(commandInfo[1]);
                                if (isOpenLastFolderPressed) {
                                    //打开上级文件夹
                                    try {
                                        Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                    } catch (IOException e1) {
                                        JOptionPane.showMessageDialog(null, TranslateUtil.getInstance().getTranslation("Execute failed"));
                                    }
                                } else if (AllConfigs.isDefaultAdmin() || isRunAsAdminPressed) {
                                    openWithAdmin(open.getAbsolutePath());
                                } else if (isCopyPathPressed) {
                                    copyToClipBoard(res);
                                } else {
                                    if (res.endsWith(".bat") || res.endsWith(".cmd")) {
                                        openWithAdmin(open.getAbsolutePath());
                                    } else {
                                        openWithoutAdmin(open.getAbsolutePath());
                                    }
                                }
                            }
                        } else if (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
                            if (showingMode.get() == AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING) {
                                if (currentUsingPlugin != null) {
                                    if (resultCount.get() != 0) {
                                        currentUsingPlugin.mousePressed(e, listResults.get(labelCount.get()));
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
                if (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
                    if (currentUsingPlugin != null) {
                        if (resultCount.get() != 0) {
                            currentUsingPlugin.mouseReleased(e, listResults.get(labelCount.get()));
                        }
                    }
                }
            }
        });
    }


    //在explorer attach模式时操作鼠标和键盘以快速跳转到文件位置
    private void quickJump(String result) {
        int x, y;
        Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
        Transferable originalData = clipboard.getContents(null);
        if (isFile(result)) {
            result = getParentPath(result);
        }
        copyToClipBoard(result);
        x = GetHandle.INSTANCE.get_toolbar_click_x();
        y = GetHandle.INSTANCE.get_toolbar_click_y();
        RobotUtil.getInstance().mouseClicked(x, y, 1, InputEvent.BUTTON1_MASK);
        try {
            TimeUnit.MILLISECONDS.sleep(50);
        } catch (InterruptedException ignored) {
        }
        RobotUtil.getInstance().keyTyped(KeyEvent.VK_CONTROL, KeyEvent.VK_V);
        RobotUtil.getInstance().keyTyped(KeyEvent.VK_ENTER);
        clipboard.setContents(originalData, null);
    }

    /**
     * 复制信息到系统剪贴板
     *
     * @param res 需要复制的信息
     */
    private void copyToClipBoard(String res) {
        Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
        Transferable trans = new StringSelection(res);
        clipboard.setContents(trans, null);
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

            @Override
            public void keyPressed(KeyEvent arg0) {
                int key = arg0.getKeyCode();
                if (key == 8 && getTextFieldText().isEmpty()) {
                    arg0.consume();
                }
                if (!(resultCount.get() == 0)) {
                    if (38 == key) {
                        //上键被点击
                        if (isFirstPress || System.currentTimeMillis() - pressTime > timeLimit) {
                            pressTime = System.currentTimeMillis();
                            isFirstPress = false;
                            if (isLabelNotEmpty(label1) && isLabelNotEmpty(label2) && isLabelNotEmpty(label3) && isLabelNotEmpty(label4)
                                    && isLabelNotEmpty(label5) && isLabelNotEmpty(label6) && isLabelNotEmpty(label7) && isLabelNotEmpty(label8)) {
                                isUserPressed = true;
                            }

                            if (!getTextFieldText().isEmpty()) {
                                labelCount.decrementAndGet();

                                if (labelCount.get() >= resultCount.get()) {
                                    labelCount.set(resultCount.get() - 1);
                                }
                                if (labelCount.get() <= 0) {
                                    labelCount.set(0);
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
                                isUserPressed = true;
                            }
                            boolean isNextLabelValid = isNextLabelValid();
                            if (isNextLabelValid) {
                                if (!getTextFieldText().isEmpty()) {
                                    labelCount.incrementAndGet();

                                    if (labelCount.get() >= resultCount.get()) {
                                        labelCount.set(resultCount.get() - 1);
                                    }
                                    if (labelCount.get() <= 0) {
                                        labelCount.set(0);
                                    }
                                    moveDownward(getCurrentLabelPos());
                                }
                            }
                        }
                    } else if (10 == key) {
                        if (runningMode.get() != AllConfigs.RunningMode.PLUGIN_MODE) {
                            //enter被点击
                            if (showingMode.get() != AllConfigs.ShowingSearchBarMode.EXPLORER_ATTACH) {
                                if (isVisible()) {
                                    setVisible(false);
                                }
                            }
                            if (!(resultCount.get() == 0)) {
                                String res = listResults.get(labelCount.get());
                                if (runningMode.get() == AllConfigs.RunningMode.NORMAL_MODE) {
                                    if (showingMode.get() == AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING) {
                                        if (isOpenLastFolderPressed) {
                                            //打开上级文件夹
                                            File open = new File(res);
                                            try {
                                                Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                            } catch (IOException e) {
                                                e.printStackTrace();
                                            }
                                        } else if (AllConfigs.isDefaultAdmin() || isRunAsAdminPressed) {
                                            openWithAdmin(res);
                                        } else if (isCopyPathPressed) {
                                            copyToClipBoard(res);
                                        } else {
                                            if (res.endsWith(".bat") || res.endsWith(".cmd")) {
                                                openWithAdmin(res);
                                            } else {
                                                openWithoutAdmin(res);
                                            }
                                        }
                                        saveCache(res);
                                    } else if (showingMode.get() == AllConfigs.ShowingSearchBarMode.EXPLORER_ATTACH) {
                                        if (isCopyPathPressed) {
                                            copyToClipBoard(res);
                                        } else {
                                            quickJump(res);
                                        }
                                    }
                                } else if (runningMode.get() == AllConfigs.RunningMode.COMMAND_MODE) {
                                    String[] commandInfo = semicolon.split(res);
                                    boolean isExecuted = false;
                                    try {
                                        isExecuted = runInternalCommand(colon.split(commandInfo[0])[1]);
                                    } catch (IOException | URISyntaxException ignored) {
                                    }
                                    if (isExecuted) {
                                        return;
                                    }
                                    File open = new File(commandInfo[1]);
                                    if (isOpenLastFolderPressed) {
                                        //打开上级文件夹
                                        try {
                                            Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                        } catch (IOException e) {
                                            JOptionPane.showMessageDialog(null, TranslateUtil.getInstance().getTranslation("Execute failed"));
                                        }
                                    } else if (AllConfigs.isDefaultAdmin() || isRunAsAdminPressed) {
                                        openWithAdmin(open.getAbsolutePath());
                                    } else if (isCopyPathPressed) {
                                        copyToClipBoard(res);
                                    } else {
                                        if (res.endsWith(".bat") || res.endsWith(".cmd")) {
                                            openWithAdmin(open.getAbsolutePath());
                                        } else {
                                            openWithoutAdmin(open.getAbsolutePath());
                                        }
                                    }
                                }
                            }
                            detectShowingModeAndClose();
                        }
                    } else if (AllConfigs.getOpenLastFolderKeyCode() == key) {
                        //打开上级文件夹热键被点击
                        isOpenLastFolderPressed = true;
                    } else if (AllConfigs.getRunAsAdminKeyCode() == key) {
                        //以管理员方式运行热键被点击
                        isRunAsAdminPressed = true;
                    } else if (AllConfigs.getCopyPathKeyCode() == key) {
                        isCopyPathPressed = true;
                    }
                }
                if (showingMode.get() == AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING) {
                    if (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
                        if (key != 38 && key != 40) {
                            if (currentUsingPlugin != null) {
                                if (!(resultCount.get() == 0)) {
                                    currentUsingPlugin.keyPressed(arg0, listResults.get(labelCount.get()));
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
                if (AllConfigs.getOpenLastFolderKeyCode() == key) {
                    //复位按键状态
                    isOpenLastFolderPressed = false;
                } else if (AllConfigs.getRunAsAdminKeyCode() == key) {
                    isRunAsAdminPressed = false;
                } else if (AllConfigs.getCopyPathKeyCode() == key) {
                    isCopyPathPressed = false;
                }

                if (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
                    if (key != 38 && key != 40) {
                        if (currentUsingPlugin != null) {
                            if (!(resultCount.get() == 0)) {
                                currentUsingPlugin.keyReleased(arg0, listResults.get(labelCount.get()));
                            }
                        }
                    }
                }
            }

            @Override
            public void keyTyped(KeyEvent arg0) {
                if (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
                    int key = arg0.getKeyCode();
                    if (key != 38 && key != 40) {
                        if (currentUsingPlugin != null) {
                            if (!(resultCount.get() == 0)) {
                                currentUsingPlugin.keyTyped(arg0, listResults.get(labelCount.get()));
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * 在command模式下，检测当前输入信息是否是软件已经定义的内部命令
     * clearbin update help version
     * return true only the internal command was executed. Otherwise false
     */
    private boolean runInternalCommand(String commandName) throws URISyntaxException, IOException {
        switch (commandName) {
            case "clearbin":
                detectShowingModeAndClose();
                int r = JOptionPane.showConfirmDialog(null, TranslateUtil.getInstance().getTranslation(
                        "Are you sure you want to empty the recycle bin"));
                if (r == 0) {
                    try {
                        File[] roots = File.listRoots();
                        for (File root : roots) {
                            Runtime.getRuntime().exec("cmd.exe /c rd /s /q " + root.getAbsolutePath() + "$Recycle.Bin");
                        }
                        JOptionPane.showMessageDialog(null, TranslateUtil.getInstance().getTranslation(
                                "Successfully empty the recycle bin"));
                    } catch (IOException e) {
                        JOptionPane.showMessageDialog(null, TranslateUtil.getInstance().getTranslation(
                                "Failed to empty the recycle bin"));
                    }
                }
                return true;
            case "update":
                detectShowingModeAndClose();
                search.setStatus(SearchUtil.MANUAL_UPDATE);
                startSignal = false;
                return true;
            case "help":
                detectShowingModeAndClose();
                Desktop desktop;
                if (Desktop.isDesktopSupported()) {
                    desktop = Desktop.getDesktop();
                    desktop.browse(new URI("https://github.com/XUANXUQAQ/File-Engine/wiki/Usage"));
                }
                return true;
            case "version":
                detectShowingModeAndClose();
                JOptionPane.showMessageDialog(null, TranslateUtil.getInstance().getTranslation(
                        "Current Version:") + AllConfigs.version);
                return true;
            default:
                break;
        }
        return false;
    }

    /**
     * 设置当前label为选中状态
     *
     * @param label 需要设置的label
     */
    private void setLabelChosen(JLabel label) {
        label.setBackground(labelColor);
    }

    /**
     * 设置当前label为未选中
     *
     * @param label 需要设置的label
     */
    private void setLabelNotChosen(JLabel label) {
        label.setBackground(backgroundColor);
    }

    /**
     * 检测鼠标在窗口的位置，并设置鼠标所在位置的label为选中
     */
    private void addSearchBarMouseMotionListener() {
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

        searchBar.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                //判定当前位置
                if (!isLockMouseMotion) {
                    int position = getCurrentLabelPos();
                    int mousePosition = 0;
                    if (labelPosition2 <= e.getY() && e.getY() < labelPosition3) {
                        mousePosition = 1;
                    } else if (labelPosition3 <= e.getY() && e.getY() < labelPosition4) {
                        mousePosition = 2;
                    } else if (labelPosition4 <= e.getY() && e.getY() < labelPosition5) {
                        mousePosition = 3;
                    } else if (labelPosition5 <= e.getY() && e.getY() < labelPosition6) {
                        mousePosition = 4;
                    } else if (labelPosition6 <= e.getY() && e.getY() < labelPosition7) {
                        mousePosition = 5;
                    } else if (labelPosition7 <= e.getY() && e.getY() < labelPosition8) {
                        mousePosition = 6;
                    } else if (labelPosition8 <= e.getY() && e.getY() < labelPosition9) {
                        mousePosition = 7;
                    }
                    if (mousePosition < resultCount.get()) {
                        int ret;
                        if (position < mousePosition) {
                            ret = mousePosition - position;
                        } else {
                            ret = -(position - mousePosition);
                        }
                        labelCount.getAndAdd(ret);
                        currentLabelSelectedPosition.getAndAdd(ret);
                        switch (mousePosition) {
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
        });
    }

    /**
     * 检测当前选中的label的下一个label是否可用（是否有结果在显示）
     *
     * @return true false
     */
    private boolean isNextLabelValid() {
        boolean isNextLabelValid = false;
        switch (labelCount.get()) {
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
                if (resultCount.get() > 8) {
                    return true;
                }
        }
        return isNextLabelValid;
    }

    /**
     * 检测当前窗口是否未显示任何结果
     *
     * @param label 判断的label
     * @return true如果label上有显示 否则false
     */
    private boolean isLabelNotEmpty(JLabel label) {
        boolean isEmpty = true;
        try {
            String text;
            if (label != null) {
                text = label.getText();
                if (text != null) {
                    isEmpty = text.isEmpty();
                }
            }
        } catch (NullPointerException ignored) {
        }
        return !isEmpty;
    }

    //获取当前选中label的编号 从0-7
    private int getCurrentLabelPos() {
        return currentLabelSelectedPosition.get();
    }

    private void addSearchBarMouseWheelListener() {
        searchBar.addMouseWheelListener(e -> {
            mouseWheelTime = System.currentTimeMillis();
            isLockMouseMotion = true;
            if (e.getPreciseWheelRotation() > 0) {
                //向下滚动
                if (isLabelNotEmpty(label1) && isLabelNotEmpty(label2) && isLabelNotEmpty(label3) && isLabelNotEmpty(label4)
                        && isLabelNotEmpty(label5) && isLabelNotEmpty(label6) && isLabelNotEmpty(label7) && isLabelNotEmpty(label8)) {
                    isUserPressed = true;
                }
                boolean isNextLabelValid = isNextLabelValid();

                if (isNextLabelValid) {
                    if (!getTextFieldText().isEmpty()) {
                        labelCount.incrementAndGet();

                        //System.out.println(labelCount);
                        if (labelCount.get() >= resultCount.get()) {
                            labelCount.set(resultCount.get() - 1);
                        }
                        if (labelCount.get() <= 0) {
                            labelCount.set(0);
                        }
                        moveDownward(getCurrentLabelPos());
                    }
                }
            } else if (e.getPreciseWheelRotation() < 0) {
                //向上滚动
                if (isLabelNotEmpty(label1) && isLabelNotEmpty(label2) && isLabelNotEmpty(label3) && isLabelNotEmpty(label4)
                        && isLabelNotEmpty(label5) && isLabelNotEmpty(label6) && isLabelNotEmpty(label7) && isLabelNotEmpty(label8)) {
                    isUserPressed = true;
                }
                if (!getTextFieldText().isEmpty()) {
                    labelCount.getAndDecrement();

                    if (labelCount.get() >= resultCount.get()) {
                        labelCount.set(resultCount.get() - 1);
                    }
                    if (labelCount.get() <= 0) {
                        labelCount.set(0);
                    }
                    moveUpward(getCurrentLabelPos());
                }

            }
        });
    }

    private void moveDownward(int position) {
        try {
            currentLabelSelectedPosition.incrementAndGet();
            if (currentLabelSelectedPosition.get() > 7) {
                currentLabelSelectedPosition.set(7);
            }
            switch (position) {
                case 0:
                    int size = resultCount.get();
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
                    size = resultCount.get();
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
                    size = resultCount.get();
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
                    size = resultCount.get();
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
                    size = resultCount.get();
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
                    size = resultCount.get();
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
                    if (runningMode.get() == AllConfigs.RunningMode.NORMAL_MODE) {
                        //到达最下端，刷新显示
                        try {
                            String path = listResults.get(labelCount.get() - 7);
                            showResultOnLabel(path, label1, false);

                            path = listResults.get(labelCount.get() - 6);
                            showResultOnLabel(path, label2, false);

                            path = listResults.get(labelCount.get() - 5);
                            showResultOnLabel(path, label3, false);

                            path = listResults.get(labelCount.get() - 4);
                            showResultOnLabel(path, label4, false);

                            path = listResults.get(labelCount.get() - 3);
                            showResultOnLabel(path, label5, false);

                            path = listResults.get(labelCount.get() - 2);
                            showResultOnLabel(path, label6, false);

                            path = listResults.get(labelCount.get() - 1);
                            showResultOnLabel(path, label7, false);

                            path = listResults.get(labelCount.get());
                            showResultOnLabel(path, label8, true);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            if (AllConfigs.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    } else if (runningMode.get() == AllConfigs.RunningMode.COMMAND_MODE) {
                        //到达了最下端，刷新显示
                        try {
                            String command = listResults.get(labelCount.get() - 7);
                            showCommandOnLabel(command, label1, false);

                            command = listResults.get(labelCount.get() - 6);
                            showCommandOnLabel(command, label2, false);

                            command = listResults.get(labelCount.get() - 5);
                            showCommandOnLabel(command, label3, false);

                            command = listResults.get(labelCount.get() - 4);
                            showCommandOnLabel(command, label4, false);

                            command = listResults.get(labelCount.get() - 3);
                            showCommandOnLabel(command, label5, false);

                            command = listResults.get(labelCount.get() - 2);
                            showCommandOnLabel(command, label6, false);

                            command = listResults.get(labelCount.get() - 1);
                            showCommandOnLabel(command, label7, false);

                            command = listResults.get(labelCount.get());
                            showCommandOnLabel(command, label8, true);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            if (AllConfigs.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    } else if (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
                        try {
                            String command = listResults.get(labelCount.get() - 7);
                            showPluginResultOnLabel(command, label1, false);

                            command = listResults.get(labelCount.get() - 6);
                            showPluginResultOnLabel(command, label2, false);

                            command = listResults.get(labelCount.get() - 5);
                            showPluginResultOnLabel(command, label3, false);

                            command = listResults.get(labelCount.get() - 4);
                            showPluginResultOnLabel(command, label4, false);

                            command = listResults.get(labelCount.get() - 3);
                            showPluginResultOnLabel(command, label5, false);

                            command = listResults.get(labelCount.get() - 2);
                            showPluginResultOnLabel(command, label6, false);

                            command = listResults.get(labelCount.get() - 1);
                            showPluginResultOnLabel(command, label7, false);

                            command = listResults.get(labelCount.get());
                            showPluginResultOnLabel(command, label8, true);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            if (AllConfigs.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    }
                    break;
            }
        } catch (NullPointerException ignored) {
        }
    }

    private void moveUpward(int position) {
        try {
            currentLabelSelectedPosition.decrementAndGet();
            if (currentLabelSelectedPosition.get() < 0) {
                currentLabelSelectedPosition.set(0);
            }
            int size;
            switch (position) {
                case 0:
                    if (runningMode.get() == AllConfigs.RunningMode.NORMAL_MODE) {
                        //到达了最上端，刷新显示
                        try {
                            String path = listResults.get(labelCount.get());
                            showResultOnLabel(path, label1, true);

                            path = listResults.get(labelCount.get() + 1);
                            showResultOnLabel(path, label2, false);

                            path = listResults.get(labelCount.get() + 2);
                            showResultOnLabel(path, label3, false);

                            path = listResults.get(labelCount.get() + 3);
                            showResultOnLabel(path, label4, false);

                            path = listResults.get(labelCount.get() + 4);
                            showResultOnLabel(path, label5, false);

                            path = listResults.get(labelCount.get() + 5);
                            showResultOnLabel(path, label6, false);

                            path = listResults.get(labelCount.get() + 6);
                            showResultOnLabel(path, label7, false);

                            path = listResults.get(labelCount.get() + 7);
                            showResultOnLabel(path, label8, false);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            if (AllConfigs.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    } else if (runningMode.get() == AllConfigs.RunningMode.COMMAND_MODE) {
                        //到达了最上端，刷新显示
                        try {
                            String command = listResults.get(labelCount.get());
                            showCommandOnLabel(command, label1, true);

                            command = listResults.get(labelCount.get() + 1);
                            showCommandOnLabel(command, label2, false);

                            command = listResults.get(labelCount.get() + 2);
                            showCommandOnLabel(command, label3, false);

                            command = listResults.get(labelCount.get() + 3);
                            showCommandOnLabel(command, label4, false);

                            command = listResults.get(labelCount.get() + 4);
                            showCommandOnLabel(command, label5, false);

                            command = listResults.get(labelCount.get() + 5);
                            showCommandOnLabel(command, label6, false);

                            command = listResults.get(labelCount.get() + 6);
                            showCommandOnLabel(command, label7, false);

                            command = listResults.get(labelCount.get() + 7);
                            showCommandOnLabel(command, label8, false);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            if (AllConfigs.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    } else if (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
                        try {
                            String command = listResults.get(labelCount.get());
                            showPluginResultOnLabel(command, label1, true);

                            command = listResults.get(labelCount.get() + 1);
                            showPluginResultOnLabel(command, label2, false);

                            command = listResults.get(labelCount.get() + 2);
                            showPluginResultOnLabel(command, label3, false);

                            command = listResults.get(labelCount.get() + 3);
                            showPluginResultOnLabel(command, label4, false);

                            command = listResults.get(labelCount.get() + 4);
                            showPluginResultOnLabel(command, label5, false);

                            command = listResults.get(labelCount.get() + 5);
                            showPluginResultOnLabel(command, label6, false);

                            command = listResults.get(labelCount.get() + 6);
                            showPluginResultOnLabel(command, label7, false);

                            command = listResults.get(labelCount.get() + 7);
                            showPluginResultOnLabel(command, label8, false);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            if (AllConfigs.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    }
                    break;
                case 1:
                    size = resultCount.get();
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
                    size = resultCount.get();
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
                    size = resultCount.get();
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
                    size = resultCount.get();
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
                    size = resultCount.get();
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
                    size = resultCount.get();
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
        } catch (NullPointerException ignored) {
        }
    }

    private void addTextFieldDocumentListener() {
        textField.getDocument().addDocumentListener(new DocumentListener() {
            final StringBuilder strb = new StringBuilder();
            @Override
            public void insertUpdate(DocumentEvent e) {
                clearLabel();
                listResults.clear();
                listResultsCopy.clear();
                tempResults.clear();
                resultCount.set(0);
                labelCount.set(0);
                currentLabelSelectedPosition.set(0);
                isCacheAndPrioritySearched = false;
                startTime = System.currentTimeMillis();
                startSignal = true;
                String text = getTextFieldText();
                char first = text.charAt(0);
                if (first == ':') {
                    runningMode.set(AllConfigs.RunningMode.COMMAND_MODE);
                } else if (first == '>') {
                    runningMode.set(AllConfigs.RunningMode.PLUGIN_MODE);
                    String subText = text.substring(1);
                    String[] s = blank.split(subText);
                    currentUsingPlugin = PluginUtil.getInstance().getPluginByIdentifier(s[0]);
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
                    runningMode.set(AllConfigs.RunningMode.NORMAL_MODE);
                }
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                clearLabel();
                listResults.clear();
                listResultsCopy.clear();
                tempResults.clear();
                resultCount.set(0);
                labelCount.set(0);
                currentLabelSelectedPosition.set(0);
                isCacheAndPrioritySearched = false;
                String text = getTextFieldText();
                try {
                    char first = text.charAt(0);
                    if (first == ':') {
                        runningMode.set(AllConfigs.RunningMode.COMMAND_MODE);
                    } else if (first == '>') {
                        runningMode.set(AllConfigs.RunningMode.PLUGIN_MODE);
                        String subText = text.substring(1);
                        String[] s = blank.split(subText);
                        currentUsingPlugin = PluginUtil.getInstance().getPluginByIdentifier(s[0]);
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
                        runningMode.set(AllConfigs.RunningMode.NORMAL_MODE);
                    }
                } catch (StringIndexOutOfBoundsException e1) {
                    runningMode.set(AllConfigs.RunningMode.NORMAL_MODE);
                }
                if (text.isEmpty()) {
                    resultCount.set(0);
                    labelCount.set(0);
                    currentLabelSelectedPosition.set(0);
                    startTime = System.currentTimeMillis();
                    startSignal = false;
                } else {
                    startTime = System.currentTimeMillis();
                    startSignal = true;
                }
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                startTime = System.currentTimeMillis();
                startSignal = false;
            }
        });
    }

    private static boolean isAdmin() {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("cmd.exe");
            Process process = processBuilder.start();
            PrintStream printStream = new PrintStream(process.getOutputStream(), true);
            Scanner scanner = new Scanner(process.getInputStream());
            printStream.println("@echo off");
            printStream.println(">nul 2>&1 \"%SYSTEMROOT%\\system32\\cacls.exe\" \"%SYSTEMROOT%\\system32\\config\\system\"");
            printStream.println("echo %errorlevel%");

            boolean printedErrorlevel = false;
            while (true) {
                String nextLine = scanner.nextLine();
                if (printedErrorlevel) {
                    int errorlevel = Integer.parseInt(nextLine);
                    scanner.close();
                    return errorlevel == 0;
                } else if ("echo %errorlevel%".equals(nextLine)) {
                    printedErrorlevel = true;
                }
            }
        } catch (IOException e) {
            return false;
        }
    }

    private void startMonitorDisk() {
        CachedThreadPool.getInstance().executeTask(() -> {
            File[] roots = File.listRoots();
            if (isAdmin()) {
                FileMonitor.INSTANCE.set_output(AllConfigs.getTmp().getAbsolutePath());
                for (File root : roots) {
                    boolean isLocal = IsLocalDisk.INSTANCE.isLocalDisk(root.getAbsolutePath());
                    if (isLocal) {
                        FileMonitor.INSTANCE.monitor(root.getAbsolutePath());
                    }
                }
            } else {
                System.out.println("Not administrator, file monitoring function is turned off");
                taskBar.showMessage(TranslateUtil.getInstance().getTranslation("Warning"), TranslateUtil.getInstance().getTranslation("Not administrator, file monitoring function is turned off"));
            }
        });
    }

    private void setLabelChosenOrNotChosenMouseMode(int labelNum, JLabel label) {
        if (!isUserPressed && isLabelNotEmpty(label)) {
            if (labelCount.get() == labelNum) {
                setLabelChosen(label);
            } else {
                setLabelNotChosen(label);
            }
            label.setBorder(border);
        }
    }

    private void addSearchWaiter() {
        if (!isWaiting) {
            isWaiting = true;
            CachedThreadPool.getInstance().executeTask(() -> {
                try {
                    while (isWaiting) {
                        if (search.getStatus() == SearchUtil.NORMAL) {
                            startTime = System.currentTimeMillis() - 500;
                            startSignal = true;
                            isWaiting = false;
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
        checkStartTimeAndOptimizeDatabase();

        startMonitorDisk();

        mergeTempQueueAndListResultsThread();

        checkPluginMessageThread();

        lockMouseMotionThread();

        setForegroundOnLabelThread();

        tryToShowRecordsThread();

        repaintFrameThread();

        addSqlCommandThread();

        pollCommandsAndSearchDatabaseThread();

        createSqlIndexThread();

        sendSignalAndShowCommandThread();

        addRecordsToDatabaseThread();

        deleteRecordsToDatabaseThread();

        checkTimeAndExecuteSqlCommandsThread();

        recreateIndexThread();

        switchSearchBarShowingMode();

        changeSearchBarSizeWhenOnExplorerMode();

        changeSearchBarSizeWhenOnNormalMode();
    }

    private void switchSearchBarShowingMode() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                GetHandle.INSTANCE.start();
                while (AllConfigs.isNotMainExit()) {
                    if (GetHandle.INSTANCE.is_explorer_at_top()) {
                        switchToExplorerAttachMode();
                        GetHandle.INSTANCE.resetMouseStatus();
                    } else {
                        if (GetHandle.INSTANCE.isDialogNotExist()) {
                            switchToNormalMode();
                            TimeUnit.MILLISECONDS.sleep(100);
                            continue;
                        }
                        if (GetHandle.INSTANCE.isExplorerAndSearchbarNotFocused()) {
                            switchToNormalMode();
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(100);
                }
            } catch (InterruptedException ignored) {
            } finally {
                GetHandle.INSTANCE.stop();
            }
        });
    }

    private void changeSearchBarSizeWhenOnNormalMode() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (AllConfigs.isNotMainExit()) {
                    if (showingMode.get() == AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING) {
                        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
                        int width = screenSize.width;
                        int height = screenSize.height;
                        int searchBarWidth = (int) (width * 0.4);
                        int searchBarHeight = (int) (height * 0.5);
                        int labelHeight = searchBarHeight / 9;
                        int positionX = width / 2 - searchBarWidth / 2;
                        int positionY = height / 2 - searchBarHeight / 2;
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
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void changeSearchBarSizeWhenOnExplorerMode() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (AllConfigs.isNotMainExit()) {
                    if (showingMode.get() == AllConfigs.ShowingSearchBarMode.EXPLORER_ATTACH) {
                        int searchBarHeight = (int) (GetHandle.INSTANCE.getHeight() * 0.75);
                        int labelHeight = searchBarHeight / 9;
                        if (labelHeight > 20) {
                            int searchBarWidth = (int) (GetHandle.INSTANCE.getWidth() / 3);
                            int positionX = (int) (GetHandle.INSTANCE.getX());
                            int positionY = (int) (GetHandle.INSTANCE.getY() - labelHeight - 5);
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
                    TimeUnit.MILLISECONDS.sleep(15);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void switchToExplorerAttachMode() {
        int searchBarHeight = (int) (GetHandle.INSTANCE.getHeight() * 0.75);
        int labelHeight = searchBarHeight / 9;
        if (labelHeight > 35) {
            if (showingMode.get() != AllConfigs.ShowingSearchBarMode.EXPLORER_ATTACH) {
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
                GetHandle.INSTANCE.resetMouseStatus();
                showingMode.set(AllConfigs.ShowingSearchBarMode.EXPLORER_ATTACH);
                //让更改窗口大小线程先运行
                try {
                    TimeUnit.MILLISECONDS.sleep(50);
                } catch (InterruptedException ignored) {
                }
                if (!isVisible()) {
                    showSearchbar(false);
                }
            }
        }
    }

    private void switchToNormalMode() {
        if (showingMode.get() != AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING) {
            closeSearchBar();
            GetHandle.INSTANCE.resetMouseStatus();
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
            showingMode.set(AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING);
        }
    }

    private void checkStartTimeAndOptimizeDatabase() {
        CachedThreadPool.getInstance().executeTask(() -> {
            int startTimes = 0;
            File startTimeCount = new File("user/startTimeCount.dat");
            boolean isFileCreated;
            try {
                if (!startTimeCount.exists()) {
                    isFileCreated = startTimeCount.createNewFile();
                } else {
                    isFileCreated = true;
                }
                if (isFileCreated) {
                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(startTimeCount), StandardCharsets.UTF_8));
                         Statement stmt = SQLiteUtil.getStatement()) {
                        //读取启动次数
                        String times = reader.readLine();
                        if (!(times == null || times.isEmpty())) {
                            startTimes = Integer.parseInt(times);
                            //使用次数大于10次，优化数据库
                            if (startTimes >= 10) {
                                startTimes = 0;
                                if (SearchUtil.getInstance().getStatus() == SearchUtil.NORMAL) {
                                    //开始优化数据库
                                    SearchUtil.getInstance().setStatus(SearchUtil.VACUUM);
                                    try {
                                        if (AllConfigs.isDebug()) {
                                            System.out.println("开启次数超过10次，优化数据库");
                                        }
                                        stmt.execute("VACUUM;");
                                    } catch (Exception ignored) {
                                    }
                                    SearchUtil.getInstance().setStatus(SearchUtil.NORMAL);
                                }
                            }
                        }
                        //自增后写入
                        startTimes++;
                        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(startTimeCount), StandardCharsets.UTF_8))) {
                            writer.write(String.valueOf(startTimes));
                        }
                    } catch (Exception throwables) {
                        throwables.printStackTrace();
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    private void mergeTempQueueAndListResultsThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            //合并搜索结果线程
            try {
                String record;
                while (AllConfigs.isNotMainExit()) {
                    if (isCacheAndPrioritySearched) {
                        while ((record = tempResults.poll()) != null) {
                            if (!listResultsCopy.contains(record)) {
                                resultCount.incrementAndGet();
                                listResults.add(record);
                                listResultsCopy.add(record);
                            }
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (Exception e) {
                if (AllConfigs.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void checkPluginMessageThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                String[] message;
                Plugin plugin;
                while (AllConfigs.isNotMainExit()) {
                    Iterator<Plugin> iter = PluginUtil.getInstance().getPluginMapIter();
                    while (iter.hasNext()) {
                        plugin = iter.next();
                        message = plugin.getMessage();
                        if (message != null) {
                            TaskBar.getInstance().showMessage(message[0], message[1]);
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    private void lockMouseMotionThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            //锁住MouseMotion检测，阻止同时发出两个动作
            try {
                while (AllConfigs.isNotMainExit()) {
                    if (System.currentTimeMillis() - mouseWheelTime > 500) {
                        isLockMouseMotion = false;
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (Exception e) {
                if (AllConfigs.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void setForegroundOnLabelThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (AllConfigs.isNotMainExit()) {
                    //字体染色线程
                    //判定当前选定位置
                    int position = getCurrentLabelPos();
                    if (position == 0) {
                        label1.setForeground(fontColorWithCoverage);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColor);
                        label5.setForeground(fontColor);
                        label6.setForeground(fontColor);
                        label7.setForeground(fontColor);
                        label8.setForeground(fontColor);
                    } else if (position == 1) {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColorWithCoverage);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColor);
                        label5.setForeground(fontColor);
                        label6.setForeground(fontColor);
                        label7.setForeground(fontColor);
                        label8.setForeground(fontColor);
                    } else if (position == 2) {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColorWithCoverage);
                        label4.setForeground(fontColor);
                        label5.setForeground(fontColor);
                        label6.setForeground(fontColor);
                        label7.setForeground(fontColor);
                        label8.setForeground(fontColor);
                    } else if (position == 3) {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColorWithCoverage);
                        label5.setForeground(fontColor);
                        label6.setForeground(fontColor);
                        label7.setForeground(fontColor);
                        label8.setForeground(fontColor);
                    } else if (position == 4) {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColor);
                        label5.setForeground(fontColorWithCoverage);
                        label6.setForeground(fontColor);
                        label7.setForeground(fontColor);
                        label8.setForeground(fontColor);
                    } else if (position == 5) {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColor);
                        label5.setForeground(fontColor);
                        label6.setForeground(fontColorWithCoverage);
                        label7.setForeground(fontColor);
                        label8.setForeground(fontColor);
                    } else if (position == 6) {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColor);
                        label5.setForeground(fontColor);
                        label6.setForeground(fontColor);
                        label7.setForeground(fontColorWithCoverage);
                        label8.setForeground(fontColor);
                    } else if (position >= 7) {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColor);
                        label5.setForeground(fontColor);
                        label6.setForeground(fontColor);
                        label7.setForeground(fontColor);
                        label8.setForeground(fontColorWithCoverage);
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (Exception e) {
                if (AllConfigs.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void tryToShowRecordsThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            //显示结果线程
            try {
                boolean isLabel1Chosen, isLabel2Chosen, isLabel3Chosen, isLabel4Chosen,
                        isLabel5Chosen, isLabel6Chosen, isLabel7Chosen, isLabel8Chosen;
                while (AllConfigs.isNotMainExit()) {
                    isLabel1Chosen = false;
                    isLabel2Chosen = false;
                    isLabel3Chosen = false;
                    isLabel4Chosen = false;
                    isLabel5Chosen = false;
                    isLabel6Chosen = false;
                    isLabel7Chosen = false;
                    isLabel8Chosen = false;
                    if (labelCount.get() < resultCount.get()) {
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
                        if (!isLabelNotEmpty(label2) || !isLabelNotEmpty(label3) || !isLabelNotEmpty(label4) || !isLabelNotEmpty(label5)
                                || !isLabelNotEmpty(label6) || !isLabelNotEmpty(label7) || !isLabelNotEmpty(label8)) {
                            showResults(isLabel1Chosen, isLabel2Chosen, isLabel3Chosen, isLabel4Chosen,
                                    isLabel5Chosen, isLabel6Chosen, isLabel7Chosen, isLabel8Chosen);
                        }
                    }
                    String text = getTextFieldText();
                    if (text.isEmpty()) {
                        clearLabel();
                        listResults.clear();
                        listResultsCopy.clear();
                        resultCount.set(0);
                    }

                    setLabelChosenOrNotChosenMouseMode(0, label1);
                    setLabelChosenOrNotChosenMouseMode(1, label2);
                    setLabelChosenOrNotChosenMouseMode(2, label3);
                    setLabelChosenOrNotChosenMouseMode(3, label4);
                    setLabelChosenOrNotChosenMouseMode(4, label5);
                    setLabelChosenOrNotChosenMouseMode(5, label6);
                    setLabelChosenOrNotChosenMouseMode(6, label7);
                    setLabelChosenOrNotChosenMouseMode(7, label8);
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void repaintFrameThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (AllConfigs.isNotMainExit()) {
                    if (isUsing) {
                        searchBar.repaint();
                    }
                    TimeUnit.MILLISECONDS.sleep(250);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void addCommandsToCommandQueue(int startIndex) {
        String command;
        for (int i = startIndex; i <= 40; ++i) {
            command = "list" + i;
            commandQueue.add(command);
        }
    }

    private void addSqlCommandThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            //添加搜索路径线程
            int ascII;
            try {
                while (AllConfigs.isNotMainExit()) {
                    if (isStartSearchLocal) {
                        isStartSearchLocal = false;
                        ascII = getAscIISum(searchText);
                        int asciiGroup = ascII / 100;

                        switch (asciiGroup) {
                            case 0:
                                addCommandsToCommandQueue(0);
                                break;

                            case 1:
                                addCommandsToCommandQueue(1);
                                break;

                            case 2:
                                addCommandsToCommandQueue(2);
                                break;

                            case 3:
                                addCommandsToCommandQueue(3);
                                break;

                            case 4:
                                addCommandsToCommandQueue(4);
                                break;

                            case 5:
                                addCommandsToCommandQueue(5);
                                break;
                            case 6:
                                addCommandsToCommandQueue(6);
                                break;

                            case 7:
                                addCommandsToCommandQueue(7);
                                break;

                            case 8:
                                addCommandsToCommandQueue(8);
                                break;

                            case 9:
                                addCommandsToCommandQueue(9);
                                break;

                            case 10:
                                addCommandsToCommandQueue(10);
                                break;

                            case 11:
                                addCommandsToCommandQueue(11);
                                break;

                            case 12:
                                addCommandsToCommandQueue(12);
                                break;

                            case 13:
                                addCommandsToCommandQueue(13);
                                break;

                            case 14:
                                addCommandsToCommandQueue(14);
                                break;

                            case 15:
                                addCommandsToCommandQueue(15);
                                break;

                            case 16:
                                addCommandsToCommandQueue(16);
                                break;

                            case 17:
                                addCommandsToCommandQueue(17);
                                break;

                            case 18:
                                addCommandsToCommandQueue(18);
                                break;

                            case 19:
                                addCommandsToCommandQueue(19);
                                break;

                            case 20:
                                addCommandsToCommandQueue(20);
                                break;

                            case 21:
                                addCommandsToCommandQueue(21);
                                break;

                            case 22:
                                addCommandsToCommandQueue(22);
                                break;

                            case 23:
                                addCommandsToCommandQueue(23);
                                break;

                            case 24:
                                addCommandsToCommandQueue(24);
                                break;

                            case 25:
                                addCommandsToCommandQueue(25);
                                break;
                            case 26:
                                addCommandsToCommandQueue(26);
                                break;
                            case 27:
                                addCommandsToCommandQueue(27);
                                break;
                            case 28:
                                addCommandsToCommandQueue(28);
                                break;
                            case 29:
                                addCommandsToCommandQueue(29);
                                break;
                            case 30:
                                addCommandsToCommandQueue(30);
                                break;
                            case 31:
                                addCommandsToCommandQueue(31);
                                break;
                            case 32:
                                addCommandsToCommandQueue(32);
                                break;
                            case 33:
                                addCommandsToCommandQueue(33);
                                break;
                            case 34:
                                addCommandsToCommandQueue(34);
                                break;
                            case 35:
                                addCommandsToCommandQueue(35);
                                break;
                            case 36:
                                addCommandsToCommandQueue(36);
                                break;
                            case 37:
                                addCommandsToCommandQueue(37);
                                break;
                            case 38:
                                addCommandsToCommandQueue(38);
                                break;
                            case 39:
                                addCommandsToCommandQueue(39);
                                break;
                            case 40:
                                addCommandsToCommandQueue(40);
                                break;
                            default:
                                break;
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void pollCommandsAndSearchDatabaseThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                String column;
                while (AllConfigs.isNotMainExit()) {
                    if (runningMode.get() == AllConfigs.RunningMode.NORMAL_MODE) {
                        try {
                            while ((column = commandQueue.poll()) != null) {
                                searchAndAddToTempResults(System.currentTimeMillis(), column);
                                if (!isUsing) {
                                    detectShowingModeAndClose();
                                }
                            }
                        } catch (SQLException e) {
                            if (AllConfigs.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void createSqlIndexThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            //后台自动创建数据库索引
            try (Statement stmt = SQLiteUtil.getStatement()) {
                OutLoop:
                while (AllConfigs.isNotMainExit()) {
                    for (int i = 0; i <= 40; ++i) {
                        String createIndex = "CREATE INDEX IF NOT EXISTS list" + i + "_index on list" + i + "(PATH);";
                        if (!AllConfigs.isNotMainExit()) {
                            break OutLoop;
                        }
                        try {
                            if (!isUsing && SearchUtil.getInstance().getStatus() == SearchUtil.NORMAL) {
                                stmt.execute(createIndex);
                            }
                        } catch (Exception e) {
                            if (AllConfigs.isDebug()) {
                                System.err.println("error sql " + createIndex);
                                e.printStackTrace();
                            }
                        }
                        TimeUnit.SECONDS.sleep(5);
                    }
                    try {
                        if (!isUsing && SearchUtil.getInstance().getStatus() == SearchUtil.NORMAL) {
                            stmt.execute("CREATE INDEX IF NOT EXISTS cache_index on cache(PATH);");
                        }
                    } catch (Exception e) {
                        if (AllConfigs.isDebug()) {
                            System.err.println("error create cache index");
                            e.printStackTrace();
                        }
                    }
                    TimeUnit.SECONDS.sleep(5);
                }
            } catch (Exception ignored) {
            }
        });
    }

    private void sendSignalAndShowCommandThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            //缓存和常用文件夹搜索线程
            //停顿时间0.5s，每一次输入会更新一次startTime，该线程记录endTime
            try {
                while (AllConfigs.isNotMainExit()) {
                    long endTime = System.currentTimeMillis();
                    if ((endTime - startTime > 500) && startSignal) {
                        startSignal = false; //开始搜索 计时停止
                        resultCount.set(0);
                        labelCount.set(0);
                        currentLabelSelectedPosition.set(0);
                        clearLabel();
                        if (!getTextFieldText().isEmpty()) {
                            setLabelChosen(label1);
                        } else {
                            clearLabel();
                        }
                        listResults.clear();
                        listResultsCopy.clear();
                        tempResults.clear();
                        String text = getTextFieldText();
                        if (search.getStatus() == SearchUtil.NORMAL) {
                            if (runningMode.get() == AllConfigs.RunningMode.COMMAND_MODE) {
                                //去掉冒号
                                runInternalCommand(text.substring(1).toLowerCase());
                                LinkedHashSet<String> cmdSet = new LinkedHashSet<>(AllConfigs.getCmdSet());
                                cmdSet.add(":clearbin;" + TranslateUtil.getInstance().getTranslation("Clear the recycle bin"));
                                cmdSet.add(":update;" + TranslateUtil.getInstance().getTranslation("Update file index"));
                                cmdSet.add(":help;" + TranslateUtil.getInstance().getTranslation("View help"));
                                cmdSet.add(":version;" + TranslateUtil.getInstance().getTranslation("View Version"));
                                for (String i : cmdSet) {
                                    if (i.startsWith(text)) {
                                        resultCount.incrementAndGet();
                                        String result = TranslateUtil.getInstance().getTranslation("Run command") + i;
                                        listResults.add(result);
                                        listResultsCopy.add(result);
                                    }
                                    String[] cmdInfo = semicolon.split(i);
                                    if (cmdInfo[0].equals(text)) {
                                        detectShowingModeAndClose();
                                        openWithAdmin(cmdInfo[1]);
                                    }
                                }
                                showResults(true, false, false, false,
                                        false, false, false, false);
                            } else if (runningMode.get() == AllConfigs.RunningMode.NORMAL_MODE) {
                                isStartSearchLocal = true;
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
                                isCacheAndPrioritySearched = false;
                                searchPriorityFolder();
                                searchCache();
                                isCacheAndPrioritySearched = true;
                            } else if (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
                                String result;
                                if (currentUsingPlugin != null) {
                                    while (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
                                        try {
                                            if ((result = currentUsingPlugin.pollFromResultQueue()) != null) {
                                                if (isResultNotRepeat(result)) {
                                                    listResults.add(result);
                                                    listResultsCopy.add(result);
                                                    resultCount.incrementAndGet();
                                                }
                                            }
                                        } catch (NullPointerException ignored) {
                                        }
                                        TimeUnit.MILLISECONDS.sleep(10);
                                    }
                                }
                            }

                            showResults(true, false, false, false,
                                    false, false, false, false);

                        } else if (search.getStatus() == SearchUtil.VACUUM) {
                            setLabelChosen(label1);
                            label1.setText(TranslateUtil.getInstance().getTranslation("Organizing database"));
                        } else if (search.getStatus() == SearchUtil.MANUAL_UPDATE) {
                            setLabelChosen(label1);
                            label1.setText(TranslateUtil.getInstance().getTranslation("Updating file index") + "...");
                        }

                        if (search.getStatus() != SearchUtil.NORMAL) {
                            //开启线程等待搜索完成
                            addSearchWaiter();
                            clearLabel();
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            } catch (InterruptedException ignored) {

            } catch (URISyntaxException | IOException e) {
                e.printStackTrace();
            }
        });
    }

    private void addRecordsToDatabaseThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            //检测文件添加线程
            String filesToAdd;
            int count;
            try (BufferedReader readerAdd = new BufferedReader(new InputStreamReader(
                    new FileInputStream(AllConfigs.getTmp().getAbsolutePath() + File.separator + "fileAdded.txt"),
                    StandardCharsets.UTF_8))) {
                while (AllConfigs.isNotMainExit()) {
                    if (search.getStatus() == SearchUtil.NORMAL) {
                        count = 0;
                        while ((filesToAdd = readerAdd.readLine()) != null) {
                            search.addFileToDatabase(filesToAdd);
                            count++;
                            if (count > 3000) {
                                break;
                            }
                            if (AllConfigs.isDebug()) {
                                System.out.println("添加" + filesToAdd);
                            }
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(1);
                }
            } catch (IOException | InterruptedException e) {
                if (AllConfigs.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void deleteRecordsToDatabaseThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            String filesToRemove;
            int count;
            try (BufferedReader readerRemove = new BufferedReader(new InputStreamReader(
                    new FileInputStream(AllConfigs.getTmp().getAbsolutePath() + File.separator + "fileRemoved.txt"),
                    StandardCharsets.UTF_8))) {
                while (AllConfigs.isNotMainExit()) {
                    if (search.getStatus() == SearchUtil.NORMAL) {
                        count = 0;
                        while ((filesToRemove = readerRemove.readLine()) != null) {
                            search.removeFileFromDatabase(filesToRemove);
                            count++;
                            if (count > 3000) {
                                break;
                            }
                            if (AllConfigs.isDebug()) {
                                System.out.println("删除" + filesToRemove);
                            }
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(1);
                }
            } catch (InterruptedException | IOException e) {
                if (AllConfigs.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void checkTimeAndExecuteSqlCommandsThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            // 时间检测线程
            final long updateTimeLimit = AllConfigs.getUpdateTimeLimit();
            while (AllConfigs.isNotMainExit()) {
                try (Statement stmt = SQLiteUtil.getStatement()) {
                    while (AllConfigs.isNotMainExit()) {
                        if (search.getStatus() == SearchUtil.NORMAL) {
                            if (search.getStatus() == SearchUtil.NORMAL) {
                                search.executeAllCommands(stmt);
                            }
                        }
                        TimeUnit.SECONDS.sleep(updateTimeLimit);
                    }
                } catch (Exception e) {
                    if (AllConfigs.isDebug()) {
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    private void recreateIndexThread() {
        //搜索本地数据线程
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (AllConfigs.isNotMainExit()) {
                    if (search.getStatus() == SearchUtil.MANUAL_UPDATE) {
                        try (Statement stmt = SQLiteUtil.getStatement()) {
                            search.updateLists(AllConfigs.getIgnorePath(), AllConfigs.getSearchDepth(), stmt);
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (Exception e) {
                if (AllConfigs.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
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
        String name = getFileName(path);
        if (searchCase == null || searchCase.length == 0) {
            return isMatched(name, true);
        } else {
            if (isMatched(name, true)) {
                for (String eachCase : searchCase) {
                    switch (eachCase) {
                        case "f":
                            if (!isFile(path)) {
                                return false;
                            }
                            break;
                        case "d":
                            if (!isDirectory(path)) {
                                return false;
                            }
                            break;
                        case "full":
                            if (!name.equalsIgnoreCase(searchText)) {
                                return false;
                            }
                            break;
                        case "case":
                            if (!isMatched(name, false)) {
                                return false;
                            }
                    }
                }
                //所有规则均已匹配
                return true;
            }
        }
        return false;
    }

    /**
     * 检查当前文件路径是否已被加入到listResults中
     *
     * @param result 文件路径
     * @return true如果还未被加入
     */
    private boolean isResultNotRepeat(String result) {
        return !(tempResults.contains(result) || listResultsCopy.contains(result));
    }


    /**
     * 检查文件路径是否匹配然后加入到列表
     *
     * @param path        文件路径
     * @param isPutToTemp 是否放到临时容器，在搜索优先文件夹和cache时为false，其他为true
     * @return 当前容器中结果数量是否已超过100个
     */
    private boolean checkIsMatchedAndAddToList(String path, boolean isPutToTemp) {
        if (check(path)) {
            if (isPutToTemp) {
                if (isExist(path)) {
                    tempResults.add(path);
                } else {
                    search.removeFileFromDatabase(path);
                }
            } else {
                if (isExist(path)) {
                    if (isResultNotRepeat(path)) {
                        resultCount.incrementAndGet();
                        listResults.add(path);
                        listResultsCopy.add(path);
                    }
                } else {
                    search.removeFileFromDatabase(path);
                }
            }
        }
        return resultCount.get() >= 100;
    }

    /**
     * 搜索数据酷并加入到tempQueue中
     *
     * @param time   开始搜索时间，用于检测用于重新输入匹配信息后快速停止
     * @param column 数据库表
     * @throws SQLException 错误处理
     */
    private void searchAndAddToTempResults(long time, String column) throws SQLException {
        //为label添加结果
        String each;
        boolean isResultsExcessive = false;
        String pSql = "SELECT PATH FROM " + column + ";";
        try (PreparedStatement pStmt = SQLiteUtil.getConnection().prepareStatement(pSql); ResultSet resultSet = pStmt.executeQuery()) {
            while (resultSet.next()) {
                each = resultSet.getString("PATH");
                if (search.getStatus() == SearchUtil.NORMAL) {
                    isResultsExcessive = checkIsMatchedAndAddToList(each, true);
                }
                //用户重新输入了信息
                if (isResultsExcessive || (startTime > time)) {
                    break;
                }
            }
        }
    }

    /**
     * 显示窗口
     *
     * @param isGrabFocus 是否强制抓取焦点（由操作系统决定，有时想抓也抓不到）
     */
    public void showSearchbar(boolean isGrabFocus) {
        searchBar.setAutoRequestFocus(isGrabFocus);
        setVisible(true);
        if (isGrabFocus) {
            searchBar.toFront();
            searchBar.requestFocusInWindow();
            textField.requestFocusInWindow();
        } else {
            textField.transferFocus();
            searchBar.transferFocus();
        }
        textField.setCaretPosition(0);
        isUsing = true;
        startTime = System.currentTimeMillis();
        visibleStartTime = startTime;
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
        ImageIcon icon = GetIconUtil.getInstance().getBigIcon(path, iconSideLength, iconSideLength);
        label.setIcon(icon);
        label.setBorder(border);
        if (name.length() >= 32) {
            name = name.substring(0, 32) + "...";
        }
        label.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>" + getParentPath(path) + "</body></html>");
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
        String[] info = semicolon.split(command);
        String path = info[1];
        String name = info[0];
        ImageIcon imageIcon;
        try {
            imageIcon = GetIconUtil.getInstance().getCommandIcon(colon.split(name)[1], iconSideLength, iconSideLength);
        } catch (NullPointerException e) {
            imageIcon = GetIconUtil.getInstance().getBigIcon(path, iconSideLength, iconSideLength);
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
        if (runningMode.get() == AllConfigs.RunningMode.NORMAL_MODE) {
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
        } else if (runningMode.get() == AllConfigs.RunningMode.COMMAND_MODE) {
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
        } else if (runningMode.get() == AllConfigs.RunningMode.PLUGIN_MODE) {
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
    private void clearLabel() {
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
        File name = new File(path);
        if (name.exists()) {
            try {
                String command = name.getAbsolutePath();
                String start = "cmd.exe /c start " + command.substring(0, 2);
                String end = "\"" + command.substring(2) + "\"";
                Runtime.getRuntime().exec(start + end, null, name.getParentFile());
            } catch (IOException e) {
                //打开上级文件夹
                try {
                    Runtime.getRuntime().exec("explorer.exe /select, \"" + name.getAbsolutePath() + "\"");
                } catch (IOException e1) {
                    JOptionPane.showMessageDialog(null, TranslateUtil.getInstance().getTranslation("Execute failed"));
                    if (AllConfigs.isDebug()) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    /**
     * 以普通权限运行文件，失败则打开文件位置
     *
     * @param path 文件路径
     */
    private void openWithoutAdmin(String path) {
        if (isExist(path)) {
            try {
                if (path.toLowerCase().endsWith(".lnk")) {
                    String command = "explorer.exe " + "\"" + path + "\"";
                    Runtime.getRuntime().exec(command);
                } else if (path.toLowerCase().endsWith(".url")) {
                    Desktop desktop;
                    if (Desktop.isDesktopSupported()) {
                        desktop = Desktop.getDesktop();
                        desktop.open(new File(path));
                    }
                } else {
                    //创建快捷方式到临时文件夹，打开后删除
                    createShortCut(path, AllConfigs.getTmp().getAbsolutePath() + File.separator + "open");
                    Runtime.getRuntime().exec("explorer.exe " + "\"" + AllConfigs.getTmp().getAbsolutePath() + File.separator + "open.lnk" + "\"");
                }
            } catch (Exception e) {
                //打开上级文件夹
                try {
                    Runtime.getRuntime().exec("explorer.exe /select, \"" + path + "\"");
                } catch (IOException e1) {
                    if (AllConfigs.isDebug()) {
                        e1.printStackTrace();
                    }
                }
            }
        }
    }

    /**
     * 创建需要打开的文件的快捷方式
     *
     * @param fileOrFolderPath  文件路径
     * @param writeShortCutPath 保存快捷方式的位置
     * @throws Exception 创建错误
     */
    private void createShortCut(String fileOrFolderPath, String writeShortCutPath) throws Exception {
        File shortcutGen = new File("user/shortcutGenerator.vbs");
        String shortcutGenPath = shortcutGen.getAbsolutePath();
        String start = "cmd.exe /c start " + shortcutGenPath.substring(0, 2);
        String end = "\"" + shortcutGenPath.substring(2) + "\"";
        String commandToGenLnk = start + end + " /target:" + "\"" + fileOrFolderPath + "\"" + " " + "/shortcut:" + "\"" + writeShortCutPath + "\"" + " /workingdir:" + "\"" + fileOrFolderPath.substring(0, fileOrFolderPath.lastIndexOf(File.separator)) + "\"";
        Runtime.getRuntime().exec("cmd.exe " + commandToGenLnk);
        TimeUnit.MILLISECONDS.sleep(500);
    }

    private String getFileName(String path) {
        if (path != null) {
            int index = path.lastIndexOf(File.separator);
            return path.substring(index + 1);
        }
        return "";
    }

    /**
     * 获取文件名对应的ascii总和
     *
     * @param path 文件路径
     * @return ascii总和
     */
    private int getAscIISum(String path) {
        if (path != null) {
            path = path.toUpperCase();
            if (path.contains(";")) {
                path = path.replace(";", "");
            }
            return GetAscII.INSTANCE.getAscII(path);
        }
        return 0;
    }

    /**
     * 保存当前文件路径到数据库缓存
     *
     * @param content 文件路径
     */
    private void saveCache(String content) {
        if (cacheNum.get() < AllConfigs.getCacheNumLimit()) {
            search.addFileToCache(content);
            SettingsFrame.getInstance().addCache(content);
        }
    }

    /**
     * 获取数据库缓存条目数量，用于判断软件是否还能继续写入缓存
     */
    private void initCacheNum() {
        try (PreparedStatement pStmt = SQLiteUtil.getConnection().prepareStatement("SELECT PATH FROM cache;");
             ResultSet resultSet = pStmt.getResultSet()) {
            while (resultSet.next()) {
                cacheNum.incrementAndGet();
            }
        } catch (SQLException throwables) {
            if (AllConfigs.isDebug()) {
                throwables.printStackTrace();
            }
        }
    }

    /**
     * 从缓存中搜索结果并将匹配的放入listResults
     */
    private void searchCache() {
        try (Statement stmt = SQLiteUtil.getStatement(); ResultSet resultSet = stmt.executeQuery("SELECT PATH FROM cache;")) {
            while (resultSet.next()) {
                String eachCache = resultSet.getString("PATH");
                if (!(isExist(eachCache))) {
                    search.removeFileFromCache(eachCache);
                } else {
                    checkIsMatchedAndAddToList(eachCache, false);
                }
            }
        } catch (Exception throwables) {
            if (AllConfigs.isDebug()) {
                throwables.printStackTrace();
            }
        }
    }

    /**
     * 判断文件路径是否满足当前匹配结果（该方法由check（String）方法使用），检查文件路径使用check（String）方法。
     *
     * @param name         文件路径
     * @param isIgnoreCase 是否忽略大小谢
     * @return true如果匹配成功
     * @see #check(String);
     */
    private boolean isMatched(String name, boolean isIgnoreCase) {
        for (String each : keywords) {
            if (isIgnoreCase) {
                if (!name.toLowerCase().contains(each.toLowerCase())) {
                    return false;
                }
            } else {
                if (!name.contains(each)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 搜索优先文件夹
     */
    private void searchPriorityFolder() {
        File path = new File(AllConfigs.getPriorityFolder());
        boolean exist = path.exists();
        AtomicInteger taskNum = new AtomicInteger(0);
        if (exist) {
            ConcurrentLinkedQueue<String> listRemain = new ConcurrentLinkedQueue<>();
            File[] files = path.listFiles();
            if (!(null == files || files.length == 0)) {
                for (File each : files) {
                    checkIsMatchedAndAddToList(each.getAbsolutePath(), false);
                    if (each.isDirectory()) {
                        listRemain.add(each.getAbsolutePath());
                    }
                }
                int threadCount = 8;
                for (int i = 0; i < threadCount; ++i) {
                    CachedThreadPool.getInstance().executeTask(() -> {
                        long startSearchTime = System.currentTimeMillis();
                        while (!listRemain.isEmpty()) {
                            String remain = listRemain.poll();
                            if (remain != null) {
                                File[] allFiles = new File(remain).listFiles();
                                if (!(allFiles == null || allFiles.length == 0)) {
                                    for (File each : allFiles) {
                                        checkIsMatchedAndAddToList(each.getAbsolutePath(), false);
                                        if (startTime > startSearchTime) {
                                            listRemain.clear();
                                            break;
                                        }
                                        if (each.isDirectory()) {
                                            listRemain.add(each.getAbsolutePath());
                                        }
                                    }
                                }
                            }
                        }
                        taskNum.incrementAndGet();
                    });
                }
                //等待所有线程完成
                try {
                    int count = 0;
                    while (taskNum.get() != threadCount) {
                        TimeUnit.MILLISECONDS.sleep(10);
                        count++;
                        if (count >= 10 || (!AllConfigs.isNotMainExit())) {
                            break;
                        }
                    }
                } catch (InterruptedException ignored) {
                }
            }
        }
    }

    /**
     * 设置窗口透明度
     *
     * @param trans 透明度
     */
    protected void setTransparency(float trans) {
        searchBar.setOpacity(trans);
    }

    private void clearTextFieldText() {
        Runnable clear = () -> {
            try {
                textField.setText(null);
            } catch (Exception ignored) {
            }
        };
        SwingUtilities.invokeLater(clear);
    }

    /**
     * 检测当前模式并重置状态
     */
    private void detectShowingModeAndClose() {
        if (showingMode.get() == AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING) {
            closeSearchBar();
        } else if (showingMode.get() == AllConfigs.ShowingSearchBarMode.EXPLORER_ATTACH) {
            closeWithoutHideSearchBar();
        }
    }

    private void setVisible(boolean b) {
        searchBar.setVisible(b);
    }

    /**
     * 重置所有状态并关闭窗口
     */
    public void closeSearchBar() {
        if (isVisible()) {
            clearLabel();
            setVisible(false);
        }
        clearTextFieldText();
        startTime = System.currentTimeMillis();//结束搜索
        isUsing = false;
        labelCount.set(0);
        resultCount.set(0);
        currentLabelSelectedPosition.set(0);
        listResults.clear();
        listResultsCopy.clear();
        tempResults.clear();
        commandQueue.clear();
        isUserPressed = false;
        isLockMouseMotion = false;
        isOpenLastFolderPressed = false;
        isRunAsAdminPressed = false;
        isCopyPathPressed = false;
        startSignal = false;
        isCacheAndPrioritySearched = false;
        isStartSearchLocal = false;
        isWaiting = false;
    }

    /**
     * 重置所有状态但不关闭窗口
     */
    private void closeWithoutHideSearchBar() {
        clearLabel();
        clearTextFieldText();
        startTime = System.currentTimeMillis();//结束搜索
        isUsing = true;
        labelCount.set(0);
        resultCount.set(0);
        currentLabelSelectedPosition.set(0);
        listResults.clear();
        listResultsCopy.clear();
        tempResults.clear();
        commandQueue.clear();
        isUserPressed = false;
        isLockMouseMotion = false;
        isOpenLastFolderPressed = false;
        isRunAsAdminPressed = false;
        isCopyPathPressed = false;
        startSignal = false;
        isCacheAndPrioritySearched = false;
        isStartSearchLocal = false;
        isWaiting = false;
    }

    /**
     * 判断窗口是否可见
     *
     * @return true如果可见 否则false
     */
    public boolean isVisible() {
        return searchBar.isVisible();
    }

    private String getParentPath(String path) {
        File f = new File(path);
        return f.getParent();
    }

    private boolean isFile(String text) {
        File file = new File(text);
        return file.isFile();
    }

    private boolean isDirectory(String text) {
        File file = new File(text);
        return file.isDirectory();
    }

    protected void setFontColorWithCoverage(int colorNum) {
        fontColorWithCoverage = new Color(colorNum);
    }

    protected void setDefaultBackgroundColor(int colorNum) {
        backgroundColor = new Color(colorNum);
    }

    protected void setLabelColor(int colorNum) {
        labelColor = new Color(colorNum);
    }

    protected void setFontColor(int colorNum) {
        fontColor = new Color(colorNum);
    }

    protected void setSearchBarColor(int colorNum) {
        textField.setBackground(new Color(colorNum));
    }

    public int getShowingMode() {
        return showingMode.get();
    }
}

