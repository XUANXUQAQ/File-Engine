package FileEngine.frames;


import FileEngine.dllInterface.FileMonitor;
import FileEngine.dllInterface.GetAscII;
import FileEngine.dllInterface.IsLocalDisk;
import FileEngine.getIcon.GetIconUtil;
import FileEngine.pluginSystem.Plugin;
import FileEngine.pluginSystem.PluginUtil;
import FileEngine.SQLiteConfig.SQLiteUtil;
import FileEngine.search.SearchUtil;
import FileEngine.threadPool.CachedThreadPool;

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
import java.util.Queue;
import java.util.*;
import java.util.concurrent.*;
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
    private static volatile boolean timer = false;
    private static volatile boolean isUserPressed = false;
    private static Border border;
    private final JFrame searchBar;
    private final JLabel label1;
    private final JLabel label2;
    private final JLabel label3;
    private final JLabel label4;
    private final JLabel label5;
    private final JLabel label6;
    private final JLabel label7;
    private final JLabel label8;
    private static AtomicInteger labelCount;
    private final JTextField textField;
    private Color labelColor;
    private Color backgroundColor;
    private Color fontColorWithCoverage;
    private Color fontColor;
    private volatile long startTime = 0;
    private Thread searchWaiter = null;
    private static Pattern semicolon;
    private static Pattern resultSplit;
    private static Pattern blank;
    private static AtomicInteger runningMode;
    private final JPanel panel;
    private long mouseWheelTime = 0;
    private final int iconSideLength;
    private long visibleStartTime = 0;
    private final ConcurrentLinkedQueue<String> tempResults;
    private final ConcurrentLinkedQueue<String> commandQueue;
    private final CopyOnWriteArrayList<String> listResults;
    private final ConcurrentHashMap<String, PreparedStatement> sqlCache;
    private volatile String[] searchCase;
    private volatile String searchText;
    private volatile String[] keywords;
    private static SearchUtil search;
    private static TaskBar taskBar;
    private static AtomicInteger resultCount;
    private static AtomicInteger currentLabelSelectedPosition;
    private static volatile Plugin currentUsingPlugin;

    private static final int NORMAL_MODE = 0;
    private static final int COMMAND_MODE = 1;
    private static final int PLUGIN_MODE = 2;

    private static class SearchBarBuilder {
        private static final SearchBar instance = new SearchBar();
    }

    private SearchBar() {
        listResults = new CopyOnWriteArrayList<>();
        tempResults = new ConcurrentLinkedQueue<>();
        commandQueue = new ConcurrentLinkedQueue<>();
        sqlCache = new ConcurrentHashMap<>();
        border = BorderFactory.createLineBorder(new Color(73, 162, 255, 255));
        searchBar = new JFrame();
        labelCount = new AtomicInteger(0);
        resultCount = new AtomicInteger(0);
        runningMode = new AtomicInteger(0);
        currentLabelSelectedPosition = new AtomicInteger(0);
        semicolon = Pattern.compile(";");
        resultSplit = Pattern.compile(":");
        blank = Pattern.compile(" ");
        panel = new JPanel();

        search = SearchUtil.getInstance();
        taskBar = TaskBar.getInstance();

        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
        int width = screenSize.width;
        int height = screenSize.height;
        int searchBarWidth = (int) (width * 0.4);
        int searchBarHeight = (int) (height * 0.5);
        int positionX = width / 2 - searchBarWidth / 2;
        int positionY = height / 2 - searchBarHeight / 2;


        labelColor = new Color(SettingsFrame.getLabelColor());
        fontColorWithCoverage = new Color(SettingsFrame.getFontColorWithCoverage());
        backgroundColor = new Color(SettingsFrame.getDefaultBackgroundColor());
        fontColor = new Color(SettingsFrame.getFontColor());

        //frame
        searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
        searchBar.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        searchBar.setUndecorated(true);
        searchBar.getRootPane().setWindowDecorationStyle(JRootPane.NONE);
        searchBar.setBackground(null);
        searchBar.setOpacity(SettingsFrame.getTransparency());
        searchBar.setContentPane(panel);
        searchBar.setType(JFrame.Type.UTILITY);


        //labels
        Font font = new Font("Microsoft JhengHei", Font.BOLD, (int) ((height * 0.1) / 96 * 72) / 4);
        label1 = new JLabel();
        label2 = new JLabel();
        label3 = new JLabel();
        label4 = new JLabel();
        label5 = new JLabel();
        label6 = new JLabel();
        label7 = new JLabel();
        label8 = new JLabel();

        int labelHeight = searchBarHeight / 9;
        initLabel(font, searchBarWidth, labelHeight, labelHeight, label1);
        initLabel(font, searchBarWidth, labelHeight, labelHeight * 2, label2);
        initLabel(font, searchBarWidth, labelHeight, labelHeight * 3, label3);
        initLabel(font, searchBarWidth, labelHeight, labelHeight * 4, label4);
        initLabel(font, searchBarWidth, labelHeight, labelHeight * 5, label5);
        initLabel(font, searchBarWidth, labelHeight, labelHeight * 6, label6);
        initLabel(font, searchBarWidth, labelHeight, labelHeight * 7, label7);
        initLabel(font, searchBarWidth, labelHeight, labelHeight * 8, label8);

        iconSideLength = labelHeight / 3; //定义图标边长

        URL icon = TaskBar.class.getResource("/icons/taskbar_32x32.png");
        Image image = new ImageIcon(icon).getImage();
        searchBar.setIconImage(image);
        Color transparentColor = new Color(0, 0, 0, 0);
        searchBar.setBackground(transparentColor);


        //TextField
        textField = new JTextField(300);
        textField.setSize(searchBarWidth - 6, labelHeight - 5);
        Font textFieldFont = new Font("Microsoft JhengHei", Font.PLAIN, (int) ((height * 0.2) / 96 * 72) / 4);
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
                    if (SettingsFrame.isLoseFocusClose()) {
                        closedTodo();
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

    private void initLabel(Font font, int width, int height, int positionY, JLabel label) {
        label.setSize(width, height);
        label.setLocation(0, positionY);
        label.setFont(font);
        label.setForeground(fontColor);
        label.setOpaque(true);
        label.setBackground(null);
        label.setFocusable(false);
    }

    private void addSearchBarMouseListener() {
        searchBar.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
            }

            @Override
            public void mousePressed(MouseEvent e) {
                int count = e.getClickCount();
                if (count == 2) {
                    searchBar.setVisible(false);
                    if (runningMode.get() != PLUGIN_MODE) {
                        //enter被点击
                        searchBar.setVisible(false);
                        String res = listResults.get(labelCount.get());
                        if (runningMode.get() == NORMAL_MODE) {
                            if (isOpenLastFolderPressed) {
                                //打开上级文件夹
                                File open = new File(res);
                                try {
                                    Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                } catch (IOException e1) {
                                    e1.printStackTrace();
                                }
                            } else if (SettingsFrame.isDefaultAdmin() || isRunAsAdminPressed) {
                                openWithAdmin(res);
                            } else if (isCopyPathPressed) {
                                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                                Transferable trans = new StringSelection(res);
                                clipboard.setContents(trans, null);
                            } else {
                                if (res.endsWith(".bat") || res.endsWith(".cmd")) {
                                    openWithAdmin(res);
                                } else {
                                    openWithoutAdmin(res);
                                }
                            }
                            saveCache(res + ';');
                        } else if (runningMode.get() == COMMAND_MODE) {
                            File open = new File(semicolon.split(res)[1]);
                            if (isOpenLastFolderPressed) {
                                //打开上级文件夹
                                try {
                                    Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                } catch (IOException e1) {
                                    JOptionPane.showMessageDialog(null, SettingsFrame.getTranslation("Execute failed"));
                                }
                            } else if (SettingsFrame.isDefaultAdmin() || isRunAsAdminPressed) {
                                openWithAdmin(open.getAbsolutePath());
                            } else if (isCopyPathPressed) {
                                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                                Transferable trans = new StringSelection(res);
                                clipboard.setContents(trans, null);
                            } else {
                                if (res.endsWith(".bat") || res.endsWith(".cmd")) {
                                    openWithAdmin(open.getAbsolutePath());
                                } else {
                                    openWithoutAdmin(open.getAbsolutePath());
                                }
                            }
                        }
                        closedTodo();
                    } else if (runningMode.get() == PLUGIN_MODE) {
                        if (currentUsingPlugin != null) {
                            if (!listResults.isEmpty()) {
                                currentUsingPlugin.mousePressed(e, listResults.get(labelCount.get()));
                            }
                        }
                    }
                    closedTodo();
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                if (runningMode.get() == PLUGIN_MODE) {
                    if (currentUsingPlugin != null) {
                        if (!listResults.isEmpty()) {
                            currentUsingPlugin.mouseReleased(e, listResults.get(labelCount.get()));
                        }
                    }
                }
            }

            @Override
            public void mouseEntered(MouseEvent e) {
            }

            @Override
            public void mouseExited(MouseEvent e) {
            }
        });
    }

    private String getTextFieldText() {
        return textField.getText();
    }

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
                if (!listResults.isEmpty()) {
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

                                //System.out.println(labelCount);
                                if (labelCount.get() >= listResults.size()) {
                                    labelCount.set(listResults.size() - 1);
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

                                    if (labelCount.get() >= listResults.size()) {
                                        labelCount.set(listResults.size() - 1);
                                    }
                                    if (labelCount.get() <= 0) {
                                        labelCount.set(0);
                                    }
                                    moveDownward(getCurrentLabelPos());
                                }
                            }
                        }
                    } else if (10 == key) {
                        if (runningMode.get() != PLUGIN_MODE) {
                            //enter被点击
                            searchBar.setVisible(false);
                            String res = listResults.get(labelCount.get());
                            if (runningMode.get() == NORMAL_MODE) {
                                if (isOpenLastFolderPressed) {
                                    //打开上级文件夹
                                    File open = new File(res);
                                    try {
                                        Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                    } catch (IOException e) {
                                        e.printStackTrace();
                                    }
                                } else if (SettingsFrame.isDefaultAdmin() || isRunAsAdminPressed) {
                                    openWithAdmin(res);
                                } else if (isCopyPathPressed) {
                                    Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                                    Transferable trans = new StringSelection(res);
                                    clipboard.setContents(trans, null);
                                } else {
                                    if (res.endsWith(".bat") || res.endsWith(".cmd")) {
                                        openWithAdmin(res);
                                    } else {
                                        openWithoutAdmin(res);
                                    }
                                }
                                saveCache(res + ';');
                            } else if (runningMode.get() == COMMAND_MODE) {
                                File open = new File(semicolon.split(res)[1]);
                                if (isOpenLastFolderPressed) {
                                    //打开上级文件夹
                                    try {
                                        Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                    } catch (IOException e) {
                                        JOptionPane.showMessageDialog(null, SettingsFrame.getTranslation("Execute failed"));
                                    }
                                } else if (SettingsFrame.isDefaultAdmin() || isRunAsAdminPressed) {
                                    openWithAdmin(open.getAbsolutePath());
                                } else if (isCopyPathPressed) {
                                    Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                                    Transferable trans = new StringSelection(res);
                                    clipboard.setContents(trans, null);
                                } else {
                                    if (res.endsWith(".bat") || res.endsWith(".cmd")) {
                                        openWithAdmin(open.getAbsolutePath());
                                    } else {
                                        openWithoutAdmin(open.getAbsolutePath());
                                    }
                                }
                            }
                            closedTodo();
                        }
                    } else if (SettingsFrame.getOpenLastFolderKeyCode() == key) {
                        //打开上级文件夹热键被点击
                        isOpenLastFolderPressed = true;
                    } else if (SettingsFrame.getRunAsAdminKeyCode() == key) {
                        //以管理员方式运行热键被点击
                        isRunAsAdminPressed = true;
                    } else if (SettingsFrame.getCopyPathKeyCode() == key) {
                        isCopyPathPressed = true;
                    }
                }
                if (runningMode.get() == PLUGIN_MODE) {
                    if (key != 38 && key != 40) {
                        if (key == 10) {
                            closedTodo();
                        }
                        if (currentUsingPlugin != null) {
                            if (!listResults.isEmpty()) {
                                currentUsingPlugin.keyPressed(arg0, listResults.get(labelCount.get()));
                            }
                        }
                    }
                }
            }

            @Override
            public void keyReleased(KeyEvent arg0) {

                int key = arg0.getKeyCode();
                if (SettingsFrame.getOpenLastFolderKeyCode() == key) {
                    //复位按键状态
                    isOpenLastFolderPressed = false;
                } else if (SettingsFrame.getRunAsAdminKeyCode() == key) {
                    isRunAsAdminPressed = false;
                } else if (SettingsFrame.getCopyPathKeyCode() == key) {
                    isCopyPathPressed = false;
                }

                if (runningMode.get() == PLUGIN_MODE) {
                    if (key != 38 && key != 40) {
                        if (currentUsingPlugin != null) {
                            if (!listResults.isEmpty()) {
                                currentUsingPlugin.keyReleased(arg0, listResults.get(labelCount.get()));
                            }
                        }
                    }
                }
            }

            @Override
            public void keyTyped(KeyEvent arg0) {
                if (runningMode.get() == PLUGIN_MODE) {
                    int key = arg0.getKeyCode();
                    if (key != 38 && key != 40) {
                        if (currentUsingPlugin != null) {
                            if (!listResults.isEmpty()) {
                                currentUsingPlugin.keyTyped(arg0, listResults.get(labelCount.get()));
                            }
                        }
                    }
                }
            }
        });
    }

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
                    if (mousePosition < listResults.size()) {
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
                                    label1.setBackground(labelColor);
                                }
                                if (isLabelNotEmpty(label2)) {
                                    label2.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label3)) {
                                    label3.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label4)) {
                                    label4.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label5)) {
                                    label5.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label6)) {
                                    label6.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label7)) {
                                    label7.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label8)) {
                                    label8.setBackground(backgroundColor);
                                }
                                break;
                            case 1:
                                if (isLabelNotEmpty(label1)) {
                                    label1.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label2)) {
                                    label2.setBackground(labelColor);
                                }
                                if (isLabelNotEmpty(label3)) {
                                    label3.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label4)) {
                                    label4.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label5)) {
                                    label5.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label6)) {
                                    label6.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label7)) {
                                    label7.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label8)) {
                                    label8.setBackground(backgroundColor);
                                }
                                break;
                            case 2:
                                if (isLabelNotEmpty(label1)) {
                                    label1.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label2)) {
                                    label2.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label3)) {
                                    label3.setBackground(labelColor);
                                }
                                if (isLabelNotEmpty(label4)) {
                                    label4.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label5)) {
                                    label5.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label6)) {
                                    label6.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label7)) {
                                    label7.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label8)) {
                                    label8.setBackground(backgroundColor);
                                }
                                break;
                            case 3:
                                if (isLabelNotEmpty(label1)) {
                                    label1.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label2)) {
                                    label2.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label3)) {
                                    label3.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label4)) {
                                    label4.setBackground(labelColor);
                                }
                                if (isLabelNotEmpty(label5)) {
                                    label5.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label6)) {
                                    label6.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label7)) {
                                    label7.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label8)) {
                                    label8.setBackground(backgroundColor);
                                }
                                break;
                            case 4:
                                if (isLabelNotEmpty(label1)) {
                                    label1.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label2)) {
                                    label2.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label3)) {
                                    label3.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label4)) {
                                    label4.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label5)) {
                                    label5.setBackground(labelColor);
                                }
                                if (isLabelNotEmpty(label6)) {
                                    label6.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label7)) {
                                    label7.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label8)) {
                                    label8.setBackground(backgroundColor);
                                }
                                break;
                            case 5:
                                if (isLabelNotEmpty(label1)) {
                                    label1.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label2)) {
                                    label2.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label3)) {
                                    label3.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label4)) {
                                    label4.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label5)) {
                                    label5.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label6)) {
                                    label6.setBackground(labelColor);
                                }
                                if (isLabelNotEmpty(label7)) {
                                    label7.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label8)) {
                                    label8.setBackground(backgroundColor);
                                }
                                break;
                            case 6:
                                if (isLabelNotEmpty(label1)) {
                                    label1.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label2)) {
                                    label2.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label3)) {
                                    label3.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label4)) {
                                    label4.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label5)) {
                                    label5.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label6)) {
                                    label6.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label7)) {
                                    label7.setBackground(labelColor);
                                }
                                if (isLabelNotEmpty(label8)) {
                                    label8.setBackground(backgroundColor);
                                }
                                break;
                            case 7:
                                if (isLabelNotEmpty(label1)) {
                                    label1.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label2)) {
                                    label2.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label3)) {
                                    label3.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label4)) {
                                    label4.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label5)) {
                                    label5.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label6)) {
                                    label6.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label7)) {
                                    label7.setBackground(backgroundColor);
                                }
                                if (isLabelNotEmpty(label8)) {
                                    label8.setBackground(labelColor);
                                }
                                break;
                        }
                    }
                }
            }
        });
    }

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
                if (listResults.size() > 8) {
                    return true;
                }
        }
        return isNextLabelValid;
    }

    private boolean isLabelNotEmpty(JLabel label) {
        boolean isEmpty;
        try {
            isEmpty = label.getText().isEmpty();
        } catch (NullPointerException e) {
            isEmpty = true;
        }
        return !isEmpty;
    }

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
                        if (labelCount.get() >= listResults.size()) {
                            labelCount.set(listResults.size() - 1);
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

                    //System.out.println(labelCount);
                    if (labelCount.get() >= listResults.size()) {
                        labelCount.set(listResults.size() - 1);
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
                    int size = listResults.size();
                    if (size == 2) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                    } else if (size == 3) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                    } else if (size == 4) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                    } else if (size == 5) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                    } else if (size == 6) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                    } else if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 1:
                    size = listResults.size();
                    if (size == 3) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                    } else if (size == 4) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                    } else if (size == 5) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                    } else if (size == 6) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                    } else if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 2:
                    size = listResults.size();
                    if (size == 4) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(labelColor);
                    } else if (size == 5) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(labelColor);
                        label5.setBackground(backgroundColor);
                    } else if (size == 6) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(labelColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                    } else if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(labelColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(labelColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 3:
                    size = listResults.size();
                    if (size == 5) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(labelColor);
                    } else if (size == 6) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(labelColor);
                        label6.setBackground(backgroundColor);
                    } else if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(labelColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(labelColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 4:
                    size = listResults.size();
                    if (size == 6) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(labelColor);
                    } else if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(labelColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(labelColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 5:
                    size = listResults.size();
                    if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(labelColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(labelColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 6:
                    label1.setBackground(backgroundColor);
                    label2.setBackground(backgroundColor);
                    label3.setBackground(backgroundColor);
                    label4.setBackground(backgroundColor);
                    label5.setBackground(backgroundColor);
                    label6.setBackground(backgroundColor);
                    label7.setBackground(backgroundColor);
                    label8.setBackground(labelColor);
                    break;
                case 7:
                    if (runningMode.get() == NORMAL_MODE) {
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
                            if (SettingsFrame.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    } else if (runningMode.get() == COMMAND_MODE) {
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
                            if (SettingsFrame.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    } else if (runningMode.get() == PLUGIN_MODE) {
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
                            if (SettingsFrame.isDebug()) {
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
                    if (runningMode.get() == NORMAL_MODE) {
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
                            if (SettingsFrame.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    } else if (runningMode.get() == COMMAND_MODE) {
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
                            if (SettingsFrame.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    } else if (runningMode.get() == PLUGIN_MODE) {
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
                            if (SettingsFrame.isDebug()) {
                                e.printStackTrace();
                            }
                        }
                    }
                    break;
                case 1:
                    size = listResults.size();
                    if (size == 2) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                    } else if (size == 3) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                    } else if (size == 4) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                    } else if (size == 5) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                    } else if (size == 6) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                    } else if (size == 7) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 2:
                    size = listResults.size();
                    if (size == 3) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                    } else if (size == 4) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                    } else if (size == 5) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                    } else if (size == 6) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                    } else if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(labelColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 3:
                    size = listResults.size();
                    if (size == 4) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                    } else if (size == 5) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                    } else if (size == 6) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                    } else if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(labelColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 4:
                    size = listResults.size();
                    if (size == 5) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(labelColor);
                        label5.setBackground(backgroundColor);
                    } else if (size == 6) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(labelColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                    } else if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(labelColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(labelColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 5:
                    size = listResults.size();
                    if (size == 6) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(labelColor);
                        label6.setBackground(backgroundColor);
                    } else if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(labelColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(labelColor);
                        label6.setBackground(backgroundColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 6:
                    size = listResults.size();
                    if (size == 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(labelColor);
                        label7.setBackground(backgroundColor);
                    } else if (size > 7) {
                        label1.setBackground(backgroundColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColor);
                        label4.setBackground(backgroundColor);
                        label5.setBackground(backgroundColor);
                        label6.setBackground(labelColor);
                        label7.setBackground(backgroundColor);
                        label8.setBackground(backgroundColor);
                    }
                    break;
                case 7:
                    label1.setBackground(backgroundColor);
                    label2.setBackground(backgroundColor);
                    label3.setBackground(backgroundColor);
                    label4.setBackground(backgroundColor);
                    label5.setBackground(backgroundColor);
                    label6.setBackground(backgroundColor);
                    label7.setBackground(labelColor);
                    label8.setBackground(backgroundColor);
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
                tempResults.clear();
                resultCount.set(0);
                labelCount.set(0);
                currentLabelSelectedPosition.set(0);
                isCacheAndPrioritySearched = false;
                startTime = System.currentTimeMillis();
                timer = true;
                String text = getTextFieldText();
                char first = text.charAt(0);
                if (first == ':') {
                    runningMode.set(COMMAND_MODE);
                } else if (first == '>') {
                    runningMode.set(PLUGIN_MODE);
                    String subText = text.substring(1);
                    String[] s = blank.split(subText);
                    currentUsingPlugin = PluginUtil.getPluginByIdentifier(s[0]);
                    int length = s.length;
                    if (currentUsingPlugin != null && length > 1) {
                        for (int i = 1; i < length - 1; ++i) {
                            strb.append(s[i]).append(" ");
                        }
                        strb.append(s[length - 1]);
                        currentUsingPlugin.textChanged(strb.toString());
                        strb.delete(0, strb.length());
                    }
                } else {
                    runningMode.set(NORMAL_MODE);
                }
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                clearLabel();
                listResults.clear();
                tempResults.clear();
                resultCount.set(0);
                labelCount.set(0);
                currentLabelSelectedPosition.set(0);
                isCacheAndPrioritySearched = false;
                String text = getTextFieldText();
                try {
                    char first = text.charAt(0);
                    if (first == ':') {
                        runningMode.set(COMMAND_MODE);
                    } else if (first == '>') {
                        runningMode.set(PLUGIN_MODE);
                        String subText = text.substring(1);
                        String[] s = blank.split(subText);
                        currentUsingPlugin = PluginUtil.getPluginByIdentifier(s[0]);
                        int length = s.length;
                        if (currentUsingPlugin != null && length > 1) {
                            for (int i = 1; i < length - 1; ++i) {
                                strb.append(s[i]).append(" ");
                            }
                            strb.append(s[length - 1]);
                            currentUsingPlugin.textChanged(strb.toString());
                            strb.delete(0, strb.length());
                        }
                    } else {
                        runningMode.set(NORMAL_MODE);
                    }
                } catch (StringIndexOutOfBoundsException e1) {
                    runningMode.set(NORMAL_MODE);
                }
                if (text.isEmpty()) {
                    panel.repaint();
                    resultCount.set(0);
                    labelCount.set(0);
                    currentLabelSelectedPosition.set(0);
                    startTime = System.currentTimeMillis();
                    timer = false;
                } else {
                    startTime = System.currentTimeMillis();
                    timer = true;
                }
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                startTime = System.currentTimeMillis();
                timer = false;
            }
        });
    }

    public void startMonitorDisk() {
        CachedThreadPool.getInstance().executeTask(() -> {
            File[] roots = File.listRoots();
            if (SettingsFrame.isAdmin()) {
                FileMonitor.INSTANCE.set_output(SettingsFrame.getTmp().getAbsolutePath());
                for (File root : roots) {
                    boolean isLocal = IsLocalDisk.INSTANCE.isLocalDisk(root.getAbsolutePath());
                    if (isLocal) {
                        FileMonitor.INSTANCE.monitor(root.getAbsolutePath());
                    }
                }
            } else {
                System.out.println("Not administrator, file monitoring function is turned off");
                taskBar.showMessage(SettingsFrame.getTranslation("Warning"), SettingsFrame.getTranslation("Not administrator, file monitoring function is turned off"));
            }
        });
    }

    private void initThreadPool() {
        //监控磁盘变化
        startMonitorDisk();

        CachedThreadPool.getInstance().executeTask(() -> {
            //合并搜索结果线程
            try {
                String record;
                while (SettingsFrame.isNotMainExit()) {
                    if (isCacheAndPrioritySearched) {
                        if (listResults.isEmpty()) {
                            while ((record = tempResults.poll()) != null) {
                                resultCount.incrementAndGet();
                                listResults.add(record);
                            }
                        } else {
                            while ((record = tempResults.poll()) != null) {
                                if (!listResults.contains(record)) {
                                    resultCount.incrementAndGet();
                                    listResults.add(record);
                                }
                            }
                        }
                    }
                    Thread.sleep(20);
                }
            } catch (Exception e) {
                if (SettingsFrame.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                String[] message;
                Plugin plugin;
                while (SettingsFrame.isNotMainExit()) {
                    Iterator<Plugin> iter = PluginUtil.getPluginMapIter();
                    while (iter.hasNext()) {
                        plugin = iter.next();
                        message = plugin.getMessage();
                        if (message != null) {
                            TaskBar.getInstance().showMessage(message[0], message[1]);
                        }
                    }
                    Thread.sleep(50);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            //锁住MouseMotion检测，阻止同时发出两个动作
            try {
                while (SettingsFrame.isNotMainExit()) {
                    if (System.currentTimeMillis() - mouseWheelTime > 500) {
                        isLockMouseMotion = false;
                    }
                    Thread.sleep(20);
                }
            } catch (Exception e) {
                if (SettingsFrame.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (SettingsFrame.isNotMainExit()) {
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
                    Thread.sleep(20);
                }
            } catch (Exception e) {
                if (SettingsFrame.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            //显示结果线程
            try {
                boolean isLabel1Chosen, isLabel2Chosen, isLabel3Chosen, isLabel4Chosen,
                        isLabel5Chosen, isLabel6Chosen, isLabel7Chosen, isLabel8Chosen;
                while (SettingsFrame.isNotMainExit()) {
                    isLabel1Chosen = false;
                    isLabel2Chosen = false;
                    isLabel3Chosen = false;
                    isLabel4Chosen = false;
                    isLabel5Chosen = false;
                    isLabel6Chosen = false;
                    isLabel7Chosen = false;
                    isLabel8Chosen = false;
                    if (labelCount.get() < listResults.size()) {//有结果可以显示
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
                    }

                    if (!isUserPressed && isLabelNotEmpty(label1)) {
                        if (labelCount.get() == 0) {
                            label1.setBackground(labelColor);
                        } else {
                            label1.setBackground(backgroundColor);
                        }
                        label1.setBorder(border);
                    }
                    if (!isUserPressed && isLabelNotEmpty(label2)) {
                        if (labelCount.get() == 1) {
                            label2.setBackground(labelColor);
                        } else {
                            label2.setBackground(backgroundColor);
                        }
                        label2.setBorder(border);
                    }
                    if (!isUserPressed && isLabelNotEmpty(label3)) {
                        if (labelCount.get() == 2) {
                            label3.setBackground(labelColor);
                        } else {
                            label3.setBackground(backgroundColor);
                        }
                        label3.setBorder(border);
                    }
                    if (!isUserPressed && isLabelNotEmpty(label4)) {
                        if (labelCount.get() == 3) {
                            label4.setBackground(labelColor);
                        } else {
                            label4.setBackground(backgroundColor);
                        }
                        label4.setBorder(border);
                    }
                    if (!isUserPressed && isLabelNotEmpty(label5)) {
                        if (labelCount.get() == 4) {
                            label5.setBackground(labelColor);
                        } else {
                            label5.setBackground(backgroundColor);
                        }
                        label5.setBorder(border);
                    }
                    if (!isUserPressed && isLabelNotEmpty(label6)) {
                        if (labelCount.get() == 5) {
                            label6.setBackground(labelColor);
                        } else {
                            label6.setBackground(backgroundColor);
                        }
                        label6.setBorder(border);
                    }
                    if (!isUserPressed && isLabelNotEmpty(label7)) {
                        if (labelCount.get() == 6) {
                            label7.setBackground(labelColor);
                        } else {
                            label7.setBackground(backgroundColor);
                        }
                        label7.setBorder(border);
                    }
                    if (!isUserPressed && isLabelNotEmpty(label8)) {
                        if (labelCount.get() == 7) {
                            label8.setBackground(labelColor);
                        } else {
                            label8.setBackground(backgroundColor);
                        }
                        label8.setBorder(border);
                    }
                    Thread.sleep(50);
                }
            } catch (InterruptedException ignored) {

            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (SettingsFrame.isNotMainExit()) {
                    if (isUsing) {
                        panel.repaint();
                    }
                    Thread.sleep(250);
                }
            } catch (InterruptedException ignored) {

            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            //添加搜索路径线程
            String command;
            int ascII;
            try {
                while (SettingsFrame.isNotMainExit()) {
                    if (isStartSearchLocal) {
                        isStartSearchLocal = false;
                        ascII = getAscIISum(searchText);
                        int asciiGroup = ascII / 100;

                        switch (asciiGroup) {
                            case 0:
                                for (int i = 0; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 1:
                                for (int i = 1; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 2:
                                for (int i = 2; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 3:
                                for (int i = 3; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 4:
                                for (int i = 4; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 5:
                                for (int i = 5; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 6:
                                for (int i = 6; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 7:
                                for (int i = 7; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 8:
                                for (int i = 8; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 9:
                                for (int i = 9; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 10:
                                for (int i = 10; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 11:
                                for (int i = 11; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 12:
                                for (int i = 12; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 13:
                                for (int i = 13; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 14:
                                for (int i = 14; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 15:
                                for (int i = 15; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 16:
                                for (int i = 16; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 17:
                                for (int i = 17; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 18:
                                for (int i = 18; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 19:
                                for (int i = 19; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 20:
                                for (int i = 20; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 21:
                                for (int i = 21; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 22:
                                for (int i = 22; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 23:
                                for (int i = 23; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 24:
                                for (int i = 24; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;

                            case 25:
                                for (int i = 25; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 26:
                                for (int i = 26; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 27:
                                for (int i = 27; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 28:
                                for (int i = 28; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 29:
                                for (int i = 29; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 30:
                                for (int i = 30; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 31:
                                for (int i = 31; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 32:
                                for (int i = 32; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 33:
                                for (int i = 33; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 34:
                                for (int i = 34; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 35:
                                for (int i = 35; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 36:
                                for (int i = 36; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 37:
                                for (int i = 37; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 38:
                                for (int i = 38; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 39:
                                for (int i = 39; i <= 40; i++) {
                                    command = "list" + i;
                                    commandQueue.add(command);
                                }
                                break;
                            case 40:
                                command = "list40";
                                commandQueue.add(command);
                                break;
                        }
                    }
                    Thread.sleep(10);
                }
            } catch (InterruptedException ignored) {
            }
        });

        for (int i = 0; i < 4; ++i) {
            CachedThreadPool.getInstance().executeTask(() -> {
                try {
                    String column;
                    while (SettingsFrame.isNotMainExit()) {
                        if (runningMode.get() == NORMAL_MODE) {
                            try {
                                while ((column = commandQueue.poll()) != null) {
                                    searchAndAddToTempResults(System.currentTimeMillis(), column);
                                }
                            } catch (SQLException e) {
                                if (SettingsFrame.isDebug()) {
                                    e.printStackTrace();
                                }
                            }
                        }
                        Thread.sleep(10);
                    }
                } catch (InterruptedException ignored) {
                }
            });
        }

        CachedThreadPool.getInstance().executeTask(() -> {
            //后台自动创建数据库索引
            try (Statement stmt = SQLiteUtil.getStatement()) {
                while (SettingsFrame.isNotMainExit()) {
                    if (!isUsing && SearchUtil.getInstance().isUsable()) {
                        for (int i = 0; i <= 40; ++i) {
                            String createIndex = "CREATE UNIQUE INDEX IF NOT EXISTS list" + i + "_index on list" + i + "(PATH);";
                            stmt.execute(createIndex);
                            Thread.sleep(5000);
                        }
                    }
                    Thread.sleep(50);
                }
            } catch (Exception ignored) {

            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            //缓存和常用文件夹搜索线程
            //停顿时间0.5s，每一次输入会更新一次startTime，该线程记录endTime
            try {
                while (SettingsFrame.isNotMainExit()) {
                    long endTime = System.currentTimeMillis();
                    if ((endTime - startTime > 500) && timer) {
                        timer = false; //开始搜索 计时停止
                        resultCount.set(0);
                        labelCount.set(0);
                        currentLabelSelectedPosition.set(0);
                        clearLabel();
                        if (!getTextFieldText().isEmpty()) {
                            label1.setBackground(labelColor);
                        } else {
                            clearLabel();
                        }
                        listResults.clear();
                        tempResults.clear();
                        String text = getTextFieldText();
                        if (search.isUsable()) {
                            if (runningMode.get() == COMMAND_MODE) {
                                if (":update".equals(text)) {
                                    closedTodo();
                                    search.setManualUpdate(true);
                                    timer = false;
                                    continue;
                                }
                                if (":version".equals(text)) {
                                    closedTodo();
                                    JOptionPane.showMessageDialog(null, SettingsFrame.getTranslation("Current Version:") + SettingsFrame.version);
                                }
                                if (":help".equals(text)) {
                                    closedTodo();
                                    Desktop desktop;
                                    if (Desktop.isDesktopSupported()) {
                                        desktop = Desktop.getDesktop();
                                        desktop.browse(new URI("https://github.com/XUANXUQAQ/File-Engine/wiki/Usage"));
                                    }
                                }
                                if (":clearbin".equals(text)) {
                                    closedTodo();
                                    int r = JOptionPane.showConfirmDialog(null, SettingsFrame.getTranslation("Are you sure you want to empty the recycle bin"));
                                    if (r == 0) {
                                        try {
                                            File[] roots = File.listRoots();
                                            for (File root : roots) {
                                                Runtime.getRuntime().exec("cmd.exe /c rd /s /q " + root.getAbsolutePath() + "$Recycle.Bin");
                                            }
                                            JOptionPane.showMessageDialog(null, SettingsFrame.getTranslation("Successfully empty the recycle bin"));
                                        } catch (IOException e) {
                                            JOptionPane.showMessageDialog(null, SettingsFrame.getTranslation("Failed to empty the recycle bin"));
                                        }
                                    }
                                }
                                for (String i : SettingsFrame.getCmdSet()) {
                                    if (i.startsWith(text)) {
                                        resultCount.incrementAndGet();
                                        listResults.add(SettingsFrame.getTranslation("Run command") + i);
                                    }
                                    String[] cmdInfo = semicolon.split(i);
                                    if (cmdInfo[0].equals(text)) {
                                        closedTodo();
                                        openWithAdmin(cmdInfo[1]);
                                    }
                                }
                                showResults(true, false, false, false,
                                        false, false, false, false);
                            } else if (runningMode.get() == NORMAL_MODE) {
                                isStartSearchLocal = true;
                                String[] strings;
                                int length;
                                strings = resultSplit.split(text);
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
                            } else if (runningMode.get() == PLUGIN_MODE) {
                                String result;
                                if (currentUsingPlugin != null) {
                                    while (runningMode.get() == PLUGIN_MODE) {
                                        try {
                                            if ((result = currentUsingPlugin.pollFromResultQueue()) != null) {
                                                if (!listResults.contains(result)) {
                                                    listResults.add(result);
                                                    resultCount.incrementAndGet();
                                                }
                                            }
                                        } catch (NullPointerException ignored) {

                                        }
                                        Thread.sleep(10);
                                    }
                                }
                            }

                            showResults(true, false, false, false,
                                    false, false, false, false);

                        } else {
                            //开启线程等待搜索完成
                            if (search.isManualUpdate()) {
                                if (searchWaiter == null || !searchWaiter.isAlive()) {
                                    searchWaiter = new Thread(() -> {
                                        try {
                                            while (SettingsFrame.isNotMainExit()) {
                                                if (Thread.currentThread().isInterrupted()) {
                                                    break;
                                                }
                                                if (search.isUsable()) {
                                                    startTime = System.currentTimeMillis() - 500;
                                                    timer = true;
                                                    break;
                                                }
                                                Thread.sleep(20);
                                            }
                                        } catch (InterruptedException ignored) {

                                        }
                                    });
                                    searchWaiter.start();
                                }
                            }
                            clearLabel();
                            if (!search.isUsable()) {
                                label1.setBackground(labelColor);
                                label1.setText(SettingsFrame.getTranslation("Updating file index") + "...");
                            }
                        }
                    }
                    Thread.sleep(20);
                }
            } catch (InterruptedException ignored) {

            } catch (URISyntaxException | IOException e) {
                e.printStackTrace();
            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            //检测文件添加线程
            String filesToAdd;
            try (BufferedReader readerAdd = new BufferedReader(new InputStreamReader(
                    new FileInputStream(SettingsFrame.getTmp().getAbsolutePath() + File.separator + "fileAdded.txt"), StandardCharsets.UTF_8))) {
                while (SettingsFrame.isNotMainExit()) {
                    if (!search.isManualUpdate()) {
                        if ((filesToAdd = readerAdd.readLine()) != null) {
                            search.addFileToDatabase(filesToAdd);
                            if (SettingsFrame.isDebug()) {
                                System.out.println("添加" + filesToAdd);
                            }
                        }
                    }
                    Thread.sleep(100);
                }
            } catch (IOException | InterruptedException e) {
                if (SettingsFrame.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });

        CachedThreadPool.getInstance().executeTask(() -> {
            String filesToRemove;
            try (BufferedReader readerRemove = new BufferedReader(new InputStreamReader(
                    new FileInputStream(SettingsFrame.getTmp().getAbsolutePath() + File.separator + "fileRemoved.txt"), StandardCharsets.UTF_8))) {
                while (SettingsFrame.isNotMainExit()) {
                    if (!search.isManualUpdate()) {
                        if ((filesToRemove = readerRemove.readLine()) != null) {
                            search.removeFileFromDatabase(filesToRemove);
                            if (SettingsFrame.isDebug()) {
                                System.out.println("删除" + filesToRemove);
                            }
                        }
                    }
                    Thread.sleep(100);
                }
            } catch (InterruptedException | IOException e) {
                if (SettingsFrame.isDebug() && !(e instanceof InterruptedException)) {
                    e.printStackTrace();
                }
            }
        });


        CachedThreadPool.getInstance().executeTask(() -> {
            // 时间检测线程
            long count = 0;
            long updateTimeLimit = SettingsFrame.getUpdateTimeLimit() * 1000;
            try {
                while (SettingsFrame.isNotMainExit()) {
                    try (Statement stmt = SQLiteUtil.getStatement()) {
                        while (SettingsFrame.isNotMainExit()) {
                            count += 1000;
                            if (count >= updateTimeLimit && !isUsing && !search.isManualUpdate()) {
                                count = 0;
                                if (search.isUsable()) {
                                    search.executeAllCommands(stmt);
                                }
                            }
                            Thread.sleep(1000);
                        }
                    } catch (Exception e) {
                        if (SettingsFrame.isDebug()) {
                            e.printStackTrace();
                        }
                    }
                    Thread.sleep(500);
                }
            } catch (Exception ignored) {

            }
        });


        //搜索本地数据线程
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                while (SettingsFrame.isNotMainExit()) {
                    if (search.isManualUpdate()) {
                        search.setUsable(false);
                        try (Statement stmt = SQLiteUtil.getStatement()) {
                            search.updateLists(SettingsFrame.getIgnorePath(), SettingsFrame.getSearchDepth(), stmt);
                        }
                    }
                    Thread.sleep(10);
                }
            } catch (Exception e) {
                if (SettingsFrame.isDebug() && !(e instanceof InterruptedException)) {
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

    private boolean checkIsMatchedAndAddToList(String path, boolean isPutToTemp) {
        if (check(path)) {
            if (isPutToTemp) {
                if (isExist(path)) {
                    resultCount.incrementAndGet();
                    tempResults.add(path);
                } else {
                    search.removeFileFromDatabase(path);
                }
            } else {
                if (isExist(path)) {
                    if (!listResults.contains(path)) {
                        resultCount.incrementAndGet();
                        listResults.add(path);
                    }
                } else {
                    search.removeFileFromDatabase(path);
                }
            }
        }
        return resultCount.get() >= 100;
    }

    public void releaseAllSqlCache() {
        for (String each : sqlCache.keySet()) {
            try {
                sqlCache.get(each).close();
            } catch (SQLException ignored) {

            }
        }
    }

    private void searchAndAddToTempResults(long time, String column) throws SQLException {
        //为label添加结果
        ResultSet resultSet;
        String each;
        boolean isResultsExcessive = false;
        String pSql = "SELECT PATH FROM " + column + ";";
        PreparedStatement pStmt;
        if ((pStmt = sqlCache.get(pSql)) == null) {
            pStmt = SQLiteUtil.getConnection().prepareStatement(pSql);
            sqlCache.put(pSql, pStmt);
        }

        resultSet = pStmt.executeQuery();
        while (resultSet.next()) {
            each = resultSet.getString("PATH");
            if (search.isUsable()) {
                isResultsExcessive = checkIsMatchedAndAddToList(each, true);
            }
            //用户重新输入了信息
            if (isResultsExcessive || (startTime > time) || !searchBar.isVisible()) {
                break;
            }
        }
        resultSet.close();
    }

    public void showSearchbar() {
        if (!searchBar.isVisible()) {
            searchBar.setVisible(true);
            searchBar.requestFocusInWindow();
            searchBar.setAlwaysOnTop(true);
            textField.setCaretPosition(0);
            textField.requestFocusInWindow();
            isUsing = true;
            startTime = System.currentTimeMillis();
            visibleStartTime = System.currentTimeMillis();
        }
    }

    private void showResultOnLabel(String path, JLabel label, boolean isChosen) {
        String name = getFileName(path);
        ImageIcon icon = GetIconUtil.getBigIcon(path, iconSideLength, iconSideLength);
        label.setIcon(icon);
        label.setBorder(border);
        label.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>" + getParentPath(path) + "</body></html>");
        if (isChosen) {
            label.setBackground(labelColor);
        } else {
            label.setBackground(backgroundColor);
        }
    }

    private void showPluginResultOnLabel(String result, JLabel label, boolean isChosen) {
        currentUsingPlugin.showResultOnLabel(result, label, isChosen);
    }

    private void showCommandOnLabel(String command, JLabel label, boolean isChosen) {
        String[] info = semicolon.split(command);
        String path = info[1];
        String name = info[0];
        ImageIcon imageIcon = GetIconUtil.getBigIcon(path, iconSideLength, iconSideLength);
        label.setIcon(imageIcon);
        label.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>" + path + "</font></body></html>");
        if (isChosen) {
            label.setBackground(labelColor);
        } else {
            label.setBackground(backgroundColor);
        }
        label.setBorder(border);
    }

    private void showResults(boolean isLabel1Chosen, boolean isLabel2Chosen, boolean isLabel3Chosen, boolean isLabel4Chosen,
                             boolean isLabel5Chosen, boolean isLabel6Chosen, boolean isLabel7Chosen, boolean isLabel8Chosen) {
        if (runningMode.get() == NORMAL_MODE) {
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
        } else if (runningMode.get() == COMMAND_MODE) {
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
        } else if (runningMode.get() == PLUGIN_MODE) {
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

    private void clearALabel(JLabel label) {
        label.setBackground(null);
        label.setText(null);
        label.setIcon(null);
        label.setBorder(null);
    }

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
                    JOptionPane.showMessageDialog(null, SettingsFrame.getTranslation("Execute failed"));
                    if (SettingsFrame.isDebug()) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

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
                    createShortCut(path, SettingsFrame.getTmp().getAbsolutePath() + File.separator + "open");
                    Runtime.getRuntime().exec("explorer.exe " + "\"" + SettingsFrame.getTmp().getAbsolutePath() + File.separator + "open.lnk" + "\"");
                }
            } catch (Exception e) {
                //打开上级文件夹
                try {
                    Runtime.getRuntime().exec("explorer.exe /select, \"" + path + "\"");
                } catch (IOException e1) {
                    if (SettingsFrame.isDebug()) {
                        e1.printStackTrace();
                    }
                }
            }
        }
    }

    private void createShortCut(String fileOrFolderPath, String writeShortCutPath) throws Exception {
        File shortcutGen = new File("user/shortcutGenerator.vbs");
        String shortcutGenPath = shortcutGen.getAbsolutePath();
        String start = "cmd.exe /c start " + shortcutGenPath.substring(0, 2);
        String end = "\"" + shortcutGenPath.substring(2) + "\"";
        String commandToGenLnk = start + end + " /target:" + "\"" + fileOrFolderPath + "\"" + " " + "/shortcut:" + "\"" + writeShortCutPath + "\"" + " /workingdir:" + "\"" + fileOrFolderPath.substring(0, fileOrFolderPath.lastIndexOf(File.separator)) + "\"";
        Process p = Runtime.getRuntime().exec("cmd.exe /c " + commandToGenLnk);
        while (p.isAlive()) {
            Thread.sleep(10);
        }
    }

    public String getFileName(String path) {
        if (path != null) {
            int index = path.lastIndexOf(File.separator);
            return path.substring(index + 1);
        }
        return "";
    }

    public int getAscIISum(String path) {
        path = path.toUpperCase();
        if (path.contains(";")) {
            path = path.replace(";", "");
        }
        return GetAscII.INSTANCE.getAscII(path);
    }

    private void saveCache(String content) {
        int cacheNum = 0;
        File cache = new File("user/cache.dat");
        StringBuilder oldCaches = new StringBuilder();
        boolean isRepeated;
        if (cache.exists()) {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(cache), StandardCharsets.UTF_8))) {
                String eachLine;
                while ((eachLine = reader.readLine()) != null) {
                    oldCaches.append(eachLine);
                    cacheNum++;
                }
            } catch (IOException e) {
                if (SettingsFrame.isDebug()) {
                    e.printStackTrace();
                }
            }
        }
        if (cacheNum < SettingsFrame.getCacheNumLimit()) {
            String _temp = oldCaches.append(";").toString();
            isRepeated = _temp.contains(content);
            if (!isRepeated) {
                try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File("user/cache.dat"), true), StandardCharsets.UTF_8))) {
                    out.write(content + "\r\n");
                } catch (Exception ignored) {

                }
            }
        }
    }

    private void delCacheRepeated() {
        File cacheFile = new File("user/cache.dat");
        HashSet<String> set = new HashSet<>();
        StringBuilder allCaches = new StringBuilder();
        String eachLine;
        if (cacheFile.exists()) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(cacheFile), StandardCharsets.UTF_8))) {
                while ((eachLine = br.readLine()) != null) {
                    String[] each = semicolon.split(eachLine);
                    Collections.addAll(set, each);
                }
            } catch (IOException e) {
                if (SettingsFrame.isDebug()) {
                    e.printStackTrace();
                }
            }
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(cacheFile), StandardCharsets.UTF_8))) {
                for (String cache : set) {
                    allCaches.append(cache).append(";\n");
                }
                bw.write(allCaches.toString());
            } catch (IOException e) {
                if (SettingsFrame.isDebug()) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void delCache(ArrayList<String> cache) {
        File cacheFile = new File("user/cache.dat");
        StringBuilder allCaches = new StringBuilder();
        String eachLine;
        if (cacheFile.exists()) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(cacheFile), StandardCharsets.UTF_8))) {
                while ((eachLine = br.readLine()) != null) {
                    String[] each = semicolon.split(eachLine);
                    for (String eachCache : each) {
                        if (!(cache.contains(eachCache))) {
                            allCaches.append(eachCache).append(";\n");
                        }
                    }
                }
            } catch (IOException e) {
                if (SettingsFrame.isDebug()) {
                    e.printStackTrace();
                }
            }
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(cacheFile), StandardCharsets.UTF_8))) {
                bw.write(allCaches.toString());
            } catch (IOException e) {
                if (SettingsFrame.isDebug()) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void searchCache() {
        String cacheResult;
        boolean isCacheRepeated = false;
        ArrayList<String> cachesToDel = new ArrayList<>();
        File cache = new File("user/cache.dat");
        if (cache.exists()) {
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(cache), StandardCharsets.UTF_8))) {
                while ((cacheResult = reader.readLine()) != null) {
                    String[] caches = semicolon.split(cacheResult);
                    for (String eachCache : caches) {
                        if (!(isExist(eachCache))) {
                            cachesToDel.add(eachCache);
                        } else {
                            if (check(eachCache)) {
                                isCacheRepeated = true;
                                if (!listResults.contains(eachCache)) {
                                    resultCount.incrementAndGet();
                                    listResults.add(eachCache);
                                }
                            }
                        }
                    }
                }
            } catch (IOException e) {
                if (SettingsFrame.isDebug()) {
                    e.printStackTrace();
                }
            }
        }
        delCache(cachesToDel);
        if (isCacheRepeated) {
            delCacheRepeated();
        }
    }

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

    private void searchPriorityFolder() {
        File path = new File(SettingsFrame.getPriorityFolder());
        boolean exist = path.exists();
        Queue<String> listRemain = new LinkedList<>();
        if (exist) {
            File[] files = path.listFiles();
            if (!(null == files || files.length == 0)) {
                for (File each : files) {
                    checkIsMatchedAndAddToList(each.getAbsolutePath(), false);
                    if (each.isDirectory()) {
                        listRemain.add(each.getAbsolutePath());
                    }
                }
                while (!listRemain.isEmpty()) {
                    String remain = listRemain.poll();
                    File[] allFiles = new File(remain).listFiles();
                    assert allFiles != null;
                    for (File each : allFiles) {
                        checkIsMatchedAndAddToList(each.getAbsolutePath(), false);
                        if (each.isDirectory()) {
                            listRemain.add(each.getAbsolutePath());
                        }
                    }
                }
            }
        }
    }

    public void setTransparency(float trans) {
        searchBar.setOpacity(trans);
    }

    private void clearTextFieldText() {
        Runnable clear = () -> textField.setText(null);
        SwingUtilities.invokeLater(clear);
    }

    public void closedTodo() {
        if (searchBar.isVisible()) {
            searchBar.setVisible(false);
        }
        clearLabel();
        clearTextFieldText();
        commandQueue.clear();
        startTime = System.currentTimeMillis();//结束搜索
        isUsing = false;
        labelCount.set(0);
        resultCount.set(0);
        currentLabelSelectedPosition.set(0);
        listResults.clear();
        tempResults.clear();
        textField.setText(null);
        isUserPressed = false;
        isLockMouseMotion = false;
        isOpenLastFolderPressed = false;
        isUsing = false;
        isRunAsAdminPressed = false;
        isCopyPathPressed = false;
        timer = false;
        isCacheAndPrioritySearched = false;
        isStartSearchLocal = false;
        try {
            searchWaiter.interrupt();
        } catch (NullPointerException ignored) {

        }
    }

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

    public void setFontColorWithCoverage(int colorNum) {
        fontColorWithCoverage = new Color(colorNum);
    }

    public void setDefaultBackgroundColor(int colorNum) {
        backgroundColor = new Color(colorNum);
    }

    public void setLabelColor(int colorNum) {
        labelColor = new Color(colorNum);
    }

    public void setFontColor(int colorNum) {
        fontColor = new Color(colorNum);
    }

    public void setSearchBarColor(int colorNum) {
        textField.setBackground(new Color(colorNum));
    }

}

