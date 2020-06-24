package frames;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SerializerFeature;
import download.DownloadUpdate;
import hotkeyListener.CheckHotKey;
import moveFiles.moveFiles;
import search.Search;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;


public class SettingsFrame {
    public static final String version = "2.0"; //TODO 更改版本号
    private static volatile boolean mainExit = false;
    public static String name;
    private static int cacheNumLimit;
    private static String hotkey;
    private static int updateTimeLimit;
    private static String ignorePath;
    private static String dataPath;
    private static String priorityFolder;
    private static int searchDepth;
    private static boolean isDefaultAdmin;
    private static boolean isLoseFocusClose;
    private static int openLastFolderKeyCode;
    private static int runAsAdminKeyCode;
    private static int copyPathKeyCode;
    private static float transparency;
    private static int _copyPathKeyCode;
    private static File tmp;
    private static File settings;
    private static HashSet<String> cmdSet;
    private static HashSet<String> languageSet;
    private static ConcurrentHashMap<String, String> translationMap;
    private static ConcurrentHashMap<String, String> fileMap;
    private static Pattern equalSign;
    private static int _openLastFolderKeyCode;
    private static int _runAsAdminKeyCode;
    private static CheckHotKey HotKeyListener;
    private static int labelColor;
    private static int defaultBackgroundColor;
    private static int fontColorWithCoverage;
    private static int fontColor;
    private static int searchBarColor;
    private static String language;
    private static boolean isStartup;
    private static SearchBar searchBar;
    private static TaskBar taskBar;
    private static Search search;
    private Thread updateThread = null;
    private JFrame frame;
    private JTextField textFieldUpdateTime;
    private JTextField textFieldCacheNum;
    private JTextArea textAreaIgnorePath;
    private JTextField textFieldSearchDepth;
    private JCheckBox checkBox1;
    private JLabel label3;
    private JLabel label1;
    private JLabel label2;
    private JLabel label4;
    private JLabel label5;
    private JLabel label6;
    private JPanel panel;
    private JLabel label7;
    private JButton buttonSaveAndRemoveDesktop;
    private JButton Button3;
    private JScrollPane scrollpane;
    private JLabel labelplaceholder2;
    private JTextField textFieldHotkey;
    private JLabel labeltip3;
    private JTextField textFieldPriorityFolder;
    private JButton ButtonPriorityFolder;
    private JTabbedPane tabbedPane1;
    private JCheckBox checkBoxAdmin;
    private JLabel labelplaceholder1;
    private JCheckBox checkBoxLoseFocus;
    private JLabel labelPlaceHolder2;
    private JLabel labelPlaceHolder3;
    private JLabel labelRunAsAdmin;
    private JTextField textFieldRunAsAdmin;
    private JLabel labelOpenFolder;
    private JTextField textFieldOpenLastFolder;
    private JButton buttonAddCMD;
    private JButton buttonDelCmd;
    private JScrollPane scrollPaneCmd;
    private JList<Object> listCmds;
    private JLabel labelPlaceHoder4;
    private JButton buttonSave;
    private JLabel labelPlaceHoder6;
    private JLabel labelPlaceHolder7;
    private JLabel labelPlaceHolder8;
    private JLabel labelAbout;
    private JLabel labelAboutGithub;
    private JLabel labelPlaceHolder12;
    private JLabel labelPlaceHolder13;
    private JLabel labelGitHubTip;
    private JLabel labelIcon;
    private JLabel labelPlaceHolder15;
    private JLabel labelGithubIssue;
    private JLabel labelPalceHolder5;
    private JLabel labelPlaceHolder16;
    private JLabel labelPlaceHolder17;
    private JButton buttonCheckUpdate;
    private JLabel labelCopyPath;
    private JTextField textFieldCopyPath;
    private JLabel labelVersion;
    private JPanel tab1;
    private JPanel tab2;
    private JPanel tab3;
    private JPanel tab6;
    private JPanel tab7;
    private JPanel tab9;
    private JTextField textFieldTransparency;
    private JLabel labelTransparency;
    private JLabel labelPlaceHolder5;
    private JPanel tab8;
    private JLabel labelColorTip;
    private JTextField textFieldBackgroundDefault;
    private JTextField textFieldFontColorWithCoverage;
    private JTextField textFieldLabelColor;
    private JLabel labelLabelColor;
    private JLabel labelFontColor;
    private JLabel label_default_Color;
    private JLabel labelSharp1;
    private JLabel labelSharp2;
    private JLabel labelSharp4;
    private JButton buttonResetColor;
    private JTextField textFieldFontColor;
    private JLabel labelSharp5;
    private JLabel labelNotChosenFontColor;
    private JLabel labelColorChooser;
    private JLabel FontColorWithCoverageChooser;
    private JLabel defaultBackgroundChooser;
    private JLabel FontColorChooser;
    private JLabel labelSearchBarColor;
    private JLabel labelSharp8;
    private JTextField textFieldSearchBarColor;
    private JLabel searchBarColorChooser;
    private JLabel labelCmdTip2;
    private JLabel labelDesciption;
    private JLabel labelBeautyEye;
    private JLabel labelFastJson;
    private JLabel labelJna;
    private JPanel tab5;
    private JList<Object> listLanguage;
    private JLabel labelLanguage;
    private JLabel labelPlaceHolderL;
    private JLabel labelLanTip2;
    private JLabel labelPlaceHolder0;
    private JLabel labelPlaceHolder1;
    private JLabel labelPlaceHolder;
    private JLabel labelPlaceHolder14;
    private JLabel labelPlaceHolderWhatever;
    private JLabel labelPlaceHolderWhatever2;


    private static class SettingsFrameBuilder {
        private static SettingsFrame instance = new SettingsFrame();
    }

    public static SettingsFrame getInstance() {
        return SettingsFrameBuilder.instance;
    }

    public static boolean isNotMainExit() {
        return !mainExit;
    }

    public static int getCacheNumLimit() {
        return cacheNumLimit;
    }

    public static int getUpdateTimeLimit() {
        return updateTimeLimit;
    }

    public static String getIgnorePath() {
        return ignorePath;
    }

    public static String getPriorityFolder() {
        return priorityFolder;
    }

    public static int getSearchDepth() {
        return searchDepth;
    }

    public static boolean isDefaultAdmin() {
        return isDefaultAdmin;
    }

    public static boolean isLoseFocusClose() {
        return isLoseFocusClose;
    }

    public static int getOpenLastFolderKeyCode() {
        return openLastFolderKeyCode;
    }

    public static int getRunAsAdminKeyCode() {
        return runAsAdminKeyCode;
    }

    public static int getCopyPathKeyCode() {
        return copyPathKeyCode;
    }

    public static float getTransparency() {
        return transparency;
    }

    public static File getTmp() {
        return tmp;
    }

    public static HashSet<String> getCmdSet() {
        return cmdSet;
    }

    public static int getLabelColor() {
        return labelColor;
    }

    public static int getDefaultBackgroundColor() {
        return defaultBackgroundColor;
    }

    public static int getFontColorWithCoverage() {
        return fontColorWithCoverage;
    }

    public static int getFontColor() {
        return fontColor;
    }

    private SettingsFrame() {
        frame = new JFrame("设置");
        tmp = new File("tmp");
        settings = new File("user/settings.json");
        cmdSet = new HashSet<>();
        equalSign = Pattern.compile("=");
        fileMap = new ConcurrentHashMap<>();
        translationMap = new ConcurrentHashMap<>();
        languageSet = new HashSet<>();
        HotKeyListener = CheckHotKey.getInstance();
        searchBar = SearchBar.getInstance();
        taskBar = TaskBar.getInstance();
        search = Search.getInstance();

        setAllSettings();
        initLanguageFileMap();
        translate(language);

        labelAboutGithub.setText("<html><a href='https://github.com/XUANXUQAQ/File-Engine'><font size=\"4\">File-Engine</font></a></html>");
        labelBeautyEye.setText("1.WebLaF");
        labelFastJson.setText("2.FastJson");
        labelJna.setText("3.Java-Native-Access");
        ImageIcon imageIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        labelIcon.setIcon(imageIcon);

        checkBox1.addActionListener(e -> setStartup(checkBox1.isSelected()));

        buttonSaveAndRemoveDesktop.addActionListener(e -> {
            String currentFolder = new File("").getAbsolutePath();
            if (currentFolder.equals(FileSystemView.getFileSystemView().getHomeDirectory().getAbsolutePath()) || currentFolder.equals("C:\\Users\\Public\\Desktop")) {
                JOptionPane.showMessageDialog(frame, SettingsFrame.getTranslation("The program is detected on the desktop and cannot be moved"));
                return;
            }
            int isConfirmed = JOptionPane.showConfirmDialog(frame, SettingsFrame.getTranslation("Whether to remove and backup all files on the desktop," +
                    "they will be in the program's Files folder, which may take a few minutes"));
            if (isConfirmed == 0) {
                Thread fileMover = new Thread(new moveDesktopFiles());
                fileMover.start();
            }
        });

        Button3.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), SettingsFrame.getTranslation("Choose"));
            File file = fileChooser.getSelectedFile();
            if (file != null && returnValue == JFileChooser.APPROVE_OPTION) {
                textAreaIgnorePath.append(file.getAbsolutePath() + ",\n");
            }
        });

        textFieldHotkey.addKeyListener(new KeyListener() {
            boolean reset = true;

            @Override
            public void keyTyped(KeyEvent e) {

            }

            @Override
            public void keyPressed(KeyEvent e) {
                int key = e.getKeyCode();
                if (reset) {
                    textFieldHotkey.setText(null);
                    reset = false;
                }
                textFieldHotkey.setCaretPosition(textFieldHotkey.getText().length());
                if (key == 17) {
                    if (!textFieldHotkey.getText().contains("Ctrl + ")) {
                        textFieldHotkey.setText(textFieldHotkey.getText() + "Ctrl + ");
                    }
                } else if (key == 18) {
                    if (!textFieldHotkey.getText().contains("Alt + ")) {
                        textFieldHotkey.setText(textFieldHotkey.getText() + "Alt + ");
                    }
                } else if (key == 524) {
                    if (!textFieldHotkey.getText().contains("Win + ")) {
                        textFieldHotkey.setText(textFieldHotkey.getText() + "Win + ");
                    }
                } else if (key == 16) {
                    if (!textFieldHotkey.getText().contains("Shift + ")) {
                        textFieldHotkey.setText(textFieldHotkey.getText() + "Shift + ");
                    }
                } else if (64 < key && key < 91) {
                    String txt = textFieldHotkey.getText();
                    if (!txt.isEmpty()) {
                        char c1 = Character.toUpperCase(txt.charAt(txt.length() - 1));
                        if (64 < c1 && c1 < 91) {
                            String text = txt.substring(0, txt.length() - 1);
                            textFieldHotkey.setText(text + (char) key);
                        } else {
                            textFieldHotkey.setText(txt + (char) key);
                        }
                    }
                    if (txt.length() == 1) {
                        textFieldHotkey.setText(null);
                    }
                }

            }

            @Override
            public void keyReleased(KeyEvent e) {
                reset = true;
            }
        });
        ButtonPriorityFolder.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), SettingsFrame.getTranslation("Choose"));
            File file = fileChooser.getSelectedFile();
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                textFieldPriorityFolder.setText(file.getAbsolutePath());
            }

        });
        textFieldPriorityFolder.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2) {
                    textFieldPriorityFolder.setText(null);
                }

            }
        });

        { //设置窗口显示
            labelVersion.setText(getTranslation("Current Version:") + version);
            if (isStartup) {
                checkBox1.setSelected(true);
            } else {
                checkBox1.setSelected(false);
            }
            textFieldUpdateTime.setText(String.valueOf(updateTimeLimit));
            textAreaIgnorePath.setText(ignorePath.replaceAll(",", ",\n"));
            textFieldCacheNum.setText(String.valueOf(cacheNumLimit));
            textFieldSearchDepth.setText(String.valueOf(searchDepth));
            textFieldHotkey.setText(hotkey);
            textFieldPriorityFolder.setText(priorityFolder);
            checkBoxAdmin.setSelected(isDefaultAdmin);
            textFieldSearchBarColor.setText(Integer.toHexString(searchBarColor));
            Color _searchBarColor = new Color(searchBarColor);
            searchBarColorChooser.setBackground(_searchBarColor);
            searchBarColorChooser.setForeground(_searchBarColor);
            textFieldBackgroundDefault.setText(Integer.toHexString(defaultBackgroundColor));
            Color _defaultBackgroundColor = new Color(defaultBackgroundColor);
            defaultBackgroundChooser.setBackground(_defaultBackgroundColor);
            defaultBackgroundChooser.setForeground(_defaultBackgroundColor);
            textFieldLabelColor.setText(Integer.toHexString(labelColor));
            Color _labelColor = new Color(labelColor);
            labelColorChooser.setBackground(_labelColor);
            labelColorChooser.setForeground(_labelColor);
            textFieldFontColorWithCoverage.setText(Integer.toHexString(fontColorWithCoverage));
            Color _fontColorWithCoverage = new Color(fontColorWithCoverage);
            FontColorWithCoverageChooser.setBackground(_fontColorWithCoverage);
            FontColorWithCoverageChooser.setForeground(_fontColorWithCoverage);
            checkBoxLoseFocus.setSelected(isLoseFocusClose);
            textFieldTransparency.setText(String.valueOf(transparency));
            textFieldFontColor.setText(Integer.toHexString(fontColor));
            Color _fontColor = new Color(fontColor);
            FontColorChooser.setBackground(_fontColor);
            FontColorChooser.setForeground(_fontColor);
            if (runAsAdminKeyCode == 17) {
                textFieldRunAsAdmin.setText("Ctrl + Enter");
            } else if (runAsAdminKeyCode == 16) {
                textFieldRunAsAdmin.setText("Shift + Enter");
            } else if (runAsAdminKeyCode == 18) {
                textFieldRunAsAdmin.setText("Alt + Enter");
            }
            if (openLastFolderKeyCode == 17) {
                textFieldOpenLastFolder.setText("Ctrl + Enter");
            } else if (openLastFolderKeyCode == 16) {
                textFieldOpenLastFolder.setText("Shift + Enter");
            } else if (openLastFolderKeyCode == 18) {
                textFieldOpenLastFolder.setText("Alt + Enter");
            }
            if (copyPathKeyCode == 17) {
                textFieldCopyPath.setText("Ctrl + Enter");
            } else if (copyPathKeyCode == 16) {
                textFieldCopyPath.setText("Shift + Enter");
            } else if (copyPathKeyCode == 18) {
                textFieldCopyPath.setText("Alt + Enter");
            }
            listCmds.setListData(cmdSet.toArray());
            listLanguage.setListData(languageSet.toArray());
            listLanguage.setSelectedValue(language, true);
        }


        checkBoxAdmin.addActionListener(e -> isDefaultAdmin = checkBoxAdmin.isSelected());
        checkBoxLoseFocus.addActionListener(e -> isLoseFocusClose = checkBoxLoseFocus.isSelected());
        textFieldRunAsAdmin.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                int code = e.getKeyCode();
                if (code == 17) {
                    textFieldRunAsAdmin.setText("Ctrl + Enter");
                    _runAsAdminKeyCode = 17;
                } else if (code == 16) {
                    textFieldRunAsAdmin.setText("Shift + Enter");
                    _runAsAdminKeyCode = 16;
                } else if (code == 18) {
                    textFieldRunAsAdmin.setText("Alt + Enter");
                    _runAsAdminKeyCode = 18;
                }
            }
        });
        textFieldOpenLastFolder.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                int code = e.getKeyCode();
                if (code == 17) {
                    textFieldOpenLastFolder.setText("Ctrl + Enter");
                    _openLastFolderKeyCode = 17;
                } else if (code == 16) {
                    textFieldOpenLastFolder.setText("Shift + Enter");
                    _openLastFolderKeyCode = 16;
                } else if (code == 18) {
                    textFieldOpenLastFolder.setText("Alt + Enter");
                    _openLastFolderKeyCode = 18;
                }
            }
        });
        buttonAddCMD.addActionListener(e -> {
            String name = JOptionPane.showInputDialog(SettingsFrame.getTranslation("Please enter the ID of the command, then you can enter \": identifier\" in the search box to execute the command directly"));
            if (name == null || name.isEmpty()) {
                //未输入
                return;
            }
            if (name.equals("update") || name.equals("clearbin") || name.equals("help") || name.equals("version") || isRepeatCommand(name)) {
                JOptionPane.showMessageDialog(frame, SettingsFrame.getTranslation("Conflict with existing commands"));
                return;
            }
            String cmd;
            JOptionPane.showMessageDialog(frame, SettingsFrame.getTranslation("Please select the location of the executable file (a folder is also acceptable)"));
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
            int returnValue = fileChooser.showDialog(new Label(), SettingsFrame.getTranslation("Choose"));
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                cmd = fileChooser.getSelectedFile().getAbsolutePath();
                cmdSet.add(":" + name + ";" + cmd);
                listCmds.setListData(cmdSet.toArray());
            }

        });
        buttonDelCmd.addActionListener(e -> {
            String del = (String) listCmds.getSelectedValue();
            cmdSet.remove(del);
            listCmds.setListData(cmdSet.toArray());

        });
        buttonSave.addActionListener(e -> {
            saveChanges();
            hideFrame();
        });
        labelAboutGithub.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Desktop desktop;
                if (Desktop.isDesktopSupported()) {
                    desktop = Desktop.getDesktop();
                    try {
                        URI uri = new URI("https://github.com/XUANXUQAQ/File-Engine");
                        desktop.browse(uri);
                    } catch (Exception ignored) {

                    }
                }
            }
        });
        buttonCheckUpdate.addActionListener(e -> {
            try {
                if (!updateThread.isAlive()) {
                    updateThread = new Thread(this::update);
                    updateThread.start();
                }
            } catch (NullPointerException e1) {
                updateThread = new Thread(this::update);
                updateThread.start();
            }
        });
        textFieldCopyPath.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                int code = e.getKeyCode();
                if (code == 17) {
                    textFieldCopyPath.setText("Ctrl + Enter");
                    _copyPathKeyCode = 17;
                } else if (code == 16) {
                    textFieldCopyPath.setText("Shift + Enter");
                    _copyPathKeyCode = 16;
                } else if (code == 18) {
                    textFieldCopyPath.setText("Alt + Enter");
                    _copyPathKeyCode = 18;
                }
            }
        });
        buttonResetColor.addActionListener(e -> {
            textFieldFontColorWithCoverage.setText(Integer.toHexString(0x1C0EFF));
            textFieldSearchBarColor.setText(Integer.toHexString(0xffffff));
            textFieldLabelColor.setText(Integer.toHexString(0xFF9868));
            textFieldBackgroundDefault.setText(Integer.toHexString(0xffffff));
            textFieldFontColor.setText(Integer.toHexString(0x333333));
        });


        labelColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, SettingsFrame.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                int r = color.getRed();
                int g = color.getGreen();
                int b = color.getBlue();
                StringBuilder rgb = new StringBuilder();
                if (r == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(r));
                }
                if (g == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(g));
                }
                if (b == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(b));
                }
                textFieldLabelColor.setText(rgb.toString());
                labelColorChooser.setBackground(color);
                labelColorChooser.setForeground(color);
            }
        });
        FontColorWithCoverageChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, SettingsFrame.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                int r = color.getRed();
                int g = color.getGreen();
                int b = color.getBlue();
                StringBuilder rgb = new StringBuilder();
                if (r == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(r));
                }
                if (g == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(g));
                }
                if (b == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(b));
                }
                textFieldFontColorWithCoverage.setText(rgb.toString());
                FontColorWithCoverageChooser.setBackground(color);
                FontColorWithCoverageChooser.setForeground(color);
            }
        });
        defaultBackgroundChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, SettingsFrame.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                int r = color.getRed();
                int g = color.getGreen();
                int b = color.getBlue();
                StringBuilder rgb = new StringBuilder();
                if (r == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(r));
                }
                if (g == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(g));
                }
                if (b == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(b));
                }
                textFieldBackgroundDefault.setText(rgb.toString());
                defaultBackgroundChooser.setBackground(color);
                defaultBackgroundChooser.setForeground(color);
            }
        });
        FontColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, SettingsFrame.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                int r = color.getRed();
                int g = color.getGreen();
                int b = color.getBlue();
                StringBuilder rgb = new StringBuilder();
                if (r == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(r));
                }
                if (g == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(g));
                }
                if (b == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(b));
                }
                textFieldFontColor.setText(rgb.toString());
                FontColorChooser.setBackground(color);
                FontColorChooser.setForeground(color);
            }
        });

        searchBarColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, SettingsFrame.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                int r = color.getRed();
                int g = color.getGreen();
                int b = color.getBlue();
                StringBuilder rgb = new StringBuilder();
                if (r == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(r));
                }
                if (g == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(g));
                }
                if (b == 0) {
                    rgb.append("00");
                } else {
                    rgb.append(Integer.toHexString(b));
                }
                textFieldSearchBarColor.setText(rgb.toString());
                searchBarColorChooser.setBackground(color);
                searchBarColorChooser.setForeground(color);
            }
        });
    }

    private void translate(String language) {
        initTranslations(language);
        tabbedPane1.setTitleAt(0, getTranslation("General"));
        tabbedPane1.setTitleAt(1, getTranslation("Search settings"));
        tabbedPane1.setTitleAt(2, getTranslation("Search bar settings"));
        tabbedPane1.setTitleAt(3, getTranslation("Language settings"));
        tabbedPane1.setTitleAt(4, getTranslation("Hotkey settings"));
        tabbedPane1.setTitleAt(5, getTranslation("My commands"));
        tabbedPane1.setTitleAt(6, getTranslation("Color settings"));
        tabbedPane1.setTitleAt(7, getTranslation("About"));
        checkBox1.setText(getTranslation("Add to startup"));
        buttonSaveAndRemoveDesktop.setText(getTranslation("Backup and remove all desktop files"));
        label4.setText(getTranslation("Set the maximum number of caches:"));
        ButtonPriorityFolder.setText(getTranslation("Choose"));
        Button3.setText(getTranslation("Choose"));
        label1.setText(getTranslation("File update detection interval:"));
        label5.setText(getTranslation("Seconds"));
        label6.setText(getTranslation("Search depth (too large may affect performance):"));
        labeltip3.setText(getTranslation("Priority search folder location (double-click to clear):"));
        label7.setText(getTranslation("Separate different paths with commas, and ignore C:\\Windows by default"));
        label3.setText(getTranslation("Set ignore folder:"));
        checkBoxLoseFocus.setText(getTranslation("Close search bar when focus lost"));
        checkBoxAdmin.setText(getTranslation("Open other programs as an administrator (provided that the software has privileges)"));
        labelTransparency.setText(getTranslation("Search bar transparency:"));
        label2.setText(getTranslation("Open search bar:"));
        labelRunAsAdmin.setText(getTranslation("Run as administrator:"));
        labelOpenFolder.setText(getTranslation("Open the parent folder:"));
        labelCopyPath.setText(getTranslation("Copy path:"));
        labelCmdTip2.setText(getTranslation("You can add custom commands here. After adding, you can enter \": + your set identifier\" in the search box to quickly open"));
        buttonAddCMD.setText(getTranslation("Add"));
        buttonDelCmd.setText(getTranslation("Delete"));
        labelColorTip.setText(getTranslation("Please enter the hexadecimal value of RGB color"));
        labelSearchBarColor.setText(getTranslation("Search bar Color:"));
        labelLabelColor.setText(getTranslation("Chosen label color:"));
        labelFontColor.setText(getTranslation("Chosen label font Color:"));
        label_default_Color.setText(getTranslation("Default background Color:"));
        labelNotChosenFontColor.setText(getTranslation("Unchosen label Color:"));
        buttonResetColor.setText(getTranslation("Reset to default"));
        labelGitHubTip.setText(getTranslation("This is an open source software,GitHub:"));
        labelGithubIssue.setText(getTranslation("If you find a bug or have some suggestions, welcome to GitHub for feedback"));
        buttonCheckUpdate.setText(getTranslation("Check for update"));
        labelDesciption.setText(getTranslation("Thanks for the following project"));
        buttonSave.setText(getTranslation("Save"));
        labelLanTip2.setText(getTranslation("The translation might not be 100% accurate"));
        labelLanguage.setText(getTranslation("Choose a language"));
        labelVersion.setText(getTranslation(getTranslation("Current Version:") + version));
    }

    private static void initLanguageSet() {
        languageSet.add("简体中文");
        languageSet.add("English(US)");
    }

    private static void initTranslations(String language) {
        if (!language.equals("English(US)")) {
            String filePath = fileMap.get(language);
            try (InputStream inputStream = SettingsFrame.class.getResourceAsStream(filePath); BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] record = equalSign.split(line);
                    translationMap.put(record[0].trim(), record[1].trim());
                }
            } catch (IOException ignored) {

            }
        } else {
            translationMap.put("#frame_width", String.valueOf(1100));
            translationMap.put("#frame_height", String.valueOf(600));
        }
    }

    private static void initLanguageFileMap() {
        fileMap.put("简体中文", "/language/Chinese(Simplified).txt");
    }

    public static String getTranslation(String text) {
        String translated;
        if (language.equals("English(US)")) {
            translated = text;
        } else {
            translated = translationMap.get(text);
        }
        return translated;
    }

    public static void setMainExit(boolean b) {
        mainExit = b;
    }

    public static boolean isAdmin() {
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
                } else if (nextLine.equals("echo %errorlevel%")) {
                    printedErrorlevel = true;
                }
            }
        } catch (IOException e) {
            return false;
        }
    }

    private boolean isRepeatCommand(String name) {
        name = ":" + name;
        for (String each : cmdSet) {
            if (each.substring(0, each.indexOf(";")).equals(name)) {
                return true;
            }
        }
        return false;
    }

    public static JSONObject getInfo() throws IOException {
        StringBuilder jsonUpdate = new StringBuilder();
        URL updateServer = new URL("https://gitee.com/xuanxuF/File-Engine/raw/master/version.json");
        URLConnection uc = updateServer.openConnection();
        uc.setConnectTimeout(3 * 1000);
        //防止屏蔽程序抓取而返回403错误
        uc.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.116 Safari/537.36 Edg/80.0.361.57");
        try (BufferedReader br = new BufferedReader(new InputStreamReader(uc.getInputStream(), StandardCharsets.UTF_8))) {
            String eachLine;
            while ((eachLine = br.readLine()) != null) {
                jsonUpdate.append(eachLine);
            }
        }
        return JSONObject.parseObject(jsonUpdate.toString());
    }

    private static void readAllSettings() {
        try (BufferedReader buffR = new BufferedReader(new InputStreamReader(new FileInputStream(settings), StandardCharsets.UTF_8))) {
            String line;
            StringBuilder result = new StringBuilder();
            while (null != (line = buffR.readLine())) {
                result.append(line);
            }
            JSONObject settingsInJson = JSON.parseObject(result.toString());
            if (settingsInJson.containsKey("isStartup")) {
                isStartup = settingsInJson.getBoolean("isStartup");
            } else {
                isStartup = false;
            }
            if (settingsInJson.containsKey("cacheNumLimit")) {
                cacheNumLimit = settingsInJson.getInteger("cacheNumLimit");
            } else {
                cacheNumLimit = 1000;
            }
            if (settingsInJson.containsKey("hotkey")) {
                hotkey = settingsInJson.getString("hotkey");
            } else {
                hotkey = "Ctrl + Alt + J";
            }
            if (settingsInJson.containsKey("dataPath")) {
                dataPath = settingsInJson.getString("dataPath");
            } else {
                dataPath = new File("data.db").getAbsolutePath();
            }
            if (settingsInJson.containsKey("priorityFolder")) {
                priorityFolder = settingsInJson.getString("priorityFolder");
            } else {
                priorityFolder = "";
            }
            if (settingsInJson.containsKey("searchDepth")) {
                searchDepth = settingsInJson.getInteger("searchDepth");
            } else {
                searchDepth = 6;
            }
            if (settingsInJson.containsKey("ignorePath")) {
                ignorePath = settingsInJson.getString("ignorePath");
            } else {
                ignorePath = "C:\\Windows,";
            }
            if (settingsInJson.containsKey("updateTimeLimit")) {
                updateTimeLimit = settingsInJson.getInteger("updateTimeLimit");
            } else {
                updateTimeLimit = 5;
            }
            if (settingsInJson.containsKey("isDefaultAdmin")) {
                isDefaultAdmin = settingsInJson.getBoolean("isDefaultAdmin");
            } else {
                isDefaultAdmin = false;
            }
            if (settingsInJson.containsKey("isLoseFocusClose")) {
                isLoseFocusClose = settingsInJson.getBoolean("isLoseFocusClose");
            } else {
                isLoseFocusClose = true;
            }
            if (settingsInJson.containsKey("openLastFolderKeyCode")) {
                openLastFolderKeyCode = settingsInJson.getInteger("openLastFolderKeyCode");
            } else {
                openLastFolderKeyCode = 17;
            }
            _openLastFolderKeyCode = openLastFolderKeyCode;
            if (settingsInJson.containsKey("runAsAdminKeyCode")) {
                runAsAdminKeyCode = settingsInJson.getInteger("runAsAdminKeyCode");
            } else {
                runAsAdminKeyCode = 16;
            }
            _runAsAdminKeyCode = runAsAdminKeyCode;
            if (settingsInJson.containsKey("copyPathKeyCode")) {
                copyPathKeyCode = settingsInJson.getInteger("copyPathKeyCode");
            } else {
                copyPathKeyCode = 18;
            }
            _copyPathKeyCode = copyPathKeyCode;
            if (settingsInJson.containsKey("transparency")) {
                transparency = settingsInJson.getFloat("transparency");
            } else {
                transparency = 0.8f;
            }
            if (settingsInJson.containsKey("searchBarColor")) {
                searchBarColor = settingsInJson.getInteger("searchBarColor");
            } else {
                searchBarColor = 0xfffff;
            }
            if (settingsInJson.containsKey("defaultBackground")) {
                defaultBackgroundColor = settingsInJson.getInteger("defaultBackground");
            } else {
                defaultBackgroundColor = 0xffffff;
            }
            if (settingsInJson.containsKey("fontColorWithCoverage")) {
                fontColorWithCoverage = settingsInJson.getInteger("fontColorWithCoverage");
            } else {
                fontColorWithCoverage = 0x1C0EFF;
            }
            if (settingsInJson.containsKey("labelColor")) {
                labelColor = settingsInJson.getInteger("labelColor");
            } else {
                labelColor = 0xFF9868;
            }
            if (settingsInJson.containsKey("fontColor")) {
                fontColor = settingsInJson.getInteger("fontColor");
            } else {
                fontColor = 0xC5C5C5;
            }
            if (settingsInJson.containsKey("language")) {
                language = settingsInJson.getString("language");
            } else {
                language = "English(US)";
            }
        } catch (NullPointerException | IOException ignored) {

        }
    }

    private static void setAllSettings() {
        readAllSettings();
        HotKeyListener.registerHotkey(hotkey);
        File data = new File(dataPath);
        if (!data.exists()) {
            taskBar.showMessage(getTranslation("Info"), getTranslation("Detected that the cache does not exist and is searching again"));
            data = new File("data.db");
            dataPath = data.getAbsolutePath();
            search.setManualUpdate(true);
        }

        searchBar.setTransparency(transparency);
        searchBar.setDefaultBackgroundColor(defaultBackgroundColor);
        searchBar.setLabelColor(labelColor);
        searchBar.setFontColorWithCoverage(fontColorWithCoverage);
        searchBar.setFontColor(fontColor);
        searchBar.setSearchBarColor(searchBarColor);

        initLanguageSet();
        initCmdSetSettings();

        //保存设置
        JSONObject allSettings = new JSONObject();
        allSettings.put("hotkey", hotkey);
        allSettings.put("isStartup", isStartup);
        allSettings.put("cacheNumLimit", cacheNumLimit);
        allSettings.put("updateTimeLimit", updateTimeLimit);
        allSettings.put("ignorePath", ignorePath);
        allSettings.put("searchDepth", searchDepth);
        allSettings.put("priorityFolder", priorityFolder);
        allSettings.put("dataPath", dataPath);
        allSettings.put("isDefaultAdmin", isDefaultAdmin);
        allSettings.put("isLoseFocusClose", isLoseFocusClose);
        allSettings.put("runAsAdminKeyCode", runAsAdminKeyCode);
        allSettings.put("openLastFolderKeyCode", openLastFolderKeyCode);
        allSettings.put("copyPathKeyCode", copyPathKeyCode);
        allSettings.put("transparency", transparency);
        allSettings.put("labelColor", labelColor);
        allSettings.put("defaultBackground", defaultBackgroundColor);
        allSettings.put("searchBarColor", searchBarColor);
        allSettings.put("fontColorWithCoverage", fontColorWithCoverage);
        allSettings.put("fontColor", fontColor);
        allSettings.put("language", language);
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(settings), StandardCharsets.UTF_8))) {
            String format = JSON.toJSONString(allSettings, SerializerFeature.PrettyFormat, SerializerFeature.WriteMapNullValue, SerializerFeature.WriteDateUseDateFormat);
            buffW.write(format);
        } catch (IOException ignored) {

        }
    }

    private static void initCmdSetSettings() {
        //获取所有自定义命令
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("user/cmds.txt"), StandardCharsets.UTF_8))) {
            String each;
            while ((each = br.readLine()) != null) {
                cmdSet.add(each);
            }
        } catch (IOException ignored) {

        }
    }

    private void update() {
        JSONObject updateInfo;
        String latestVersion;
        try {
            updateInfo = getInfo();
            latestVersion = updateInfo.getString("version");
        } catch (IOException e1) {
            JOptionPane.showMessageDialog(frame, getTranslation("Check update failed"));
            return;
        }
        File test = new File("TEST");
        if (Double.parseDouble(latestVersion) > Double.parseDouble(version) || test.exists()) {
            String description = updateInfo.getString("description");
            int result = JOptionPane.showConfirmDialog(frame,
                    getTranslation("New Version available") + latestVersion + "," + getTranslation("Whether to update") + "\n" + getTranslation("update content") + "\n" + description);
            if (result == 0) {
                //开始更新,下载更新文件到tmp
                String urlChoose;
                String fileName;
                if (name.contains("x64")) {
                    urlChoose = "url64";
                    fileName = "File-Engine-x64.exe";
                } else {
                    urlChoose = "url86";
                    fileName = "File-Engine-x86.exe";
                }
                DownloadUpdate download = DownloadUpdate.getInstance();
                try {
                    download.downLoadFromUrl(updateInfo.getString(urlChoose), fileName, tmp.getAbsolutePath());
                } catch (Exception e) {
                    if (!e.getMessage().equals("User Interrupted")) {
                        JOptionPane.showMessageDialog(frame, getTranslation("Download failed"));
                    }
                    download.hideFrame();
                    return;
                }
                TaskBar.getInstance().showMessage(getTranslation("Info"), getTranslation("The download is complete and the update will start at the next boot"));
                try {
                    File updateSignal = new File("user/update");
                    updateSignal.createNewFile();
                } catch (Exception ignored) {

                }
            }
        } else {
            JOptionPane.showMessageDialog(frame, getTranslation("Latest version:") + latestVersion + "\n" + getTranslation("The current version is the latest"));
        }
    }

    public void hideFrame() {
        frame.setVisible(false);
    }


    public boolean isSettingsVisible() {
        return frame.isVisible();
    }

    public void showWindow() {
        frame.setContentPane(SettingsFrameBuilder.instance.panel);
        URL frameIcon = SettingsFrame.class.getResource("/icons/frame.png");
        frame.setIconImage(new ImageIcon(frameIcon).getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.setResizable(true);
        frame.setSize(Integer.parseInt(translationMap.get("#frame_width")), Integer.parseInt(translationMap.get("#frame_height")));
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }


    private void saveChanges() {
        JSONObject allSettings = new JSONObject();
        int updateTimeLimitTemp;
        int cacheNumLimitTemp;
        String ignorePathTemp;
        int searchDepthTemp;
        float transparencyTemp;

        try {
            updateTimeLimitTemp = Integer.parseInt(textFieldUpdateTime.getText());
        } catch (Exception e1) {
            updateTimeLimitTemp = -1; // 输入不正确
        }
        if (updateTimeLimitTemp > 3600 || updateTimeLimitTemp <= 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("The file index update setting is wrong, please change"));
            return;
        }
        isStartup = checkBox1.isSelected();
        try {
            cacheNumLimitTemp = Integer.parseInt(textFieldCacheNum.getText());
        } catch (Exception e1) {
            cacheNumLimitTemp = -1;
        }
        if (cacheNumLimitTemp > 10000 || cacheNumLimitTemp <= 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("The cache capacity is set incorrectly, please change"));
            return;
        }
        ignorePathTemp = textAreaIgnorePath.getText();
        ignorePathTemp = ignorePathTemp.replaceAll("\n", "");
        if (!ignorePathTemp.toLowerCase().contains("c:\\windows")) {
            ignorePathTemp = ignorePathTemp + "C:\\Windows,";
        }
        try {
            searchDepthTemp = Integer.parseInt(textFieldSearchDepth.getText());
        } catch (Exception e1) {
            searchDepthTemp = -1;
        }

        if (searchDepthTemp > 10 || searchDepthTemp <= 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("Search depth setting is wrong, please change"));
            return;
        }

        String _hotkey = textFieldHotkey.getText();
        if (_hotkey.length() == 1) {
            JOptionPane.showMessageDialog(frame, getTranslation("Hotkey setting is wrong, please change"));
            return;
        } else {
            if (!CheckHotKey.getInstance().isHotkeyAvailable(_hotkey)) {
                JOptionPane.showMessageDialog(frame, getTranslation("Hotkey setting is wrong, please change"));
                return;
            }
        }
        try {
            transparencyTemp = Float.parseFloat(textFieldTransparency.getText());
        } catch (Exception e) {
            transparencyTemp = -1f;
        }
        if (transparencyTemp > 1 || transparencyTemp <= 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("Transparency setting error"));
            return;
        }

        if (_openLastFolderKeyCode == _runAsAdminKeyCode || _openLastFolderKeyCode == _copyPathKeyCode || _runAsAdminKeyCode == _copyPathKeyCode) {
            JOptionPane.showMessageDialog(frame, getTranslation("HotKey conflict"));
            return;
        } else {
            openLastFolderKeyCode = _openLastFolderKeyCode;
            runAsAdminKeyCode = _runAsAdminKeyCode;
            copyPathKeyCode = _copyPathKeyCode;
        }

        int _labelColor;
        try {
            _labelColor = Integer.parseInt(textFieldLabelColor.getText(), 16);
        } catch (Exception e) {
            _labelColor = -1;
        }
        if (_labelColor < 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("Chosen label color is set incorrectly"));
            return;
        }


        int _fontColorWithCoverage;
        try {
            _fontColorWithCoverage = Integer.parseInt(textFieldFontColorWithCoverage.getText(), 16);
        } catch (Exception e) {
            _fontColorWithCoverage = -1;
        }
        if (_fontColorWithCoverage < 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("Chosen label font color is set incorrectly"));
            return;
        }


        int _defaultBackgroundColor;
        try {
            _defaultBackgroundColor = Integer.parseInt(textFieldBackgroundDefault.getText(), 16);
        } catch (Exception e) {
            _defaultBackgroundColor = -1;
        }
        if (_defaultBackgroundColor < 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("Incorrect defaultBackground color setting"));
            return;
        }


        int _fontColor;
        try {
            _fontColor = Integer.parseInt(textFieldFontColor.getText(), 16);
        } catch (Exception e) {
            _fontColor = -1;
        }
        if (_fontColor < 0) {
            JOptionPane.showMessageDialog(frame, "Unchosen label font color is set incorrectly");
            return;
        }


        int _searchBarColor;
        try {
            _searchBarColor = Integer.parseInt(textFieldSearchBarColor.getText(), 16);
        } catch (Exception e) {
            _searchBarColor = -1;
        }
        if (_searchBarColor < 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("The color of the search bar is set incorrectly"));
            return;
        }
        if (!listLanguage.getSelectedValue().equals(language)) {
            language = (String) listLanguage.getSelectedValue();
            translate(language);
        }

        priorityFolder = textFieldPriorityFolder.getText();
        hotkey = textFieldHotkey.getText();
        HotKeyListener.changeHotKey(hotkey);
        cacheNumLimit = cacheNumLimitTemp;
        updateTimeLimit = updateTimeLimitTemp;
        ignorePath = ignorePathTemp;
        searchDepth = searchDepthTemp;
        transparency = transparencyTemp;
        labelColor = _labelColor;
        defaultBackgroundColor = _defaultBackgroundColor;
        searchBarColor = _searchBarColor;
        fontColorWithCoverage = _fontColorWithCoverage;
        fontColor = _fontColor;
        SearchBar instance = SearchBar.getInstance();
        instance.setTransparency(transparency);
        instance.setDefaultBackgroundColor(defaultBackgroundColor);
        instance.setLabelColor(labelColor);
        instance.setFontColorWithCoverage(fontColorWithCoverage);
        instance.setFontColor(fontColor);
        instance.setSearchBarColor(searchBarColor);
        Color _tmp = new Color(labelColor);
        labelColorChooser.setBackground(_tmp);
        labelColorChooser.setForeground(_tmp);
        _tmp = new Color(defaultBackgroundColor);
        defaultBackgroundChooser.setBackground(_tmp);
        defaultBackgroundChooser.setForeground(_tmp);
        _tmp = new Color(fontColorWithCoverage);
        FontColorWithCoverageChooser.setBackground(_tmp);
        FontColorWithCoverageChooser.setForeground(_tmp);
        _tmp = new Color(fontColor);
        FontColorChooser.setBackground(_tmp);
        FontColorChooser.setForeground(_tmp);


        //保存设置
        allSettings.put("hotkey", hotkey);
        allSettings.put("isStartup", isStartup);
        allSettings.put("cacheNumLimit", cacheNumLimit);
        allSettings.put("updateTimeLimit", updateTimeLimit);
        allSettings.put("ignorePath", ignorePath);
        allSettings.put("searchDepth", searchDepth);
        allSettings.put("priorityFolder", priorityFolder);
        allSettings.put("dataPath", dataPath);
        allSettings.put("isDefaultAdmin", isDefaultAdmin);
        allSettings.put("isLoseFocusClose", isLoseFocusClose);
        allSettings.put("runAsAdminKeyCode", runAsAdminKeyCode);
        allSettings.put("openLastFolderKeyCode", openLastFolderKeyCode);
        allSettings.put("copyPathKeyCode", copyPathKeyCode);
        allSettings.put("transparency", transparency);
        allSettings.put("labelColor", labelColor);
        allSettings.put("defaultBackground", defaultBackgroundColor);
        allSettings.put("searchBarColor", searchBarColor);
        allSettings.put("fontColorWithCoverage", fontColorWithCoverage);
        allSettings.put("fontColor", fontColor);
        allSettings.put("language", language);
        try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
            String format = JSON.toJSONString(allSettings, SerializerFeature.PrettyFormat, SerializerFeature.WriteMapNullValue, SerializerFeature.WriteDateUseDateFormat);
            buffW.write(format);
        } catch (IOException ignored) {

        }
        //保存自定义命令
        StringBuilder strb = new StringBuilder();
        for (String each : cmdSet) {
            strb.append(each);
            strb.append("\n");
        }
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/cmds.txt"), StandardCharsets.UTF_8))) {
            bw.write(strb.toString());
        } catch (IOException ignored) {

        }
    }

    public static boolean isDebug() {
        try {
            String res = System.getProperty("File_Engine_Debug");
            return res.equalsIgnoreCase("true");
        } catch (NullPointerException e) {
            return false;
        }
    }

    private void setStartup(boolean b) {
        if (b) {
            String command = "cmd.exe /c schtasks /create /ru \"administrators\" /rl HIGHEST /sc ONLOGON /tn \"File-Engine\" /tr ";
            File FileEngine = new File(name);
            String absolutePath = "\"\"" + FileEngine.getAbsolutePath() + "\"\" /f";
            command += absolutePath;
            Process p;
            try {
                p = Runtime.getRuntime().exec(command);
                p.waitFor();
                BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                String line;
                StringBuilder result = new StringBuilder();
                while ((line = outPut.readLine()) != null) {
                    result.append(line);
                }
                outPut.close();
                if (!result.toString().isEmpty()) {
                    checkBox1.setSelected(false);
                    JOptionPane.showMessageDialog(frame, getTranslation("Add to startup failed, please try to run as administrator"));
                }
            } catch (IOException | InterruptedException ignored) {

            }
        } else {
            String command = "cmd.exe /c schtasks /delete /tn \"File-Engine\" /f";
            Process p;
            try {
                p = Runtime.getRuntime().exec(command);
                p.waitFor();
                BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                String line;
                StringBuilder result = new StringBuilder();
                while ((line = outPut.readLine()) != null) {
                    result.append(line);
                }
                outPut.close();
                if (!result.toString().isEmpty()) {
                    checkBox1.setSelected(true);
                    JOptionPane.showMessageDialog(frame, getTranslation("Delete startup failed, please try to run as administrator"));
                }
            } catch (IOException | InterruptedException ignored) {

            }
        }

        boolean isSelected = checkBox1.isSelected();
        JSONObject allSettings = null;
        try (BufferedReader buffR1 = new BufferedReader(new InputStreamReader(new FileInputStream(settings), StandardCharsets.UTF_8))) {
            String line;
            StringBuilder result = new StringBuilder();
            while (null != (line = buffR1.readLine())) {
                result.append(line);
            }
            allSettings = JSON.parseObject(result.toString());
            allSettings.put("isStartup", isSelected);
        } catch (IOException ignored) {

        }
        //保存设置
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(settings), StandardCharsets.UTF_8))) {
            String format = JSON.toJSONString(allSettings, SerializerFeature.PrettyFormat, SerializerFeature.WriteMapNullValue, SerializerFeature.WriteDateUseDateFormat);
            buffW.write(format);
        } catch (IOException ignored) {

        }
    }
}

class moveDesktopFiles implements Runnable {
    @Override
    public void run() {
        boolean desktop1;
        boolean desktop2;
        File fileDesktop = FileSystemView.getFileSystemView().getHomeDirectory();
        File fileBackUp = new File("Files");
        if (!fileBackUp.exists()) {
            fileBackUp.mkdir();
        }
        ArrayList<String> preserveFiles = new ArrayList<>();
        preserveFiles.add(fileDesktop.getAbsolutePath());
        preserveFiles.add("C:\\Users\\Public\\Desktop");
        moveFiles moveFiles = new moveFiles(preserveFiles);
        desktop1 = moveFiles.moveFolder(fileDesktop.getAbsolutePath(), fileBackUp.getAbsolutePath());
        desktop2 = moveFiles.moveFolder("C:\\Users\\Public\\Desktop", fileBackUp.getAbsolutePath());
        if (desktop1 || desktop2) {
            JOptionPane.showMessageDialog(null, SettingsFrame.getTranslation("Files with the same name are detected, please move them by yourself"));
        }
    }
}