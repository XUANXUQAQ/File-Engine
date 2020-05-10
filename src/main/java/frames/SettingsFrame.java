package frames;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import download.DownloadUpdate;
import hotkeyListener.CheckHotKey;
import main.MainClass;
import moveFiles.moveFiles;
import search.Search;
import unzipFile.Unzip;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class SettingsFrame {
    public static int cacheNumLimit;
    public static String hotkey;
    public static int updateTimeLimit;
    public static String ignorePath;
    public static String dataPath;
    public static String priorityFolder;
    public static int searchDepth;
    public static boolean isDefaultAdmin;
    public static boolean isLoseFocusClose;
    public static int openLastFolderKeyCode;
    public static int runAsAdminKeyCode;
    public static int copyPathKeyCode;
    public static float transparency;
    private static int _copyPathKeyCode;
    public static File tmp = new File("tmp");
    public static File settings = new File("user/settings.json");
    public static HashSet<String> cmdSet = new HashSet<>();
    private static int _openLastFolderKeyCode;
    private static int _runAsAdminKeyCode;
    private static CheckHotKey HotKeyListener;
    public static int labelColor;
    public static int backgroundColor1;
    public static int backgroundColor2;
    public static int backgroundColor3;
    public static int backgroundColor4;
    public static int fontColorWithCoverage;
    public static int fontColor;
    public static int searchBarColor;
    public static int maxConnectionNum;
    public static int minConnectionNum;
    public static int connectionTimeLimit;
    public static int diskCount;
    private JTextField textFieldUpdateTime;
    private JTextField textFieldCacheNum;
    private JTextArea textAreaIgnorePath;
    private JTextField textFieldSearchDepth;
    private JCheckBox checkBox1;
    private JFrame frame = new JFrame("设置");
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
    private JButton ButtonCloseAdmin;
    private JButton ButtonRecoverAdmin;
    private JLabel labeltip2;
    private JLabel labeltip3;
    private JTextField textFieldDataPath;
    private JTextField textFieldPriorityFolder;
    private JButton ButtonDataPath;
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
    private JPanel tab5;
    private JPanel tab6;
    private JPanel tab8;
    private JTextField textFieldTransparency;
    private JLabel labelTransparency;
    private JLabel labelPlaceHolder5;
    private JPanel tab7;
    private JLabel labelColorTip;
    private JTextField textFieldBackground1;
    private JTextField textFieldBackground2;
    private JTextField textFieldFontColorWithCoverage;
    private JTextField textFieldLabelColor;
    private JLabel labelLabelColor;
    private JLabel labelFontColor;
    private JLabel label_2_Color;
    private JLabel label_1_Color;
    private JLabel labelSharp1;
    private JLabel labelSharp2;
    private JLabel labelSharp3;
    private JLabel labelSharp4;
    private JButton buttonResetColor;
    private JTextField textFieldFontColor;
    private JLabel labelSharp5;
    private JLabel labelFontColors;
    private JLabel labelColorChooser;
    private JLabel FontColorWithCoverageChooser;
    private JLabel background2Chooser;
    private JLabel background1Chooser;
    private JLabel FontColorChooser;
    private JLabel label_3_Color;
    private JLabel labelSharp6;
    private JTextField textFieldBackground3;
    private JLabel label_4_Color;
    private JLabel labelSharp7;
    private JTextField textFieldBackground4;
    private JLabel background3Chooser;
    private JLabel background4Chooser;
    private JLabel labelSearchBarColor;
    private JLabel labelSharp8;
    private JTextField textFieldSearchBarColor;
    private JLabel searchBarColorChooser;
    private JPanel tab4;
    private JLabel labelFileConnectionTip;
    private JLabel labelMaxConnection;
    private JLabel labelMinConnection;
    private JTextField textFieldMaxConnectionNum;
    private JTextField textFieldMinConnectionNum;
    private JLabel labelFileConnectionTip3;
    private JTextField textFieldConnectionTimeLimit;
    private JLabel labelPlaceHolderF1;
    private JLabel labelPlaceHolderF2;
    private JLabel labelTip4F;
    private JLabel labelPlaceHolder;
    private JLabel labelFileConnectionTip2;
    private JLabel labelCmdTip2;
    private JLabel labelCurrentConnection;
    private JLabel currentConnection;
    private JButton buttonClearConnection;
    private JLabel labelDesciption;
    private JLabel labelBeautyEye;
    private JLabel labelFastJson;
    private JLabel labelJna;
    private static boolean isStartup;
    private Unzip unzipInstance = Unzip.getInstance();
    private Thread updateThread = null;


    public SettingsFrame() {
        labelAboutGithub.setText("<html><a href='https://github.com/XUANXUQAQ/File-Engine'><font size=\"4\">File-Engine</font></a></html>");
        labelBeautyEye.setText("1.beautyeye");
        labelFastJson.setText("2.fastjson");
        labelJna.setText("3.jna");
        ImageIcon imageIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        labelIcon.setIcon(imageIcon);

        checkBox1.addActionListener(e -> setStartup(checkBox1.isSelected()));


        buttonSaveAndRemoveDesktop.addActionListener(e -> {
            String currentFolder = new File("").getAbsolutePath();
            if (currentFolder.equals(FileSystemView.getFileSystemView().getHomeDirectory().getAbsolutePath()) || currentFolder.equals("C:\\Users\\Public\\Desktop")) {
                JOptionPane.showMessageDialog(null, "检测到该程序在桌面，无法移动");
                return;
            }
            int isConfirmed = JOptionPane.showConfirmDialog(null, "是否移除并备份桌面上的所有文件\n它们会在该程序的Files文件夹中\n这可能需要几分钟时间");
            if (isConfirmed == 0) {
                Thread fileMover = new Thread(new moveDesktopFiles());
                fileMover.start();
            }
        });

        Button3.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), "选择");
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
        ButtonCloseAdmin.addActionListener(e -> {
            if (MainClass.isAdmin()) {
                JOptionPane.showMessageDialog(null, "请将弹出窗口的最后一栏改为当前文件夹，\n修改成功后请重新设置开机启动，\n下次启动时请使用带(elevated)的快捷方式。");
                InputStream UAC = getClass().getResourceAsStream("/UACTrustShortCut.zip");
                File target = new File(tmp.getAbsolutePath() + "/UACTrustShortCut.zip");
                if (!target.exists()) {
                    copyFile(UAC, target);
                }
                File mainUAC = new File(tmp.getAbsolutePath() + "/UACTrustShortCut/ElevatedShortcut.exe");
                if (!mainUAC.exists()) {
                    unzipInstance.unZipFiles(target, tmp.getAbsolutePath());
                }
                File mainFile = new File(MainClass.name);
                try {
                    Runtime.getRuntime().exec("\"" + mainUAC.getAbsolutePath() + "\"" + " \"" + mainFile.getAbsolutePath() + "\"");
                } catch (IOException ignored) {

                }
            } else {
                JOptionPane.showMessageDialog(null, "请以管理员身份运行");
            }
        });
        ButtonRecoverAdmin.addActionListener(e -> {
            if (MainClass.isAdmin()) {
                JOptionPane.showMessageDialog(null, "点击最后一个Remove shortcut即可选择删除快捷方式，\n修改成功后请重新设置开机启动。");
                InputStream UAC = getClass().getResourceAsStream("/UACTrustShortCut.zip");
                File target = new File(tmp.getAbsolutePath() + "/UACTrustShortCut.zip");
                if (!target.exists()) {
                    copyFile(UAC, target);
                }
                File mainUAC = new File(tmp.getAbsolutePath() + "/UACTrustShortCut/ElevatedShortcut.exe");
                if (!mainUAC.exists()) {
                    unzipInstance.unZipFiles(target, tmp.getAbsolutePath());
                }
                try {
                    Runtime.getRuntime().exec("\"" + mainUAC.getAbsolutePath() + "\"");
                } catch (IOException ignored) {

                }
            } else {
                JOptionPane.showMessageDialog(null, "请以管理员身份运行");
            }
        });
        ButtonDataPath.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), "选择");
            File file = fileChooser.getSelectedFile();
            if (file != null && returnValue == JFileChooser.APPROVE_OPTION) {
                textFieldDataPath.setText(file.getAbsolutePath());
            }

        });
        ButtonPriorityFolder.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), "选择");
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
        labelVersion.setText("当前版本：" + MainClass.version);
        try (BufferedReader buffR = new BufferedReader(new FileReader(settings))) {
            String line;
            StringBuilder result = new StringBuilder();
            while (null != (line = buffR.readLine())) {
                result.append(line);
            }
            JSONObject settings = JSON.parseObject(result.toString());
            isStartup = settings.getBoolean("isStartup");
            if (isStartup) {
                checkBox1.setSelected(true);
            } else {
                checkBox1.setSelected(false);
            }
            updateTimeLimit = settings.getInteger("updateTimeLimit");
            textFieldUpdateTime.setText(String.valueOf(updateTimeLimit));
            ignorePath = settings.getString("ignorePath");
            textAreaIgnorePath.setText(ignorePath.replaceAll(",", ",\n"));
            cacheNumLimit = settings.getInteger("cacheNumLimit");
            textFieldCacheNum.setText(String.valueOf(cacheNumLimit));
            searchDepth = settings.getInteger("searchDepth");
            textFieldSearchDepth.setText(String.valueOf(searchDepth));
            textFieldHotkey.setText(hotkey);
            textFieldDataPath.setText(dataPath);
            textFieldPriorityFolder.setText(priorityFolder);
            isDefaultAdmin = settings.getBoolean("isDefaultAdmin");
            checkBoxAdmin.setSelected(isDefaultAdmin);
            isLoseFocusClose = settings.getBoolean("isLoseFocusClose");
            textFieldMaxConnectionNum.setText(String.valueOf(maxConnectionNum));
            textFieldMinConnectionNum.setText(String.valueOf(minConnectionNum));
            textFieldConnectionTimeLimit.setText(String.valueOf(connectionTimeLimit / 1000));

            textFieldSearchBarColor.setText(Integer.toHexString(searchBarColor));
            Color _searchBarColor = new Color(searchBarColor);
            searchBarColorChooser.setBackground(_searchBarColor);
            searchBarColorChooser.setForeground(_searchBarColor);

            textFieldBackground1.setText(Integer.toHexString(backgroundColor1));
            Color _backgroundColor1 = new Color(backgroundColor1);
            background1Chooser.setBackground(_backgroundColor1);
            background1Chooser.setForeground(_backgroundColor1);

            textFieldBackground2.setText(Integer.toHexString(backgroundColor2));
            Color _backgroundColor2 = new Color(backgroundColor2);
            background2Chooser.setBackground(_backgroundColor2);
            background2Chooser.setForeground(_backgroundColor2);

            textFieldBackground3.setText(Integer.toHexString(backgroundColor3));
            Color _backgroundColor3 = new Color(backgroundColor3);
            background3Chooser.setBackground(_backgroundColor3);
            background3Chooser.setForeground(_backgroundColor3);

            textFieldBackground4.setText(Integer.toHexString(backgroundColor4));
            Color _backgroundColor4 = new Color(backgroundColor4);
            background4Chooser.setBackground(_backgroundColor4);
            background4Chooser.setForeground(_backgroundColor4);

            textFieldLabelColor.setText(Integer.toHexString(labelColor));
            Color _labelColor = new Color(labelColor);
            labelColorChooser.setBackground(_labelColor);
            labelColorChooser.setForeground(_labelColor);

            textFieldFontColorWithCoverage.setText(Integer.toHexString(fontColorWithCoverage));
            Color _fontColorWithCoverage = new Color(fontColorWithCoverage);
            FontColorWithCoverageChooser.setBackground(_fontColorWithCoverage);
            FontColorWithCoverageChooser.setForeground(_fontColorWithCoverage);

            checkBoxLoseFocus.setSelected(isLoseFocusClose);
            runAsAdminKeyCode = settings.getInteger("runAsAdminKeyCode");
            openLastFolderKeyCode = settings.getInteger("openLastFolderKeyCode");
            transparency = settings.getFloat("transparency");
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
        } catch (IOException ignored) {

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
            String name = JOptionPane.showInputDialog("请输入对该命令的标识,之后你可以在搜索框中输入 \": 标识符\" 直接执行该命令");
            if (name == null || name.isEmpty()) {
                //未输入
                return;
            }
            if (name.equals("update") || name.equals("clearbin") || name.equals("help") || name.equals("version") || isRepeatCommand(name)) {
                JOptionPane.showMessageDialog(null, "和已有的命令冲突");
                return;
            }
            String cmd;
            JOptionPane.showMessageDialog(null, "请选择可执行文件位置(文件夹也可以)");
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
            int returnValue = fileChooser.showDialog(new Label(), "选择");
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
        buttonSave.addActionListener(e -> saveChanges());
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
            textFieldBackground1.setText(Integer.toHexString(0xffffff));
            textFieldBackground2.setText(Integer.toHexString(0xffffff));
            textFieldBackground3.setText(Integer.toHexString(0xffffff));
            textFieldBackground4.setText(Integer.toHexString(0xffffff));
            textFieldFontColor.setText(Integer.toHexString(0x333333));
        });


        labelColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, "选择颜色", null);
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
                Color color = JColorChooser.showDialog(null, "选择颜色", null);
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
        background2Chooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, "选择颜色", null);
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
                textFieldBackground2.setText(rgb.toString());
                background2Chooser.setBackground(color);
                background2Chooser.setForeground(color);
            }
        });
        background1Chooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, "选择颜色", null);
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
                textFieldBackground1.setText(rgb.toString());
                background1Chooser.setBackground(color);
                background1Chooser.setForeground(color);
            }
        });
        FontColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, "选择颜色", null);
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

        background3Chooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, "选择颜色", null);
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
                textFieldBackground3.setText(rgb.toString());
                background3Chooser.setBackground(color);
                background3Chooser.setForeground(color);
            }
        });
        background4Chooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, "选择颜色", null);
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
                textFieldBackground4.setText(rgb.toString());
                background4Chooser.setBackground(color);
                background4Chooser.setForeground(color);
            }
        });
        searchBarColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, "选择颜色", null);
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

        ExecutorService fixedThreadPool = Executors.newFixedThreadPool(1);
        fixedThreadPool.execute(() -> {
            try {
                while (!MainClass.mainExit) {
                    Thread.sleep(50);
                    currentConnection.setText(String.valueOf(SearchBar.getInstance().currentConnectionNum()));
                }
            } catch (InterruptedException ignored) {

            }
        });

        buttonClearConnection.addActionListener(e -> SearchBar.getInstance().closeAllConnection());
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
        try (BufferedReader br = new BufferedReader(new InputStreamReader(uc.getInputStream()))) {
            String eachLine;
            while ((eachLine = br.readLine()) != null) {
                jsonUpdate.append(eachLine);
            }
        }
        return JSONObject.parseObject(jsonUpdate.toString());
    }

    public static void initSettings() {
        //获取所有设置信息
        FileSystemView sys = FileSystemView.getFileSystemView();
        for (File root : File.listRoots()) {
            String driveType = sys.getSystemTypeDescription(root);
            if (driveType.equals("本地磁盘")) {
                diskCount++;
            }
        }
        try (BufferedReader buffR = new BufferedReader(new FileReader(settings))) {
            String line;
            StringBuilder result = new StringBuilder();
            while (null != (line = buffR.readLine())) {
                result.append(line);
            }
            JSONObject settings = JSON.parseObject(result.toString());
            if (settings.containsKey("isStartup")) {
                isStartup = settings.getBoolean("isStartup");
            } else {
                isStartup = false;
            }
            if (settings.containsKey("cacheNumLimit")) {
                cacheNumLimit = settings.getInteger("cacheNumLimit");
            } else {
                cacheNumLimit = 1000;
            }
            if (settings.containsKey("hotkey")) {
                hotkey = settings.getString("hotkey");
            } else {
                hotkey = "Ctrl + Alt + J";
            }
            HotKeyListener = CheckHotKey.getInstance();
            HotKeyListener.registerHotkey(hotkey);
            if (settings.containsKey("dataPath")) {
                dataPath = settings.getString("dataPath");
                File data = new File(dataPath);
                if (!data.exists()) {
                    MainClass.showMessage("提示", "检测到缓存不存在，正在重新搜索");
                    data = new File("data");
                    dataPath = data.getAbsolutePath();
                    Search.getInstance().setManualUpdate(true);
                }
            } else {
                dataPath = new File("data").getAbsolutePath();
            }
            if (settings.containsKey("priorityFolder")) {
                priorityFolder = settings.getString("priorityFolder");
            } else {
                priorityFolder = "";
            }
            if (settings.containsKey("searchDepth")) {
                searchDepth = settings.getInteger("searchDepth");
            } else {
                searchDepth = 6;
            }
            if (settings.containsKey("ignorePath")) {
                ignorePath = settings.getString("ignorePath");
            } else {
                ignorePath = "C:\\Windows,";
            }
            if (settings.containsKey("updateTimeLimit")) {
                updateTimeLimit = settings.getInteger("updateTimeLimit");
            } else {
                updateTimeLimit = 5;
            }
            if (settings.containsKey("isDefaultAdmin")) {
                isDefaultAdmin = settings.getBoolean("isDefaultAdmin");
            } else {
                isDefaultAdmin = false;
            }
            if (settings.containsKey("isLoseFocusClose")) {
                isLoseFocusClose = settings.getBoolean("isLoseFocusClose");
            } else {
                isLoseFocusClose = true;
            }
            if (settings.containsKey("openLastFolderKeyCode")) {
                openLastFolderKeyCode = settings.getInteger("openLastFolderKeyCode");
            } else {
                openLastFolderKeyCode = 17;
            }
            _openLastFolderKeyCode = openLastFolderKeyCode;
            if (settings.containsKey("runAsAdminKeyCode")) {
                runAsAdminKeyCode = settings.getInteger("runAsAdminKeyCode");
            } else {
                runAsAdminKeyCode = 16;
            }
            _runAsAdminKeyCode = runAsAdminKeyCode;
            if (settings.containsKey("copyPathKeyCode")) {
                copyPathKeyCode = settings.getInteger("copyPathKeyCode");
            } else {
                copyPathKeyCode = 18;
            }
            _copyPathKeyCode = copyPathKeyCode;
            if (settings.containsKey("transparency")) {
                transparency = settings.getFloat("transparency");
            } else {
                transparency = 0.8f;
            }
            if (settings.containsKey("searchBarColor")) {
                searchBarColor = settings.getInteger("searchBarColor");
            } else {
                searchBarColor = 0xfffff;
            }
            if (settings.containsKey("backgroundColor1")) {
                backgroundColor1 = settings.getInteger("backgroundColor1");
            } else {
                backgroundColor1 = 0xffffff;
            }
            if (settings.containsKey("backgroundColor2")) {
                backgroundColor2 = settings.getInteger("backgroundColor2");
            } else {
                backgroundColor2 = 0xffffff;
            }
            if (settings.containsKey("backgroundColor3")) {
                backgroundColor3 = settings.getInteger("backgroundColor3");
            } else {
                backgroundColor3 = 0xffffff;
            }
            if (settings.containsKey("backgroundColor4")) {
                backgroundColor4 = settings.getInteger("backgroundColor4");
            } else {
                backgroundColor4 = 0xffffff;
            }
            if (settings.containsKey("fontColorWithCoverage")) {
                fontColorWithCoverage = settings.getInteger("fontColorWithCoverage");
            } else {
                fontColorWithCoverage = 0x1C0EFF;
            }
            if (settings.containsKey("labelColor")) {
                labelColor = settings.getInteger("labelColor");
            } else {
                labelColor = 0xFF9868;
            }
            if (settings.containsKey("fontColor")) {
                fontColor = settings.getInteger("fontColor");
            } else {
                fontColor = 0xC5C5C5;
            }
            if (settings.containsKey("maxConnectionNum")) {
                maxConnectionNum = settings.getInteger("maxConnectionNum");
            } else {
                maxConnectionNum = 15;
            }
            if (settings.containsKey("minConnectionNum")) {
                minConnectionNum = settings.getInteger("minConnectionNum");
            } else {
                minConnectionNum = 10;
            }
            if (settings.containsKey("connectionTimeLimit")) {
                connectionTimeLimit = settings.getInteger("connectionTimeLimit");
            } else {
                connectionTimeLimit = 600000;
            }
        } catch (NullPointerException | IOException ignored) {

        }

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
        allSettings.put("backgroundColor1", backgroundColor1);
        allSettings.put("backgroundColor2", backgroundColor2);
        allSettings.put("backgroundColor3", backgroundColor3);
        allSettings.put("backgroundColor4", backgroundColor4);
        allSettings.put("searchBarColor", searchBarColor);
        allSettings.put("fontColorWithCoverage", fontColorWithCoverage);
        allSettings.put("fontColor", fontColor);
        allSettings.put("maxConnectionNum", maxConnectionNum);
        allSettings.put("minConnectionNum", minConnectionNum);
        allSettings.put("connectionTimeLimit", connectionTimeLimit);
        try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
            buffW.write(allSettings.toJSONString());
        } catch (IOException ignored) {

        }

        //获取所有自定义命令
        try (BufferedReader br = new BufferedReader(new FileReader(new File("user/cmds.txt")))) {
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
            JOptionPane.showMessageDialog(null, "检查更新失败");
            return;
        }
        File test = new File("TEST");
        if (Double.parseDouble(latestVersion) > Double.parseDouble(MainClass.version) || test.exists()) {
            String description = updateInfo.getString("description");
            int result = JOptionPane.showConfirmDialog(null, "有新版本" + latestVersion + "，是否更新\n更新内容：\n" + description);
            if (result == 0) {
                //开始更新,下载更新文件到tmp
                String urlChoose;
                String fileName;
                if (MainClass.name.contains("x64")) {
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
                    if (!e.getMessage().equals("用户中断下载")) {
                        JOptionPane.showMessageDialog(null, "下载失败");
                    }
                    download.hideFrame();
                    return;
                }
                MainClass.showMessage("提示", "下载完成，更新将在下次启动时开始");
                try {
                    File updateSignal = new File("user/update");
                    updateSignal.createNewFile();
                } catch (Exception ignored) {

                }
            }
        } else {
            JOptionPane.showMessageDialog(null, "最新版本：" + latestVersion + "\n当前版本已是最新");
        }
    }

    public void hideFrame() {
        frame.setVisible(false);
    }


    public boolean isSettingsVisible() {
        return frame.isVisible();
    }

    public void showWindow() {
        frame.setContentPane(new SettingsFrame().panel);
        URL frameIcon = SettingsFrame.class.getResource("/icons/frame.png");
        frame.setIconImage(new ImageIcon(frameIcon).getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.setSize(942, 600);
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
            JOptionPane.showMessageDialog(null, "文件索引更新设置错误，请更改");
            return;
        }
        isStartup = checkBox1.isSelected();
        try {
            cacheNumLimitTemp = Integer.parseInt(textFieldCacheNum.getText());
        } catch (Exception e1) {
            cacheNumLimitTemp = -1;
        }
        if (cacheNumLimitTemp > 10000 || cacheNumLimitTemp <= 0) {
            JOptionPane.showMessageDialog(null, "缓存容量设置错误，请更改");
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
            JOptionPane.showMessageDialog(null, "搜索深度设置错误，请更改");
            return;
        }

        String _hotkey = textFieldHotkey.getText();
        if (_hotkey.length() == 1) {
            JOptionPane.showMessageDialog(null, "快捷键设置错误");
            return;
        } else {
            if (!(64 < _hotkey.charAt(_hotkey.length() - 1) && _hotkey.charAt(_hotkey.length() - 1) < 91)) {
                JOptionPane.showMessageDialog(null, "快捷键设置错误");
                return;
            }
        }
        try {
            transparencyTemp = Float.parseFloat(textFieldTransparency.getText());
        } catch (Exception e) {
            transparencyTemp = -1f;
        }
        if (transparencyTemp > 1 || transparencyTemp <= 0) {
            JOptionPane.showMessageDialog(null, "透明度设置错误");
            return;
        }

        if (_openLastFolderKeyCode == _runAsAdminKeyCode || _openLastFolderKeyCode == _copyPathKeyCode || _runAsAdminKeyCode == _copyPathKeyCode) {
            JOptionPane.showMessageDialog(null, "快捷键冲突");
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
            JOptionPane.showMessageDialog(null, "选中框颜色设置错误");
            return;
        }
        int _fontColorWithCoverage;
        try {
            _fontColorWithCoverage = Integer.parseInt(textFieldFontColorWithCoverage.getText(), 16);
        } catch (Exception e) {
            _fontColorWithCoverage = -1;
        }
        if (_fontColorWithCoverage < 0) {
            JOptionPane.showMessageDialog(null, "选中框字体颜色设置错误");
            return;
        }
        int _backgroundColor1;
        try {
            _backgroundColor1 = Integer.parseInt(textFieldBackground1.getText(), 16);
        } catch (Exception e) {
            _backgroundColor1 = -1;
        }
        if (_backgroundColor1 < 0) {
            JOptionPane.showMessageDialog(null, "1框默认颜色设置错误");
            return;
        }
        int _backgroundColor2;
        try {
            _backgroundColor2 = Integer.parseInt(textFieldBackground2.getText(), 16);
        } catch (Exception e) {
            _backgroundColor2 = -1;
        }
        if (_backgroundColor2 < 0) {
            JOptionPane.showMessageDialog(null, "2框默认颜色设置错误");
            return;
        }
        int _backgroundColor3;
        try {
            _backgroundColor3 = Integer.parseInt(textFieldBackground3.getText(), 16);
        } catch (Exception e) {
            _backgroundColor3 = -1;
        }
        if (_backgroundColor3 < 0) {
            JOptionPane.showMessageDialog(null, "3框默认颜色设置错误");
            return;
        }
        int _backgroundColor4;
        try {
            _backgroundColor4 = Integer.parseInt(textFieldBackground4.getText(), 16);
        } catch (Exception e) {
            _backgroundColor4 = -1;
        }
        if (_backgroundColor4 < 0) {
            JOptionPane.showMessageDialog(null, "4框默认颜色设置错误");
            return;
        }
        int _fontColor;
        try {
            _fontColor = Integer.parseInt(textFieldFontColor.getText(), 16);
        } catch (Exception e) {
            _fontColor = -1;
        }
        if (_fontColor < 0) {
            JOptionPane.showMessageDialog(null, "未选中框内字体颜色设置错误");
            return;
        }
        int _searchBarColor;
        try {
            _searchBarColor = Integer.parseInt(textFieldSearchBarColor.getText(), 16);
        } catch (Exception e) {
            _searchBarColor = -1;
        }
        if (_searchBarColor < 0) {
            JOptionPane.showMessageDialog(null, "1框默认颜色设置错误");
            return;
        }
        int _maxConnectionNum;
        try {
            _maxConnectionNum = Integer.parseInt(textFieldMaxConnectionNum.getText());
        } catch (Exception e) {
            _maxConnectionNum = -1;
        }
        if (_maxConnectionNum < 0 || _maxConnectionNum > 50) {
            JOptionPane.showMessageDialog(null, "最大连接保持数量设置错误");
            return;
        }
        int _minConnectionNum;
        try {
            _minConnectionNum = Integer.parseInt(textFieldMinConnectionNum.getText());
        } catch (Exception e) {
            _minConnectionNum = -1;
        }
        if (_minConnectionNum < 0 || _minConnectionNum > 50) {
            JOptionPane.showMessageDialog(null, "最小连接保持数量设置错误");
            return;
        }
        if (_minConnectionNum > _maxConnectionNum) {
            JOptionPane.showMessageDialog(null, "最小连接保持数量需要大于最大连接保持数量");
            return;
        }
        int _connectionTimeLimit;
        try {
            _connectionTimeLimit = Integer.parseInt(textFieldConnectionTimeLimit.getText()) * 1000;
        } catch (Exception e) {
            _connectionTimeLimit = -1;
        }
        if (_connectionTimeLimit < 0 || _connectionTimeLimit > 1800000) {
            JOptionPane.showMessageDialog(null, "连接保持时间设置错误");
            return;
        }

        priorityFolder = textFieldPriorityFolder.getText();
        dataPath = textFieldDataPath.getText();
        hotkey = textFieldHotkey.getText();
        HotKeyListener.changeHotKey(hotkey);
        cacheNumLimit = cacheNumLimitTemp;
        updateTimeLimit = updateTimeLimitTemp;
        ignorePath = ignorePathTemp;
        searchDepth = searchDepthTemp;
        transparency = transparencyTemp;
        SearchBar.getInstance().setTransparency(transparency);
        labelColor = _labelColor;
        backgroundColor2 = _backgroundColor2;
        backgroundColor1 = _backgroundColor1;
        backgroundColor3 = _backgroundColor3;
        backgroundColor4 = _backgroundColor4;
        searchBarColor = _searchBarColor;
        fontColorWithCoverage = _fontColorWithCoverage;
        fontColor = _fontColor;
        SearchBar instance = SearchBar.getInstance();
        instance.setBackgroundColor1(backgroundColor1);
        instance.setBackgroundColor2(backgroundColor2);
        instance.setBackgroundColor3(backgroundColor3);
        instance.setBackgroundColor4(backgroundColor4);
        instance.setLabelColor(labelColor);
        instance.setFontColorWithCoverage(fontColorWithCoverage);
        instance.setFontColor(fontColor);
        instance.setSearchBarColor(searchBarColor);
        Color _tmp = new Color(labelColor);
        labelColorChooser.setBackground(_tmp);
        labelColorChooser.setForeground(_tmp);
        _tmp = new Color(backgroundColor1);
        background1Chooser.setBackground(_tmp);
        background1Chooser.setForeground(_tmp);
        _tmp = new Color(backgroundColor2);
        background2Chooser.setForeground(_tmp);
        background2Chooser.setBackground(_tmp);
        _tmp = new Color(backgroundColor3);
        background3Chooser.setForeground(_tmp);
        background3Chooser.setBackground(_tmp);
        _tmp = new Color(backgroundColor4);
        background4Chooser.setForeground(_tmp);
        background4Chooser.setBackground(_tmp);
        _tmp = new Color(fontColorWithCoverage);
        FontColorWithCoverageChooser.setBackground(_tmp);
        FontColorWithCoverageChooser.setForeground(_tmp);
        _tmp = new Color(fontColor);
        FontColorChooser.setBackground(_tmp);
        FontColorChooser.setForeground(_tmp);
        maxConnectionNum = _maxConnectionNum;
        minConnectionNum = _minConnectionNum;
        connectionTimeLimit = _connectionTimeLimit;

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
        allSettings.put("backgroundColor1", backgroundColor1);
        allSettings.put("backgroundColor2", backgroundColor2);
        allSettings.put("backgroundColor3", backgroundColor3);
        allSettings.put("backgroundColor4", backgroundColor4);
        allSettings.put("searchBarColor", searchBarColor);
        allSettings.put("fontColorWithCoverage", fontColorWithCoverage);
        allSettings.put("fontColor", fontColor);
        allSettings.put("maxConnectionNum", maxConnectionNum);
        allSettings.put("minConnectionNum", minConnectionNum);
        allSettings.put("connectionTimeLimit", connectionTimeLimit);
        try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
            buffW.write(allSettings.toJSONString());
            JOptionPane.showMessageDialog(null, "保存成功");
        } catch (IOException ignored) {

        }
        //保存自定义命令
        StringBuilder strb = new StringBuilder();
        for (String each : cmdSet) {
            strb.append(each);
            strb.append("\n");
        }
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File("user/cmds.txt")))) {
            bw.write(strb.toString());
        } catch (IOException ignored) {

        }
    }

    private void setStartup(boolean b) {
        Process p;
        File FileEngine = new File(MainClass.name);
        if (b) {
            try {
                String currentPath = System.getProperty("user.dir");
                File file = new File(currentPath);
                for (File each : Objects.requireNonNull(file.listFiles())) {
                    if (each.getName().contains("elevated")) {
                        FileEngine = each;
                        break;
                    }
                }
                p = Runtime.getRuntime().exec("reg add \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\" /v FileEngine /t REG_SZ /d " + "\"" + FileEngine.getAbsolutePath() + "\"" + " /f");
                p.waitFor();
                BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                String line;
                StringBuilder result = new StringBuilder();
                while ((line = outPut.readLine()) != null) {
                    result.append(line);
                }
                if (!result.toString().isEmpty()) {
                    checkBox1.setSelected(false);
                    JOptionPane.showMessageDialog(null, "添加到开机启动失败，请尝试以管理员身份运行");
                }
            } catch (IOException | InterruptedException ignored) {

            }
        } else {
            try {
                p = Runtime.getRuntime().exec("reg delete \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\" /v FileEngine /f");
                p.waitFor();
                BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                String line;
                StringBuilder result = new StringBuilder();
                while ((line = outPut.readLine()) != null) {
                    result.append(line);
                }
                if (!result.toString().isEmpty()) {
                    checkBox1.setSelected(true);
                    JOptionPane.showMessageDialog(null, "删除开机启动失败，请尝试以管理员身份运行");
                }
            } catch (IOException | InterruptedException ignored) {

            }
        }
        boolean isSelected = checkBox1.isSelected();
        JSONObject allSettings = null;
        try (BufferedReader buffR1 = new BufferedReader(new FileReader(settings))) {
            String line;
            StringBuilder result = new StringBuilder();
            while (null != (line = buffR1.readLine())) {
                result.append(line);
            }
            allSettings = JSON.parseObject(result.toString());
            allSettings.put("isStartup", isSelected);
        } catch (IOException ignored) {

        }
        try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
            assert allSettings != null;
            buffW.write(allSettings.toJSONString());
        } catch (IOException ignored) {

        }
    }

    private void copyFile(InputStream source, File dest) {
        try (OutputStream os = new FileOutputStream(dest); BufferedInputStream bis = new BufferedInputStream(source); BufferedOutputStream bos = new BufferedOutputStream(os)) {
            // 创建缓冲流
            byte[] buffer = new byte[8192];
            int count = bis.read(buffer);
            while (count != -1) {
                //使用缓冲流写数据
                bos.write(buffer, 0, count);
                //刷新
                bos.flush();
                count = bis.read(buffer);
            }
        } catch (IOException ignored) {

        }
    }

    static class moveDesktopFiles implements Runnable {

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
                JOptionPane.showMessageDialog(null, "检测到重名文件，请自行移动");
            }
        }
    }
}