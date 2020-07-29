package FileEngine.frames;

import FileEngine.SQLiteConfig.SQLiteUtil;
import FileEngine.download.DownloadManager;
import FileEngine.download.DownloadUtil;
import FileEngine.checkHotkey.CheckHotKeyUtil;
import FileEngine.pluginSystem.Plugin;
import FileEngine.pluginSystem.PluginUtil;
import FileEngine.search.SearchUtil;
import FileEngine.threadPool.CachedThreadPool;
import FileEngine.translate.TranslateUtil;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SerializerFeature;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.Proxy;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.sql.Statement;
import java.util.HashSet;
import java.util.Scanner;


public class SettingsFrame {
    public static final String version = "2.5"; //TODO 更改版本号
    private static volatile boolean mainExit = false;
    private static volatile int cacheNumLimit;
    private static volatile String hotkey;
    private static volatile int updateTimeLimit;
    private static volatile String ignorePath;
    private static volatile String priorityFolder;
    private static volatile int searchDepth;
    private static volatile boolean isDefaultAdmin;
    private static volatile boolean isLoseFocusClose;
    private static volatile int openLastFolderKeyCode;
    private static volatile int runAsAdminKeyCode;
    private static volatile int copyPathKeyCode;
    private static volatile float transparency;
    private static volatile int tmp_copyPathKeyCode;
    private static volatile String proxyAddress;
    private static volatile int proxyPort;
    private static volatile String proxyUserName;
    private static volatile String proxyPassword;
    private static volatile int proxyType;
    private static File tmp;
    private static File settings;
    private static HashSet<String> cmdSet;
    private static volatile int tmp_openLastFolderKeyCode;
    private static volatile int tmp_runAsAdminKeyCode;
    private static CheckHotKeyUtil hotKeyListener;
    private static volatile int labelColor;
    private static volatile int defaultBackgroundColor;
    private static volatile int fontColorWithCoverage;
    private static volatile int fontColor;
    private static volatile int searchBarColor;
    private static volatile boolean isStartup;
    private static SearchBar searchBar;
    private final JFrame frame;
    private static ImageIcon frameIcon;
    private JTextField textFieldUpdateInterval;
    private JTextField textFieldCacheNum;
    private JTextArea textAreaIgnorePath;
    private JTextField textFieldSearchDepth;
    private JCheckBox checkBoxAddToStartup;
    private JLabel labelSetIgnorePathTip;
    private JLabel labelUpdateInterval;
    private JLabel labelOpenSearchBarHotKey;
    private JLabel labelMaxCacheNum;
    private JLabel labelSecond;
    private JLabel labelSearchDepth;
    private JPanel panel;
    private JLabel labelConstIgnorePathTip;
    private JButton buttonSaveAndRemoveDesktop;
    private JButton buttonChooseFile;
    private JScrollPane scrollpaneIgnorePath;
    private JTextField textFieldHotkey;
    private JLabel labeltipPriorityFolder;
    private JTextField textFieldPriorityFolder;
    private JButton ButtonPriorityFolder;
    private JTabbedPane tabbedPane;
    private JCheckBox checkBoxAdmin;
    private JCheckBox checkBoxLoseFocus;
    private JLabel labelRunAsAdminHotKey;
    private JTextField textFieldRunAsAdminHotKey;
    private JLabel labelOpenFolderHotKey;
    private JTextField textFieldOpenLastFolder;
    private JButton buttonAddCMD;
    private JButton buttonDelCmd;
    private JScrollPane scrollPaneCmd;
    private JList<Object> listCmds;
    private JButton buttonSave;
    private JLabel labelAbout;
    private JLabel labelAboutGithub;
    private JLabel labelGitHubTip;
    private JLabel labelIcon;
    private JLabel labelGithubIssue;
    private JButton buttonCheckUpdate;
    private JLabel labelCopyPathHotKey;
    private JTextField textFieldCopyPath;
    private JLabel labelVersion;
    private JPanel tabGeneral;
    private JPanel tabSearchSettings;
    private JPanel tabSearchBarSettings;
    private JPanel tabHotKey;
    private JPanel tabCommands;
    private JTextField textFieldTransparency;
    private JLabel labelTransparency;
    private JPanel tabColorSettings;
    private JLabel labelColorTip;
    private JTextField textFieldBackgroundDefault;
    private JTextField textFieldFontColorWithCoverage;
    private JTextField textFieldLabelColor;
    private JLabel labelLabelColor;
    private JLabel labelFontColor;
    private JLabel labelDefaultColor;
    private JLabel labelSharpSign1;
    private JLabel labelSharpSign2;
    private JLabel labelSharp4;
    private JButton buttonResetColor;
    private JTextField textFieldFontColor;
    private JLabel labelSharpSign5;
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
    private JLabel labelDescription;
    private JLabel labelWebLookAndFeel;
    private JLabel labelFastJson;
    private JLabel labelJna;
    private JPanel tabLanguage;
    private JList<Object> listLanguage;
    private JLabel labelLanguageChooseTip;
    private JLabel labelPlaceHolder4;
    private JLabel labelTranslationTip;
    private JLabel labelPlaceHolder1;
    private JLabel labelPlaceHolder5;
    private JLabel labelPlaceHolderWhatever;
    private JPanel tabPlugin;
    private JLabel labelInstalledPluginNum;
    private JLabel labelPluginNum;
    private JLabel labelPlaceHolder2;
    private JLabel labelPlaceHolder3;
    private JPanel PluginPanel;
    private JPanel PluginSettingsPanel;
    private JList<Object> listPlugins;
    private JLabel PluginIconLabel;
    private JLabel PluginNamelabel;
    private JTextArea textAreaDescription;
    private JLabel labelPluginVersion;
    private JButton buttonUpdatePlugin;
    private JPanel tabAbout;
    private JScrollPane scrollPane;
    private JLabel labelSQLite;
    private JButton buttonPluginMarket;
    private JLabel labelAuthor;
    private JLabel labelOfficialSite;
    private JLabel labelDownloadProgress;
    private JLabel labelProgress;
    private JButton buttonVacuum;
    private JLabel labelVacuumTip;
    private JPanel tabProxy;
    private JRadioButton radioButtonNoProxy;
    private JRadioButton radioButtonUseProxy;
    private JTextField textFieldAddress;
    private JTextField textFieldPort;
    private JTextField textFieldUserName;
    private JTextField textFieldPassword;
    private JLabel labelAddress;
    private JLabel labelPort;
    private JLabel labelUserName;
    private JLabel labelPassword;
    private JLabel labelPlaceHolder7;
    private JRadioButton radioButtonProxyTypeHttp;
    private JRadioButton radioButtonProxyTypeSocks5;
    private JLabel labelProxyTip;
    private JLabel labelPlaceHolder6;
    private JLabel labelPlaceHolder8;
    private JSeparator proxySeperater;
    private JLabel labelVacuumStatus;
    private JLabel labelApiVersion;

    private static class SettingsFrameBuilder {
        private static final SettingsFrame instance = new SettingsFrame();
    }

    private static final int PROXY_HTTP = 0;
    private static final int PROXY_SOCKS = 1;
    private static final int PROXY_DIRECT = 2;

    public static class ProxyInfo {
        public final String address;
        public final int port;
        public final String userName;
        public final String password;
        public final Proxy.Type type;

        protected ProxyInfo(String proxyAddress, int proxyPort, String proxyUserName, String proxyPassword, int proxyType) {
            this.address = proxyAddress;
            this.port = proxyPort;
            this.userName = proxyUserName;
            this.password = proxyPassword;
            if (PROXY_HTTP == proxyType) {
                this.type = Proxy.Type.HTTP;
            } else if (PROXY_SOCKS == proxyType) {
                this.type = Proxy.Type.SOCKS;
            } else {
                this.type = Proxy.Type.DIRECT;
            }
        }
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

    public static String getName() {
        return "File-Engine-x64.exe";
    }

    public static ProxyInfo getProxy() {
        return new ProxyInfo(proxyAddress, proxyPort, proxyUserName, proxyPassword, proxyType);
    }

    private void addCheckboxListener() {
        checkBoxAddToStartup.addActionListener(e -> setStartup(checkBoxAddToStartup.isSelected()));
        checkBoxAdmin.addActionListener(e -> isDefaultAdmin = checkBoxAdmin.isSelected());
        checkBoxLoseFocus.addActionListener(e -> isLoseFocusClose = checkBoxLoseFocus.isSelected());
    }

    private void addButtonRemoveDesktopListener() {
        buttonSaveAndRemoveDesktop.addActionListener(e -> {
            String currentFolder = new File("").getAbsolutePath();
            if (currentFolder.equals(FileSystemView.getFileSystemView().getHomeDirectory().getAbsolutePath()) || "C:\\Users\\Public\\Desktop".equals(currentFolder)) {
                JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("The program is detected on the desktop and cannot be moved"));
                return;
            }
            int isConfirmed = JOptionPane.showConfirmDialog(frame, TranslateUtil.getInstance().getTranslation("Whether to remove and backup all files on the desktop," +
                    "they will be in the program's Files folder, which may take a few minutes"));
            if (isConfirmed == 0) {
                Thread fileMover = new Thread(new MoveDesktopFiles());
                fileMover.start();
            }
        });
    }

    private void addFileChooserButtonListener() {
        buttonChooseFile.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), TranslateUtil.getInstance().getTranslation("Choose"));
            File file = fileChooser.getSelectedFile();
            if (file != null && returnValue == JFileChooser.APPROVE_OPTION) {
                textAreaIgnorePath.append(file.getAbsolutePath() + ",\n");
            }
        });
    }

    private void addTextFieldListener() {
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
    }

    private void addTextFieldRunAsAdminListener() {
        textFieldRunAsAdminHotKey.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                int code = e.getKeyCode();
                if (code == 17) {
                    textFieldRunAsAdminHotKey.setText("Ctrl + Enter");
                    tmp_runAsAdminKeyCode = 17;
                } else if (code == 16) {
                    textFieldRunAsAdminHotKey.setText("Shift + Enter");
                    tmp_runAsAdminKeyCode = 16;
                } else if (code == 18) {
                    textFieldRunAsAdminHotKey.setText("Alt + Enter");
                    tmp_runAsAdminKeyCode = 18;
                }
            }
        });
    }

    private void addPriorityFileChooserListener() {
        ButtonPriorityFolder.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), TranslateUtil.getInstance().getTranslation("Choose"));
            File file = fileChooser.getSelectedFile();
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                textFieldPriorityFolder.setText(file.getAbsolutePath());
            }
        });
    }

    private void addPriorityTextFieldListener() {
        textFieldPriorityFolder.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2) {
                    textFieldPriorityFolder.setText(null);
                }
            }
        });
    }

    private void addTextFieldOpenLastFolderListener() {
        textFieldOpenLastFolder.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                int code = e.getKeyCode();
                if (code == 17) {
                    textFieldOpenLastFolder.setText("Ctrl + Enter");
                    tmp_openLastFolderKeyCode = 17;
                } else if (code == 16) {
                    textFieldOpenLastFolder.setText("Shift + Enter");
                    tmp_openLastFolderKeyCode = 16;
                } else if (code == 18) {
                    textFieldOpenLastFolder.setText("Alt + Enter");
                    tmp_openLastFolderKeyCode = 18;
                }
            }
        });
    }

    private void addButtonCMDListener() {
        buttonAddCMD.addActionListener(e -> {
            String name = JOptionPane.showInputDialog(TranslateUtil.getInstance().getTranslation("Please enter the ID of the command, then you can enter \": identifier\" in the search box to execute the command directly"));
            if (name == null || name.isEmpty()) {
                //未输入
                return;
            }
            if ("update".equals(name) || "clearbin".equals(name) || "help".equals(name) || "version".equals(name) || isRepeatCommand(name)) {
                JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Conflict with existing commands"));
                return;
            }
            String cmd;
            JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Please select the location of the executable file (a folder is also acceptable)"));
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
            int returnValue = fileChooser.showDialog(new Label(), TranslateUtil.getInstance().getTranslation("Choose"));
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                cmd = fileChooser.getSelectedFile().getAbsolutePath();
                cmdSet.add(":" + name + ";" + cmd);
                listCmds.setListData(cmdSet.toArray());
            }

        });
    }

    private void addButtonDelCMDListener() {
        buttonDelCmd.addActionListener(e -> {
            String del = (String) listCmds.getSelectedValue();
            cmdSet.remove(del);
            listCmds.setListData(cmdSet.toArray());

        });
    }

    private void addButtonSaveListener() {
        buttonSave.addActionListener(e -> saveChanges());
    }

    private void addGitHubLabelListener() {
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
    }

    private void addCheckForUpdateButtonListener() {
        buttonCheckUpdate.addActionListener(e -> {
            boolean isDownloading = DownloadUtil.getInstance().getDownloadStatus(getName()) != DownloadManager.DOWNLOAD_NO_TASK;

            //检查是否已开始下载
            if (!isDownloading) {
                JSONObject updateInfo;
                String latestVersion;
                try {
                    updateInfo = getUpdateInfo();
                    if (updateInfo != null) {
                        latestVersion = updateInfo.getString("version");
                    } else {
                        throw new IOException("failed");
                    }
                } catch (IOException | InterruptedException e1) {
                    JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Check update failed"));
                    return;
                }

                if (Double.parseDouble(latestVersion) > Double.parseDouble(version) || isDebug()) {
                    String description = updateInfo.getString("description");
                    int result = JOptionPane.showConfirmDialog(frame,
                            TranslateUtil.getInstance().getTranslation(
                                    "New Version available") + latestVersion + "," +
                                    TranslateUtil.getInstance().getTranslation("Whether to update") + "\n" +
                                    TranslateUtil.getInstance().getTranslation("update content") + "\n" + description);
                    if (result == 0) {
                        //开始更新,下载更新文件到tmp
                        String urlChoose;
                        String fileName;
                        urlChoose = "url64";
                        fileName = getName();
                        DownloadUtil download = DownloadUtil.getInstance();
                        download.downLoadFromUrl(updateInfo.getString(urlChoose), fileName, tmp.getAbsolutePath());
                        //更新button为取消
                        buttonCheckUpdate.setText(TranslateUtil.getInstance().getTranslation("Cancel"));
                    }
                } else {
                    JOptionPane.showMessageDialog(frame,
                            TranslateUtil.getInstance().getTranslation("Latest version:") + latestVersion + "\n" +
                                    TranslateUtil.getInstance().getTranslation("The current version is the latest"));
                }
            } else {
                //取消下载
                String fileName = getName();
                DownloadUtil instance = DownloadUtil.getInstance();
                instance.cancelDownload(fileName);
                CachedThreadPool.getInstance().executeTask(() -> {
                    //等待下载取消
                    try {
                        while (instance.getDownloadStatus(fileName) != DownloadManager.DOWNLOAD_INTERRUPTED) {
                            if (instance.getDownloadStatus(fileName) == DownloadManager.DOWNLOAD_ERROR) {
                                break;
                            }
                            if (buttonCheckUpdate.isEnabled()) {
                                buttonCheckUpdate.setEnabled(false);
                            }
                            Thread.sleep(50);
                        }
                    } catch (InterruptedException ignored) {
                    }
                    //复位button
                    buttonCheckUpdate.setText(TranslateUtil.getInstance().getTranslation("Check for update"));
                    buttonCheckUpdate.setEnabled(true);
                });
            }
        });
    }

    private void addTextFieldCopyPathListener() {
        textFieldCopyPath.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                int code = e.getKeyCode();
                if (code == 17) {
                    textFieldCopyPath.setText("Ctrl + Enter");
                    tmp_copyPathKeyCode = 17;
                } else if (code == 16) {
                    textFieldCopyPath.setText("Shift + Enter");
                    tmp_copyPathKeyCode = 16;
                } else if (code == 18) {
                    textFieldCopyPath.setText("Alt + Enter");
                    tmp_copyPathKeyCode = 18;
                }
            }
        });
    }

    private void addResetColorButtonListener() {
        buttonResetColor.addActionListener(e -> {
            textFieldFontColorWithCoverage.setText(Integer.toHexString(0x6666ff));
            textFieldSearchBarColor.setText(Integer.toHexString(0xffffff));
            textFieldLabelColor.setText(Integer.toHexString(0xcccccc));
            textFieldBackgroundDefault.setText(Integer.toHexString(0xffffff));
            textFieldFontColor.setText(Integer.toHexString(0));
        });
    }

    private void addColorChooserLabelListener() {
        labelColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, TranslateUtil.getInstance().getTranslation("Choose Color"), null);
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
                Color color = JColorChooser.showDialog(null, TranslateUtil.getInstance().getTranslation("Choose Color"), null);
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
                Color color = JColorChooser.showDialog(null, TranslateUtil.getInstance().getTranslation("Choose Color"), null);
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
                Color color = JColorChooser.showDialog(null, TranslateUtil.getInstance().getTranslation("Choose Color"), null);
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
                Color color = JColorChooser.showDialog(null, TranslateUtil.getInstance().getTranslation("Choose Color"), null);
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

    private void addListPluginMouseListener() {
        listPlugins.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                String pluginName = (String) listPlugins.getSelectedValue();
                if (pluginName != null) {
                    String pluginIdentifier = PluginUtil.getIdentifierByName(pluginName);
                    Plugin plugin = PluginUtil.getPluginByIdentifier(pluginIdentifier);
                    ImageIcon icon;
                    String description;
                    String officialSite;
                    String version;
                    String author;
                    int apiVersion;
                    icon = plugin.getPluginIcon();
                    description = plugin.getDescription();
                    officialSite = plugin.getOfficialSite();
                    version = plugin.getVersion();
                    author = plugin.getAuthor();
                    apiVersion = plugin.getApiVersion();

                    labelPluginVersion.setText(TranslateUtil.getInstance().getTranslation("Version") + ":" + version);
                    labelApiVersion.setText("API " + TranslateUtil.getInstance().getTranslation("Version") + ":" + apiVersion);
                    PluginIconLabel.setIcon(icon);
                    PluginNamelabel.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                    textAreaDescription.setText(description);
                    labelAuthor.setText(TranslateUtil.getInstance().getTranslation("Author") + ":" + author);
                    labelOfficialSite.setText("<html><a href='" + officialSite + "'><font size=\"4\">" + pluginName + "</font></a></html>");
                    buttonUpdatePlugin.setVisible(true);
                }
            }
        });
    }

    private void addPluginOfficialSiteListener() {
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

    private void addButtonProxyListener() {
        radioButtonNoProxy.addActionListener(e -> {
            radioButtonProxyTypeHttp.setEnabled(false);
            radioButtonProxyTypeSocks5.setEnabled(false);
            textFieldAddress.setEnabled(false);
            textFieldPort.setEnabled(false);
            textFieldUserName.setEnabled(false);
            textFieldPassword.setEnabled(false);
        });

        radioButtonUseProxy.addActionListener(e -> {
            radioButtonProxyTypeHttp.setEnabled(true);
            radioButtonProxyTypeSocks5.setEnabled(true);
            textFieldAddress.setEnabled(true);
            textFieldPort.setEnabled(true);
            textFieldUserName.setEnabled(true);
            textFieldPassword.setEnabled(true);
            radioButtonProxyTypeHttp.setSelected(true);
        });
    }

    private void addButtonVacuumListener() {
        buttonVacuum.addActionListener(e -> {
            int ret = JOptionPane.showConfirmDialog(frame, TranslateUtil.getInstance().getTranslation("Confirm whether to start optimizing the database?"));
            if (0 == ret) {
                int status = SearchUtil.getInstance().getStatus();
                if (status == SearchUtil.NORMAL) {
                    if (isDebug()) {
                        System.out.println("开始VACUUM");
                    }
                    SearchUtil.getInstance().setStatus(SearchUtil.VACUUM);
                    CachedThreadPool.getInstance().executeTask(() -> {
                        try (Statement stmt = SQLiteUtil.getStatement()) {
                            stmt.execute("VACUUM;");
                        } catch (Exception throwables) {
                            if (isDebug()) {
                                throwables.printStackTrace();
                            }
                        } finally {
                            if (isDebug()) {
                                System.out.println("结束VACUUM");
                            }
                            SearchUtil.getInstance().setStatus(SearchUtil.NORMAL);
                        }
                    });
                    CachedThreadPool.getInstance().executeTask(() -> {
                        try {
                            SearchUtil instance = SearchUtil.getInstance();
                            while (instance.getStatus() == SearchUtil.VACUUM) {
                                labelVacuumStatus.setText(TranslateUtil.getInstance().getTranslation("Optimizing..."));
                                Thread.sleep(50);
                            }
                            labelVacuumStatus.setText(TranslateUtil.getInstance().getTranslation("Optimized"));
                            Thread.sleep(3000);
                            labelVacuumStatus.setText("");
                        } catch (InterruptedException ignored) {
                        }
                    });
                } else if (status == SearchUtil.MANUAL_UPDATE) {
                    JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Database is not usable yet, please wait..."));
                } else if (status == SearchUtil.VACUUM) {
                    JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Task is still running."));
                }
            }
        });
    }

    private void addButtonViewPluginMarketListener() {
        buttonPluginMarket.addActionListener(e -> {
            PluginMarket pluginMarket = PluginMarket.getInstance();
            pluginMarket.showWindow();
        });
    }

    private void addButtonPluginUpdateCheckListener() {
        buttonUpdatePlugin.addActionListener(e -> {
            boolean isVersionLatest;
            String pluginName = (String) listPlugins.getSelectedValue();
            String pluginIdentifier = PluginUtil.getIdentifierByName(pluginName);
            Plugin plugin = PluginUtil.getPluginByIdentifier(pluginIdentifier);
            String pluginFullName = pluginName + ".jar";
            isVersionLatest = plugin.isLatest();
            if (isVersionLatest) {
                JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("The current Version is the latest."));
            } else {
                //检查是否已经开始下载
                boolean isPluginUpdating = DownloadUtil.getInstance().getDownloadStatus(pluginFullName) != DownloadManager.DOWNLOAD_NO_TASK;

                if (!isPluginUpdating) {
                    int ret = JOptionPane.showConfirmDialog(frame, TranslateUtil.getInstance().getTranslation("New version available, do you want to update?"));
                    if (ret == 0) {
                        //开始下载
                        String url = plugin.getUpdateURL();
                        DownloadUtil.getInstance().downLoadFromUrl(url, pluginFullName, "tmp/pluginsUpdate");
                        buttonUpdatePlugin.setText(TranslateUtil.getInstance().getTranslation("Cancel"));
                    }
                } else {
                    //取消下载
                    DownloadUtil instance = DownloadUtil.getInstance();
                    instance.cancelDownload(pluginFullName);
                    CachedThreadPool.getInstance().executeTask(() -> {
                        try {
                            //等待下载取消
                            while (instance.getDownloadStatus(pluginFullName) != DownloadManager.DOWNLOAD_INTERRUPTED) {
                                if (instance.getDownloadStatus(pluginFullName) == DownloadManager.DOWNLOAD_ERROR) {
                                    break;
                                }
                                if (buttonUpdatePlugin.isEnabled()) {
                                    buttonUpdatePlugin.setEnabled(false);
                                }
                                Thread.sleep(50);
                            }
                            //复位button
                            buttonUpdatePlugin.setText(TranslateUtil.getInstance().getTranslation("Install"));
                            buttonUpdatePlugin.setEnabled(true);
                        } catch (InterruptedException ignored) {
                        }
                    });
                }
            }
        });
    }

    private void initGUI() {
        //设置窗口显示
        labelAboutGithub.setText("<html><a href='https://github.com/XUANXUQAQ/File-Engine'><font size=\"4\">File-Engine</font></a></html>");
        labelWebLookAndFeel.setText("1.WebLookAndFeel");
        labelFastJson.setText("2.FastJson");
        labelJna.setText("3.Java-Native-Access");
        labelSQLite.setText("4.SQLite-JDBC");
        labelPluginNum.setText(String.valueOf(PluginUtil.getInstalledPluginNum()));
        ImageIcon imageIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        labelIcon.setIcon(imageIcon);
        labelVersion.setText(TranslateUtil.getInstance().getTranslation("Current Version:") + version);
        checkBoxAddToStartup.setSelected(isStartup);
        textFieldUpdateInterval.setText(String.valueOf(updateTimeLimit));
        textAreaIgnorePath.setText(ignorePath.replaceAll(",", ",\n"));
        textFieldCacheNum.setText(String.valueOf(cacheNumLimit));
        textFieldSearchDepth.setText(String.valueOf(searchDepth));
        textFieldHotkey.setText(hotkey);
        textFieldPriorityFolder.setText(priorityFolder);
        checkBoxAdmin.setSelected(isDefaultAdmin);
        textFieldSearchBarColor.setText(Integer.toHexString(searchBarColor));
        Color tmp_searchBarColor = new Color(searchBarColor);
        searchBarColorChooser.setBackground(tmp_searchBarColor);
        searchBarColorChooser.setForeground(tmp_searchBarColor);
        textFieldBackgroundDefault.setText(Integer.toHexString(defaultBackgroundColor));
        Color tmp_defaultBackgroundColor = new Color(defaultBackgroundColor);
        defaultBackgroundChooser.setBackground(tmp_defaultBackgroundColor);
        defaultBackgroundChooser.setForeground(tmp_defaultBackgroundColor);
        textFieldLabelColor.setText(Integer.toHexString(labelColor));
        Color tmp_labelColor = new Color(labelColor);
        labelColorChooser.setBackground(tmp_labelColor);
        labelColorChooser.setForeground(tmp_labelColor);
        textFieldFontColorWithCoverage.setText(Integer.toHexString(fontColorWithCoverage));
        Color tmp_fontColorWithCoverage = new Color(fontColorWithCoverage);
        FontColorWithCoverageChooser.setBackground(tmp_fontColorWithCoverage);
        FontColorWithCoverageChooser.setForeground(tmp_fontColorWithCoverage);
        checkBoxLoseFocus.setSelected(isLoseFocusClose);
        textFieldTransparency.setText(String.valueOf(transparency));
        textFieldFontColor.setText(Integer.toHexString(fontColor));
        Color tmp_fontColor = new Color(fontColor);
        FontColorChooser.setBackground(tmp_fontColor);
        FontColorChooser.setForeground(tmp_fontColor);
        if (runAsAdminKeyCode == 17) {
            textFieldRunAsAdminHotKey.setText("Ctrl + Enter");
        } else if (runAsAdminKeyCode == 16) {
            textFieldRunAsAdminHotKey.setText("Shift + Enter");
        } else if (runAsAdminKeyCode == 18) {
            textFieldRunAsAdminHotKey.setText("Alt + Enter");
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
        listLanguage.setListData(TranslateUtil.getInstance().getLanguageSet().toArray());
        listLanguage.setSelectedValue(TranslateUtil.getInstance().getLanguage(), true);
        Object[] plugins = PluginUtil.getPluginArray();
        listPlugins.setListData(plugins);
        buttonUpdatePlugin.setVisible(false);
        if (plugins.length == 0) {
            PluginSettingsPanel.setVisible(false);
        }
        if (proxyType == PROXY_DIRECT) {
            radioButtonNoProxy.setSelected(true);
            radioButtonUseProxy.setSelected(false);
            radioButtonProxyTypeHttp.setEnabled(false);
            radioButtonProxyTypeSocks5.setEnabled(false);
            textFieldAddress.setEnabled(false);
            textFieldPort.setEnabled(false);
            textFieldUserName.setEnabled(false);
            textFieldPassword.setEnabled(false);
        } else {
            radioButtonUseProxy.setSelected(true);
            radioButtonNoProxy.setSelected(false);
            radioButtonProxyTypeHttp.setEnabled(true);
            radioButtonProxyTypeSocks5.setEnabled(true);
            textFieldAddress.setEnabled(true);
            textFieldPort.setEnabled(true);
            textFieldUserName.setEnabled(true);
            textFieldPassword.setEnabled(true);
            selectProxyType();
        }
        textFieldAddress.setText(proxyAddress);
        textFieldPort.setText(String.valueOf(proxyPort));
        textFieldUserName.setText(proxyUserName);
        textFieldPassword.setText(proxyPassword);
    }

    private void selectProxyType() {
        if (proxyType == PROXY_HTTP) {
            radioButtonProxyTypeHttp.setSelected(true);
        } else if (proxyType == PROXY_SOCKS) {
            radioButtonProxyTypeSocks5.setSelected(true);
        }
    }

    private SettingsFrame() {
        frame = new JFrame("Settings");
        tmp = new File("tmp");
        settings = new File("user/settings.json");
        cmdSet = new HashSet<>();
        frameIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        ButtonGroup proxyButtonGroup = new ButtonGroup();
        proxyButtonGroup.add(radioButtonNoProxy);
        proxyButtonGroup.add(radioButtonUseProxy);

        ButtonGroup proxyTypeButtonGroup = new ButtonGroup();
        proxyTypeButtonGroup.add(radioButtonProxyTypeHttp);
        proxyTypeButtonGroup.add(radioButtonProxyTypeSocks5);

        readAllSettings();

        hotKeyListener = CheckHotKeyUtil.getInstance();
        searchBar = SearchBar.getInstance();

        setAllSettings();

        translate();

        addCheckboxListener();

        addButtonRemoveDesktopListener();

        addFileChooserButtonListener();

        addTextFieldListener();

        addPriorityFileChooserListener();

        addPriorityTextFieldListener();

        addTextFieldRunAsAdminListener();

        addTextFieldOpenLastFolderListener();

        addButtonCMDListener();

        addButtonDelCMDListener();

        addButtonSaveListener();

        addGitHubLabelListener();

        addCheckForUpdateButtonListener();

        addTextFieldCopyPathListener();

        addResetColorButtonListener();

        addColorChooserLabelListener();

        addListPluginMouseListener();

        addButtonPluginUpdateCheckListener();

        addButtonViewPluginMarketListener();

        addPluginOfficialSiteListener();

        addButtonVacuumListener();

        addButtonProxyListener();

        initGUI();

        initThreadPool();
    }

    private void checkDownloadTask(JLabel label, JButton button, String fileName, String originButtonString) throws InterruptedException {
        //设置进度显示线程
        double progress;
        if (DownloadUtil.getInstance().getDownloadStatus(fileName) != DownloadManager.DOWNLOAD_NO_TASK) {
            progress = DownloadUtil.getInstance().getDownloadProgress(fileName);
            label.setText(TranslateUtil.getInstance().getTranslation("Downloading:") + (int) (progress * 100) + "%");

            int downloadingStatus = DownloadUtil.getInstance().getDownloadStatus(fileName);
            if (downloadingStatus == DownloadManager.DOWNLOAD_DONE) {
                //下载完成，禁用按钮
                label.setText(TranslateUtil.getInstance().getTranslation("Download Done"));
                label.setText(TranslateUtil.getInstance().getTranslation("Downloaded"));
                label.setEnabled(false);
                File updatePluginSign = new File("user/update");
                if (!updatePluginSign.exists()) {
                    try {
                        updatePluginSign.createNewFile();
                    } catch (IOException ignored) {
                    }
                }
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_ERROR) {
                //下载错误，重置button
                label.setText(TranslateUtil.getInstance().getTranslation("Download failed"));
                button.setText(TranslateUtil.getInstance().getTranslation(originButtonString));
                button.setEnabled(true);
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_DOWNLOADING) {
                //正在下载
                button.setText(TranslateUtil.getInstance().getTranslation("Cancel"));
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_INTERRUPTED) {
                //用户自行中断
                label.setText("");
                button.setText(TranslateUtil.getInstance().getTranslation(originButtonString));
                button.setEnabled(true);
            }
        } else {
            label.setText("");
            button.setText(TranslateUtil.getInstance().getTranslation(originButtonString));
            button.setEnabled(true);
        }
        Thread.sleep(100);
    }

    private void addShowDownloadProgressTask(JLabel label, JButton button, String fileName) {
        try {
            String originString = button.getText();
            while (SettingsFrame.isNotMainExit()) {
                checkDownloadTask(label, button, fileName, originString);
            }
        } catch (InterruptedException ignored) {
        }
    }

    private void initThreadPool() {
        CachedThreadPool.getInstance().executeTask(() -> addShowDownloadProgressTask(labelDownloadProgress, buttonCheckUpdate, getName()));

        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                String fileName;
                String originString = buttonUpdatePlugin.getText();
                while (isNotMainExit()) {
                    fileName = (String) listPlugins.getSelectedValue();
                    checkDownloadTask(labelProgress, buttonUpdatePlugin, fileName + ".jar", originString);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private void translate() {
        tabbedPane.setTitleAt(0, TranslateUtil.getInstance().getTranslation("General"));
        tabbedPane.setTitleAt(1, TranslateUtil.getInstance().getTranslation("Search settings"));
        tabbedPane.setTitleAt(2, TranslateUtil.getInstance().getTranslation("Search bar settings"));
        tabbedPane.setTitleAt(3, TranslateUtil.getInstance().getTranslation("Proxy settings"));
        tabbedPane.setTitleAt(4, TranslateUtil.getInstance().getTranslation("Plugins"));
        tabbedPane.setTitleAt(5, TranslateUtil.getInstance().getTranslation("Hotkey settings"));
        tabbedPane.setTitleAt(6, TranslateUtil.getInstance().getTranslation("Language settings"));
        tabbedPane.setTitleAt(7, TranslateUtil.getInstance().getTranslation("My commands"));
        tabbedPane.setTitleAt(8, TranslateUtil.getInstance().getTranslation("Color settings"));
        tabbedPane.setTitleAt(9, TranslateUtil.getInstance().getTranslation("About"));
        checkBoxAddToStartup.setText(TranslateUtil.getInstance().getTranslation("Add to startup"));
        buttonSaveAndRemoveDesktop.setText(TranslateUtil.getInstance().getTranslation("Backup and remove all desktop files"));
        labelMaxCacheNum.setText(TranslateUtil.getInstance().getTranslation("Set the maximum number of caches:"));
        ButtonPriorityFolder.setText(TranslateUtil.getInstance().getTranslation("Choose"));
        buttonChooseFile.setText(TranslateUtil.getInstance().getTranslation("Choose"));
        labelUpdateInterval.setText(TranslateUtil.getInstance().getTranslation("File update detection interval:"));
        labelSecond.setText(TranslateUtil.getInstance().getTranslation("Seconds"));
        labelSearchDepth.setText(TranslateUtil.getInstance().getTranslation("Search depth (too large may affect performance):"));
        labeltipPriorityFolder.setText(TranslateUtil.getInstance().getTranslation("Priority search folder location (double-click to clear):"));
        labelConstIgnorePathTip.setText(TranslateUtil.getInstance().getTranslation("Separate different paths with commas, and ignore C:\\Windows by default"));
        labelSetIgnorePathTip.setText(TranslateUtil.getInstance().getTranslation("Set ignore folder:"));
        checkBoxLoseFocus.setText(TranslateUtil.getInstance().getTranslation("Close search bar when focus lost"));
        checkBoxAdmin.setText(TranslateUtil.getInstance().getTranslation("Open other programs as an administrator (provided that the software has privileges)"));
        labelTransparency.setText(TranslateUtil.getInstance().getTranslation("Search bar transparency:"));
        labelOpenSearchBarHotKey.setText(TranslateUtil.getInstance().getTranslation("Open search bar:"));
        labelRunAsAdminHotKey.setText(TranslateUtil.getInstance().getTranslation("Run as administrator:"));
        labelOpenFolderHotKey.setText(TranslateUtil.getInstance().getTranslation("Open the parent folder:"));
        labelCopyPathHotKey.setText(TranslateUtil.getInstance().getTranslation("Copy path:"));
        labelCmdTip2.setText(TranslateUtil.getInstance().getTranslation("You can add custom commands here. After adding, you can enter \": + your set identifier\" in the search box to quickly open"));
        buttonAddCMD.setText(TranslateUtil.getInstance().getTranslation("Add"));
        buttonDelCmd.setText(TranslateUtil.getInstance().getTranslation("Delete"));
        labelColorTip.setText(TranslateUtil.getInstance().getTranslation("Please enter the hexadecimal value of RGB color"));
        labelSearchBarColor.setText(TranslateUtil.getInstance().getTranslation("Search bar Color:"));
        labelLabelColor.setText(TranslateUtil.getInstance().getTranslation("Chosen label color:"));
        labelFontColor.setText(TranslateUtil.getInstance().getTranslation("Chosen label font Color:"));
        labelDefaultColor.setText(TranslateUtil.getInstance().getTranslation("Default background Color:"));
        labelNotChosenFontColor.setText(TranslateUtil.getInstance().getTranslation("Unchosen label Color:"));
        buttonResetColor.setText(TranslateUtil.getInstance().getTranslation("Reset to default"));
        labelGitHubTip.setText(TranslateUtil.getInstance().getTranslation("This is an open source software,GitHub:"));
        labelGithubIssue.setText(TranslateUtil.getInstance().getTranslation("If you find a bug or have some suggestions, welcome to GitHub for feedback"));
        buttonCheckUpdate.setText(TranslateUtil.getInstance().getTranslation("Check for update"));
        labelDescription.setText(TranslateUtil.getInstance().getTranslation("Thanks for the following project"));
        buttonSave.setText(TranslateUtil.getInstance().getTranslation("Save"));
        labelTranslationTip.setText(TranslateUtil.getInstance().getTranslation("The translation might not be 100% accurate"));
        labelLanguageChooseTip.setText(TranslateUtil.getInstance().getTranslation("Choose a language"));
        labelVersion.setText(TranslateUtil.getInstance().getTranslation("Current Version:") + version);
        labelInstalledPluginNum.setText(TranslateUtil.getInstance().getTranslation("Installed plugins num:"));
        buttonUpdatePlugin.setText(TranslateUtil.getInstance().getTranslation("Check for update"));
        buttonPluginMarket.setText(TranslateUtil.getInstance().getTranslation("Plugin Market"));
        labelVacuumTip.setText(TranslateUtil.getInstance().getTranslation("Click to organize the database and reduce the size of the database, but it will consume a lot of time."));
        radioButtonNoProxy.setText(TranslateUtil.getInstance().getTranslation("No proxy"));
        radioButtonUseProxy.setText(TranslateUtil.getInstance().getTranslation("Configure proxy"));
        labelAddress.setText(TranslateUtil.getInstance().getTranslation("Address"));
        labelPort.setText(TranslateUtil.getInstance().getTranslation("Port"));
        labelUserName.setText(TranslateUtil.getInstance().getTranslation("User name"));
        labelPassword.setText(TranslateUtil.getInstance().getTranslation("Password"));
        labelProxyTip.setText(TranslateUtil.getInstance().getTranslation("If you need a proxy to access the Internet, You can add a proxy here."));
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
                } else if ("echo %errorlevel%".equals(nextLine)) {
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

    public static JSONObject getUpdateInfo() throws IOException, InterruptedException {
        DownloadUtil downloadUtil = DownloadUtil.getInstance();
        int downloadStatus = downloadUtil.getDownloadStatus("version.json");
        if (downloadStatus != DownloadManager.DOWNLOAD_DOWNLOADING) {
            downloadUtil.downLoadFromUrl("https://raw.githubusercontent.com/XUANXUQAQ/File-Engine-Version/master/version.json",
                    "version.json", "tmp");
            int count = 0;
            boolean isError = false;
            //wait for task
            while (downloadUtil.getDownloadStatus("version.json") != DownloadManager.DOWNLOAD_DONE) {
                count++;
                if (count >= 3) {
                    isError = true;
                    break;
                }
                Thread.sleep(1000);
            }
            if (isError) {
                return null;
            }
            String eachLine;
            StringBuilder strBuilder = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("tmp/version.json"), StandardCharsets.UTF_8))) {
                while ((eachLine = br.readLine()) != null) {
                    strBuilder.append(eachLine);
                }
            }
            return JSONObject.parseObject(strBuilder.toString());
        }
        return null;
    }

    private void readAllSettings() {
        try (BufferedReader buffR = new BufferedReader(new InputStreamReader(new FileInputStream("user/settings.json"), StandardCharsets.UTF_8))) {
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
                if (hotkey == null) {
                    hotkey = "Ctrl + Alt + K";
                }
            } else {
                hotkey = "Ctrl + Alt + K";
            }
            if (settingsInJson.containsKey("priorityFolder")) {
                priorityFolder = settingsInJson.getString("priorityFolder");
                if (priorityFolder == null) {
                    priorityFolder = "";
                }
            } else {
                priorityFolder = "";
            }
            if (settingsInJson.containsKey("searchDepth")) {
                searchDepth = settingsInJson.getInteger("searchDepth");
            } else {
                searchDepth = 8;
            }
            if (settingsInJson.containsKey("ignorePath")) {
                ignorePath = settingsInJson.getString("ignorePath");
                if (ignorePath == null) {
                    ignorePath = "C:\\Windows,";
                }
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
            tmp_openLastFolderKeyCode = openLastFolderKeyCode;
            if (settingsInJson.containsKey("runAsAdminKeyCode")) {
                runAsAdminKeyCode = settingsInJson.getInteger("runAsAdminKeyCode");
            } else {
                runAsAdminKeyCode = 16;
            }
            tmp_runAsAdminKeyCode = runAsAdminKeyCode;
            if (settingsInJson.containsKey("copyPathKeyCode")) {
                copyPathKeyCode = settingsInJson.getInteger("copyPathKeyCode");
            } else {
                copyPathKeyCode = 18;
            }
            tmp_copyPathKeyCode = copyPathKeyCode;
            if (settingsInJson.containsKey("transparency")) {
                transparency = settingsInJson.getFloat("transparency");
            } else {
                transparency = 0.8f;
            }
            if (settingsInJson.containsKey("searchBarColor")) {
                searchBarColor = settingsInJson.getInteger("searchBarColor");
            } else {
                searchBarColor = 0xffffff;
            }
            if (settingsInJson.containsKey("defaultBackground")) {
                defaultBackgroundColor = settingsInJson.getInteger("defaultBackground");
            } else {
                defaultBackgroundColor = 0xffffff;
            }
            if (settingsInJson.containsKey("fontColorWithCoverage")) {
                fontColorWithCoverage = settingsInJson.getInteger("fontColorWithCoverage");
            } else {
                fontColorWithCoverage = 0x6666ff;
            }
            if (settingsInJson.containsKey("labelColor")) {
                labelColor = settingsInJson.getInteger("labelColor");
            } else {
                labelColor = 0xcccccc;
            }
            if (settingsInJson.containsKey("fontColor")) {
                fontColor = settingsInJson.getInteger("fontColor");
            } else {
                fontColor = 0;
            }
            if (settingsInJson.containsKey("language")) {
                String language = settingsInJson.getString("language");
                if (language == null || language.isEmpty()) {
                    language = TranslateUtil.getInstance().getDefaultLang();
                }
                TranslateUtil.getInstance().setLanguage(language);
            } else {
                TranslateUtil.getInstance().setLanguage(TranslateUtil.getInstance().getDefaultLang());
            }
            if (settingsInJson.containsKey("proxyAddress")) {
                proxyAddress = settingsInJson.getString("proxyAddress");
            } else {
                proxyAddress = "";
            }
            if (settingsInJson.containsKey("proxyPort")) {
                proxyPort = settingsInJson.getInteger("proxyPort");
            } else {
                proxyPort = 0;
            }
            if (settingsInJson.containsKey("proxyUserName")) {
                proxyUserName = settingsInJson.getString("proxyUserName");
            } else {
                proxyUserName = "";
            }
            if (settingsInJson.containsKey("proxyPassword")) {
                proxyPassword = settingsInJson.getString("proxyPassword");
            } else {
                proxyPassword = "";
            }
            if (settingsInJson.containsKey("proxyType")) {
                proxyType = settingsInJson.getInteger("proxyType");
            } else {
                proxyType = PROXY_DIRECT;
            }
        } catch (NullPointerException | IOException e) {
            isStartup = false;
            cacheNumLimit = 1000;
            hotkey = "Ctrl + Alt + K";
            priorityFolder = "";
            searchDepth = 8;
            ignorePath = "C:\\Windows,";
            updateTimeLimit = 5;
            isDefaultAdmin = false;
            isLoseFocusClose = true;
            openLastFolderKeyCode = 17;
            runAsAdminKeyCode = 16;
            copyPathKeyCode = 18;
            transparency = 0.8f;
            searchBarColor = 0xffffff;
            defaultBackgroundColor = 0xffffff;
            fontColorWithCoverage = 0x6666ff;
            labelColor = 0xcccccc;
            fontColor = 0;
            TranslateUtil.getInstance().setLanguage(TranslateUtil.getInstance().getDefaultLang());
            tmp_openLastFolderKeyCode = openLastFolderKeyCode;
            tmp_runAsAdminKeyCode = runAsAdminKeyCode;
            tmp_copyPathKeyCode = copyPathKeyCode;
            proxyAddress = "";
            proxyPort = 0;
            proxyUserName = "";
            proxyPassword = "";
            proxyType = PROXY_DIRECT;
        }
    }

    private void setAllSettings() {
        hotKeyListener.registerHotkey(hotkey);
        searchBar.setTransparency(transparency);
        searchBar.setDefaultBackgroundColor(defaultBackgroundColor);
        searchBar.setLabelColor(labelColor);
        searchBar.setFontColorWithCoverage(fontColorWithCoverage);
        searchBar.setFontColor(fontColor);
        searchBar.setSearchBarColor(searchBarColor);

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
        allSettings.put("language", TranslateUtil.getInstance().getLanguage());
        allSettings.put("proxyAddress", proxyAddress);
        allSettings.put("proxyPort", proxyPort);
        allSettings.put("proxyUserName", proxyUserName);
        allSettings.put("proxyPassword", proxyPassword);
        allSettings.put("proxyType", proxyType);
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(settings), StandardCharsets.UTF_8))) {
            String format = JSON.toJSONString(allSettings, SerializerFeature.PrettyFormat, SerializerFeature.WriteMapNullValue, SerializerFeature.WriteDateUseDateFormat);
            buffW.write(format);
        } catch (IOException ignored) {

        }
    }

    private void initCmdSetSettings() {
        //获取所有自定义命令
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("user/cmds.txt"), StandardCharsets.UTF_8))) {
            String each;
            while ((each = br.readLine()) != null) {
                cmdSet.add(each);
            }
        } catch (IOException ignored) {
        }
    }

    protected void hideFrame() {
        frame.setVisible(false);
    }


    protected boolean isSettingsVisible() {
        return frame.isVisible();
    }

    protected void showWindow() {
        initGUI();
        frame.setResizable(true);
        int width = Integer.parseInt(TranslateUtil.getInstance().getFrameWidth());
        int height = Integer.parseInt(TranslateUtil.getInstance().getFrameHeight());

        panel.setSize(width, height);
        frame.setSize(width, height);
        frame.setContentPane(SettingsFrameBuilder.instance.panel);
        frame.setIconImage(frameIcon.getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.setResizable(false);

        tabbedPane.setSelectedIndex(0);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }


    private void saveChanges() {
        StringBuilder strBuilder = new StringBuilder();
        JSONObject allSettings = new JSONObject();
        int updateTimeLimitTemp;
        int cacheNumLimitTemp;
        String ignorePathTemp;
        int searchDepthTemp;
        float transparencyTemp;

        try {
            updateTimeLimitTemp = Integer.parseInt(textFieldUpdateInterval.getText());
        } catch (Exception e1) {
            updateTimeLimitTemp = -1; // 输入不正确
        }
        if (updateTimeLimitTemp > 3600 || updateTimeLimitTemp <= 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("The file index update setting is wrong, please change")).append("\n");
        }
        boolean tmp_isStartup = checkBoxAddToStartup.isSelected();
        try {
            cacheNumLimitTemp = Integer.parseInt(textFieldCacheNum.getText());
        } catch (Exception e1) {
            cacheNumLimitTemp = -1;
        }
        if (cacheNumLimitTemp > 10000 || cacheNumLimitTemp <= 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("The cache capacity is set incorrectly, please change")).append("\n");
        }
        ignorePathTemp = textAreaIgnorePath.getText();
        ignorePathTemp = ignorePathTemp.replaceAll("\n", "");

        try {
            searchDepthTemp = Integer.parseInt(textFieldSearchDepth.getText());
        } catch (Exception e1) {
            searchDepthTemp = -1;
        }

        if (searchDepthTemp > 10 || searchDepthTemp <= 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Search depth setting is wrong, please change")).append("\n");
        }

        String tmp_hotkey = textFieldHotkey.getText();
        if (tmp_hotkey.length() == 1) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Hotkey setting is wrong, please change")).append("\n");
        } else {
            if (!CheckHotKeyUtil.getInstance().isHotkeyAvailable(tmp_hotkey)) {
                strBuilder.append(TranslateUtil.getInstance().getTranslation("Hotkey setting is wrong, please change")).append("\n");
            }
        }
        try {
            transparencyTemp = Float.parseFloat(textFieldTransparency.getText());
        } catch (Exception e) {
            transparencyTemp = -1f;
        }
        if (transparencyTemp > 1 || transparencyTemp <= 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Transparency setting error")).append("\n");
        }

        if (tmp_openLastFolderKeyCode == tmp_runAsAdminKeyCode || tmp_openLastFolderKeyCode == tmp_copyPathKeyCode || tmp_runAsAdminKeyCode == tmp_copyPathKeyCode) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("HotKey conflict")).append("\n");
        }

        int tmp_labelColor;
        try {
            tmp_labelColor = Integer.parseInt(textFieldLabelColor.getText(), 16);
        } catch (Exception e) {
            tmp_labelColor = -1;
        }
        if (tmp_labelColor < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Chosen label color is set incorrectly")).append("\n");
        }

        int tmp_fontColorWithCoverage;
        try {
            tmp_fontColorWithCoverage = Integer.parseInt(textFieldFontColorWithCoverage.getText(), 16);
        } catch (Exception e) {
            tmp_fontColorWithCoverage = -1;
        }
        if (tmp_fontColorWithCoverage < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Chosen label font color is set incorrectly")).append("\n");
        }

        int tmp_defaultBackgroundColor;
        try {
            tmp_defaultBackgroundColor = Integer.parseInt(textFieldBackgroundDefault.getText(), 16);
        } catch (Exception e) {
            tmp_defaultBackgroundColor = -1;
        }
        if (tmp_defaultBackgroundColor < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Incorrect default background color setting")).append("\n");
        }

        int tmp_fontColor;
        try {
            tmp_fontColor = Integer.parseInt(textFieldFontColor.getText(), 16);
        } catch (Exception e) {
            tmp_fontColor = -1;
        }
        if (tmp_fontColor < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Unchosen label font color is set incorrectly")).append("\n");
        }

        int tmp_searchBarColor;
        try {
            tmp_searchBarColor = Integer.parseInt(textFieldSearchBarColor.getText(), 16);
        } catch (Exception e) {
            tmp_searchBarColor = -1;
        }
        if (tmp_searchBarColor < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("The color of the search bar is set incorrectly")).append("\n");
        }
        if (!listLanguage.getSelectedValue().equals(TranslateUtil.getInstance().getLanguage())) {
            TranslateUtil.getInstance().setLanguage((String) listLanguage.getSelectedValue());
            translate();
        }

        String tmp_proxyAddress = textFieldAddress.getText();
        String tmp_proxyUserName = textFieldUserName.getText();
        String tmp_proxyPassword = textFieldPassword.getText();

        int tmp_proxyPort = 0;
        try {
            tmp_proxyPort = Integer.parseInt(textFieldPort.getText());
        } catch (Exception e) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Proxy port is set incorrectly.")).append("\n");
        }

        String errors = strBuilder.toString();
        if (!errors.isEmpty()) {
            JOptionPane.showMessageDialog(frame, errors);
            return;
        }

        if (radioButtonProxyTypeSocks5.isSelected()) {
            proxyType = PROXY_SOCKS;
        } else if (radioButtonProxyTypeHttp.isSelected()) {
            proxyType = PROXY_HTTP;
        }
        if (radioButtonNoProxy.isSelected()) {
            proxyType = PROXY_DIRECT;
        }

        priorityFolder = textFieldPriorityFolder.getText();
        hotkey = textFieldHotkey.getText();
        hotKeyListener.changeHotKey(hotkey);
        cacheNumLimit = cacheNumLimitTemp;
        updateTimeLimit = updateTimeLimitTemp;
        ignorePath = ignorePathTemp;
        searchDepth = searchDepthTemp;
        transparency = transparencyTemp;
        labelColor = tmp_labelColor;
        defaultBackgroundColor = tmp_defaultBackgroundColor;
        searchBarColor = tmp_searchBarColor;
        fontColorWithCoverage = tmp_fontColorWithCoverage;
        fontColor = tmp_fontColor;
        SearchBar instance = SearchBar.getInstance();
        instance.setTransparency(transparency);
        instance.setDefaultBackgroundColor(defaultBackgroundColor);
        instance.setLabelColor(labelColor);
        instance.setFontColorWithCoverage(fontColorWithCoverage);
        instance.setFontColor(fontColor);
        instance.setSearchBarColor(searchBarColor);
        Color tmp_color = new Color(labelColor);
        labelColorChooser.setBackground(tmp_color);
        labelColorChooser.setForeground(tmp_color);
        tmp_color = new Color(defaultBackgroundColor);
        defaultBackgroundChooser.setBackground(tmp_color);
        defaultBackgroundChooser.setForeground(tmp_color);
        tmp_color = new Color(fontColorWithCoverage);
        FontColorWithCoverageChooser.setBackground(tmp_color);
        FontColorWithCoverageChooser.setForeground(tmp_color);
        tmp_color = new Color(fontColor);
        FontColorChooser.setBackground(tmp_color);
        FontColorChooser.setForeground(tmp_color);
        proxyAddress = tmp_proxyAddress;
        proxyPort = tmp_proxyPort;
        proxyUserName = tmp_proxyUserName;
        proxyPassword = tmp_proxyPassword;
        isStartup = tmp_isStartup;
        openLastFolderKeyCode = tmp_openLastFolderKeyCode;
        runAsAdminKeyCode = tmp_runAsAdminKeyCode;
        copyPathKeyCode = tmp_copyPathKeyCode;

        //保存设置
        allSettings.put("hotkey", hotkey);
        allSettings.put("isStartup", isStartup);
        allSettings.put("cacheNumLimit", cacheNumLimit);
        allSettings.put("updateTimeLimit", updateTimeLimit);
        allSettings.put("ignorePath", ignorePath);
        allSettings.put("searchDepth", searchDepth);
        allSettings.put("priorityFolder", priorityFolder);
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
        allSettings.put("language", TranslateUtil.getInstance().getLanguage());
        allSettings.put("proxyAddress", proxyAddress);
        allSettings.put("proxyPort", proxyPort);
        allSettings.put("proxyUserName", proxyUserName);
        allSettings.put("proxyPassword", proxyPassword);
        allSettings.put("proxyType", proxyType);
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(settings), StandardCharsets.UTF_8))) {
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
        hideFrame();
    }

    public static boolean isDebug() {
        try {
            String res = System.getProperty("File_Engine_Debug");
            return "true".equalsIgnoreCase(res);
        } catch (NullPointerException e) {
            return false;
        }
    }

    private void setStartup(boolean b) {
        if (b) {
            String command = "cmd.exe /c schtasks /create /ru \"administrators\" /rl HIGHEST /sc ONLOGON /tn \"File-Engine\" /tr ";
            File FileEngine = new File(getName());
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
                    checkBoxAddToStartup.setSelected(false);
                    JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Add to startup failed, please try to run as administrator"));
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
                    checkBoxAddToStartup.setSelected(true);
                    JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Delete startup failed, please try to run as administrator"));
                }
            } catch (IOException | InterruptedException ignored) {

            }
        }

        boolean isSelected = checkBoxAddToStartup.isSelected();
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
