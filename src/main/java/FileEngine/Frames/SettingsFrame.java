package FileEngine.Frames;

import FileEngine.Download.DownloadManager;
import FileEngine.Download.DownloadUtil;
import FileEngine.HotkeyListener.CheckHotKey;
import FileEngine.PluginSystem.Plugin;
import FileEngine.PluginSystem.PluginUtil;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SerializerFeature;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.URI;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Locale;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;


public class SettingsFrame {
    public static final String version = "2.2"; //TODO 更改版本号
    private static volatile boolean mainExit = false;
    private static boolean is64Bit;
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
    private static File tmp;
    private static File settings;
    private static HashSet<String> cmdSet;
    private static HashSet<String> languageSet;
    private static ConcurrentHashMap<String, String> translationMap;
    private static ConcurrentHashMap<String, String> fileMap;
    private static Pattern equalSign;
    private static volatile int tmp_openLastFolderKeyCode;
    private static volatile int tmp_runAsAdminKeyCode;
    private static CheckHotKey HotKeyListener;
    private static volatile int labelColor;
    private static volatile int defaultBackgroundColor;
    private static volatile int fontColorWithCoverage;
    private static volatile int fontColor;
    private static volatile int searchBarColor;
    private volatile static String language;
    private static volatile boolean isStartup;
    private static SearchBar searchBar;
    private final ExecutorService threadPool;
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


    private static class SettingsFrameBuilder {
        private static final SettingsFrame instance = new SettingsFrame();
    }

    public static void set64Bit(boolean b) {
        is64Bit = b;
    }

    public static boolean is64Bit() {
        return is64Bit;
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
        if (is64Bit) {
            return "File-Engine-x64.exe";
        } else {
            return "File-Engine-x86.exe";
        }
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
                JOptionPane.showMessageDialog(frame, SettingsFrame.getTranslation("The program is detected on the desktop and cannot be moved"));
                return;
            }
            int isConfirmed = JOptionPane.showConfirmDialog(frame, SettingsFrame.getTranslation("Whether to remove and backup all files on the desktop," +
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
            int returnValue = fileChooser.showDialog(new JLabel(), SettingsFrame.getTranslation("Choose"));
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
            int returnValue = fileChooser.showDialog(new JLabel(), SettingsFrame.getTranslation("Choose"));
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
            String name = JOptionPane.showInputDialog(SettingsFrame.getTranslation("Please enter the ID of the command, then you can enter \": identifier\" in the search box to execute the command directly"));
            if (name == null || name.isEmpty()) {
                //未输入
                return;
            }
            if ("update".equals(name) || "clearbin".equals(name) || "help".equals(name) || "version".equals(name) || isRepeatCommand(name)) {
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
            boolean isDownloading = false;
            try {
                isDownloading = DownloadUtil.getInstance().hasTask(getName());
            } catch (Exception ignored) {
            }
            //检查是否已开始下载
            if (!isDownloading) {
                JSONObject updateInfo;
                String latestVersion;
                try {
                    updateInfo = getInfo();
                    latestVersion = updateInfo.getString("version");
                } catch (IOException e1) {
                    JOptionPane.showMessageDialog(frame, getTranslation("Check update failed"));
                    return;
                }

                if (Double.parseDouble(latestVersion) > Double.parseDouble(version) || isDebug()) {
                    String description = updateInfo.getString("description");
                    int result = JOptionPane.showConfirmDialog(frame,
                            getTranslation("New Version available") + latestVersion + "," + getTranslation("Whether to update") + "\n" + getTranslation("update content") + "\n" + description);
                    if (result == 0) {
                        //开始更新,下载更新文件到tmp
                        String urlChoose;
                        String fileName;
                        if (is64Bit) {
                            urlChoose = "url64";
                        } else {
                            urlChoose = "url86";
                        }
                        fileName = getName();
                        DownloadUtil download = DownloadUtil.getInstance();
                        download.downLoadFromUrl(updateInfo.getString(urlChoose), fileName, tmp.getAbsolutePath());
                        //更新button为取消
                        buttonCheckUpdate.setText(getTranslation("Cancel"));
                    }
                } else {
                    JOptionPane.showMessageDialog(frame,
                            getTranslation("Latest version:") + latestVersion + "\n" +
                                    getTranslation("The current version is the latest"));
                }
            } else {
                //取消下载
                String fileName = getName();
                DownloadUtil instance = DownloadUtil.getInstance();
                instance.cancelDownload(fileName);
                threadPool.execute(() -> {
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
                    buttonCheckUpdate.setText(getTranslation("Check for update"));
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
            textFieldFontColorWithCoverage.setText(Integer.toHexString(0x1C0EFF));
            textFieldSearchBarColor.setText(Integer.toHexString(0xffffff));
            textFieldLabelColor.setText(Integer.toHexString(0xFF9868));
            textFieldBackgroundDefault.setText(Integer.toHexString(0xffffff));
            textFieldFontColor.setText(Integer.toHexString(0x333333));
        });
    }

    private void addColorChooserLabelListener() {
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
                    icon = plugin.getPluginIcon();
                    description = plugin.getDescription();
                    officialSite = plugin.getOfficialSite();
                    version = plugin.getVersion();
                    author = plugin.getAuthor();

                    labelPluginVersion.setText(getTranslation("Version") + ":" + version);
                    PluginIconLabel.setIcon(icon);
                    PluginNamelabel.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                    textAreaDescription.setText(description);
                    labelAuthor.setText(getTranslation("Author") + ":" + author);
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
                JOptionPane.showMessageDialog(frame, getTranslation("The current Version is the latest."));
            } else {
                //检查是否已经开始下载
                boolean isPluginUpdating = false;
                try {
                    isPluginUpdating = DownloadUtil.getInstance().hasTask(pluginFullName);
                } catch (Exception ignored) {
                }
                if (!isPluginUpdating) {
                    int ret = JOptionPane.showConfirmDialog(frame, getTranslation("New version available, do you want to update?"));
                    if (ret == 0) {
                        //开始下载
                        String url = plugin.getUpdateURL();
                        DownloadUtil.getInstance().downLoadFromUrl(url, pluginFullName, "tmp/pluginsUpdate");
                        buttonUpdatePlugin.setText(getTranslation("Cancel"));
                    }
                } else {
                    //取消下载
                    DownloadUtil instance = DownloadUtil.getInstance();
                    instance.cancelDownload(pluginFullName);
                    threadPool.execute(() -> {
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
                            buttonUpdatePlugin.setText(getTranslation("Install"));
                            buttonUpdatePlugin.setEnabled(true);
                        } catch (InterruptedException ignored) {
                        }
                    });
                }
            }
        });
    }

    private void initAll() {
        //设置窗口显示
        labelAboutGithub.setText("<html><a href='https://github.com/XUANXUQAQ/File-Engine'><font size=\"4\">File-Engine</font></a></html>");
        labelWebLookAndFeel.setText("1.WebLookAndFeel");
        labelFastJson.setText("2.FastJson");
        labelJna.setText("3.Java-Native-Access");
        labelSQLite.setText("4.SQLite-JDBC");
        labelPluginNum.setText(String.valueOf(PluginUtil.getInstalledPluginNum()));
        ImageIcon imageIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        labelIcon.setIcon(imageIcon);
        labelVersion.setText(getTranslation("Current Version:") + version);
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
        listLanguage.setListData(languageSet.toArray());
        listLanguage.setSelectedValue(language, true);
        Object[] plugins = PluginUtil.getPluginArray();
        listPlugins.setListData(plugins);
        buttonUpdatePlugin.setVisible(false);
        if (plugins.length == 0) {
            PluginSettingsPanel.setVisible(false);
        }
    }

    private SettingsFrame() {
        frame = new JFrame("Settings");
        tmp = new File("tmp");
        settings = new File("user/settings.json");
        cmdSet = new HashSet<>();
        equalSign = Pattern.compile("=");
        fileMap = new ConcurrentHashMap<>();
        translationMap = new ConcurrentHashMap<>();
        languageSet = new HashSet<>();
        frameIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        threadPool = Executors.newCachedThreadPool();
        readAllSettings();

        HotKeyListener = CheckHotKey.getInstance();
        searchBar = SearchBar.getInstance();

        setAllSettings();

        initLanguageFileMap();
        translate(language);

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

        initAll();

        initThreadPool();
    }

    private void checkDownloadTask(JLabel label, JButton button, String fileName, String originButtonString) throws InterruptedException {
        //设置进度显示线程
        double progress;
        if (DownloadUtil.getInstance().hasTask(fileName)) {
            progress = DownloadUtil.getInstance().getDownloadProgress(fileName);
            label.setText(getTranslation("Downloading:") + progress * 100 + "%");

            int downloadingStatus = DownloadUtil.getInstance().getDownloadStatus(fileName);
            if (downloadingStatus == DownloadManager.DOWNLOAD_DONE) {
                //下载完成，禁用按钮
                label.setText(getTranslation("Download Done"));
                label.setText(getTranslation("Downloaded"));
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
                label.setText(getTranslation("Download failed"));
                button.setText(getTranslation(originButtonString));
                button.setEnabled(true);
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_DOWNLOADING) {
                //正在下载
                button.setText(SettingsFrame.getTranslation("Cancel"));
            } else if (downloadingStatus == DownloadManager.DOWNLOAD_INTERRUPTED) {
                //用户自行中断
                label.setText("");
                button.setText(getTranslation(originButtonString));
                button.setEnabled(true);
            }
        } else {
            label.setText("");
            button.setText(getTranslation(originButtonString));
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
        threadPool.execute(() -> addShowDownloadProgressTask(labelDownloadProgress, buttonCheckUpdate, getName()));

        threadPool.execute(() -> {
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

    private void translate(String language) {
        initTranslations(language);
        tabbedPane.setTitleAt(0, getTranslation("General"));
        tabbedPane.setTitleAt(1, getTranslation("Search settings"));
        tabbedPane.setTitleAt(2, getTranslation("Search bar settings"));
        tabbedPane.setTitleAt(3, getTranslation("Plugins"));
        tabbedPane.setTitleAt(4, getTranslation("Hotkey settings"));
        tabbedPane.setTitleAt(5, getTranslation("Language settings"));
        tabbedPane.setTitleAt(6, getTranslation("My commands"));
        tabbedPane.setTitleAt(7, getTranslation("Color settings"));
        tabbedPane.setTitleAt(8, getTranslation("About"));
        checkBoxAddToStartup.setText(getTranslation("Add to startup"));
        buttonSaveAndRemoveDesktop.setText(getTranslation("Backup and remove all desktop files"));
        labelMaxCacheNum.setText(getTranslation("Set the maximum number of caches:"));
        ButtonPriorityFolder.setText(getTranslation("Choose"));
        buttonChooseFile.setText(getTranslation("Choose"));
        labelUpdateInterval.setText(getTranslation("File update detection interval:"));
        labelSecond.setText(getTranslation("Seconds"));
        labelSearchDepth.setText(getTranslation("Search depth (too large may affect performance):"));
        labeltipPriorityFolder.setText(getTranslation("Priority search folder location (double-click to clear):"));
        labelConstIgnorePathTip.setText(getTranslation("Separate different paths with commas, and ignore C:\\Windows by default"));
        labelSetIgnorePathTip.setText(getTranslation("Set ignore folder:"));
        checkBoxLoseFocus.setText(getTranslation("Close search bar when focus lost"));
        checkBoxAdmin.setText(getTranslation("Open other programs as an administrator (provided that the software has privileges)"));
        labelTransparency.setText(getTranslation("Search bar transparency:"));
        labelOpenSearchBarHotKey.setText(getTranslation("Open search bar:"));
        labelRunAsAdminHotKey.setText(getTranslation("Run as administrator:"));
        labelOpenFolderHotKey.setText(getTranslation("Open the parent folder:"));
        labelCopyPathHotKey.setText(getTranslation("Copy path:"));
        labelCmdTip2.setText(getTranslation("You can add custom commands here. After adding, you can enter \": + your set identifier\" in the search box to quickly open"));
        buttonAddCMD.setText(getTranslation("Add"));
        buttonDelCmd.setText(getTranslation("Delete"));
        labelColorTip.setText(getTranslation("Please enter the hexadecimal value of RGB color"));
        labelSearchBarColor.setText(getTranslation("Search bar Color:"));
        labelLabelColor.setText(getTranslation("Chosen label color:"));
        labelFontColor.setText(getTranslation("Chosen label font Color:"));
        labelDefaultColor.setText(getTranslation("Default background Color:"));
        labelNotChosenFontColor.setText(getTranslation("Unchosen label Color:"));
        buttonResetColor.setText(getTranslation("Reset to default"));
        labelGitHubTip.setText(getTranslation("This is an open source software,GitHub:"));
        labelGithubIssue.setText(getTranslation("If you find a bug or have some suggestions, welcome to GitHub for feedback"));
        buttonCheckUpdate.setText(getTranslation("Check for update"));
        labelDescription.setText(getTranslation("Thanks for the following project"));
        buttonSave.setText(getTranslation("Save"));
        labelTranslationTip.setText(getTranslation("The translation might not be 100% accurate"));
        labelLanguageChooseTip.setText(getTranslation("Choose a language"));
        labelVersion.setText(getTranslation("Current Version:") + version);
        labelInstalledPluginNum.setText(getTranslation("Installed plugins num:"));
        buttonUpdatePlugin.setText(getTranslation("Check for update"));
        buttonPluginMarket.setText(getTranslation("Plugin Market"));
    }

    private static void initLanguageSet() {
        //TODO 添加语言
        languageSet.add("简体中文");
        languageSet.add("English(US)");
        languageSet.add("日本語");
        languageSet.add("繁體中文");
    }

    private static void initTranslations(String language) {
        if (!"English(US)".equals(language)) {
            String filePath = fileMap.get(language);
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(SettingsFrame.class.getResourceAsStream(filePath), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] record = equalSign.split(line);
                    translationMap.put(record[0].trim(), record[1].trim());
                }
            } catch (IOException ignored) {

            }
        } else {
            translationMap.put("#frame_width", String.valueOf(1000));
            translationMap.put("#frame_height", String.valueOf(600));
        }
    }

    private static void initLanguageFileMap() {
        //TODO 添加语言
        fileMap.put("简体中文", "/language/Chinese(Simplified).txt");
        fileMap.put("日本語", "/language/Japanese.txt");
        fileMap.put("繁體中文", "/language/Chinese(Traditional).txt");
    }

    private static String getDefaultLang() {
        //TODO 添加语言
        Locale l = Locale.getDefault();
        String lang = l.toLanguageTag();
        switch (lang) {
            case "zh-CN":
                return "简体中文";
            case "ja-JP":
            case "ja-JP-u-ca-japanese":
            case "ja-JP-x-lvariant-JP":
                return "日本語";
            case "zh-HK":
            case "zh-TW":
                return "繁體中文";
            default:
                return "English(US)";
        }
    }

    public static String getTranslation(String text) {
        String translated;
        if ("English(US)".equals(language)) {
            translated = text;
        } else {
            translated = translationMap.get(text);
        }
        if (translated != null) {
            return translated;
        } else {
            return text;
        }
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
                fontColor = 0x333333;
            }
            if (settingsInJson.containsKey("language")) {
                language = settingsInJson.getString("language");
                if (language == null) {
                    language = getDefaultLang();
                }
            } else {
                language = getDefaultLang();
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
            fontColorWithCoverage = 0x1C0EFF;
            labelColor = 0xFF9868;
            fontColor = 0x333333;
            language = getDefaultLang();
            tmp_openLastFolderKeyCode = openLastFolderKeyCode;
            tmp_runAsAdminKeyCode = runAsAdminKeyCode;
            tmp_copyPathKeyCode = copyPathKeyCode;
        }
    }

    private static void setAllSettings() {
        HotKeyListener.registerHotkey(hotkey);
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

    public void hideFrame() {
        frame.setVisible(false);
    }


    public boolean isSettingsVisible() {
        return frame.isVisible();
    }

    public void showWindow() {
        frame.setResizable(true);
        int width = Integer.parseInt(translationMap.get("#frame_width"));
        int height = Integer.parseInt(translationMap.get("#frame_height"));

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
            JOptionPane.showMessageDialog(frame, getTranslation("The file index update setting is wrong, please change"));
            return;
        }
        isStartup = checkBoxAddToStartup.isSelected();
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

        try {
            searchDepthTemp = Integer.parseInt(textFieldSearchDepth.getText());
        } catch (Exception e1) {
            searchDepthTemp = -1;
        }

        if (searchDepthTemp > 10 || searchDepthTemp <= 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("Search depth setting is wrong, please change"));
            return;
        }

        String tmp_hotkey = textFieldHotkey.getText();
        if (tmp_hotkey.length() == 1) {
            JOptionPane.showMessageDialog(frame, getTranslation("Hotkey setting is wrong, please change"));
            return;
        } else {
            if (!CheckHotKey.getInstance().isHotkeyAvailable(tmp_hotkey)) {
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

        if (tmp_openLastFolderKeyCode == tmp_runAsAdminKeyCode || tmp_openLastFolderKeyCode == tmp_copyPathKeyCode || tmp_runAsAdminKeyCode == tmp_copyPathKeyCode) {
            JOptionPane.showMessageDialog(frame, getTranslation("HotKey conflict"));
            return;
        } else {
            openLastFolderKeyCode = tmp_openLastFolderKeyCode;
            runAsAdminKeyCode = tmp_runAsAdminKeyCode;
            copyPathKeyCode = tmp_copyPathKeyCode;
        }

        int tmp_labelColor;
        try {
            tmp_labelColor = Integer.parseInt(textFieldLabelColor.getText(), 16);
        } catch (Exception e) {
            tmp_labelColor = -1;
        }
        if (tmp_labelColor < 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("Chosen label color is set incorrectly"));
            return;
        }


        int tmp_fontColorWithCoverage;
        try {
            tmp_fontColorWithCoverage = Integer.parseInt(textFieldFontColorWithCoverage.getText(), 16);
        } catch (Exception e) {
            tmp_fontColorWithCoverage = -1;
        }
        if (tmp_fontColorWithCoverage < 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("Chosen label font color is set incorrectly"));
            return;
        }


        int tmp_defaultBackgroundColor;
        try {
            tmp_defaultBackgroundColor = Integer.parseInt(textFieldBackgroundDefault.getText(), 16);
        } catch (Exception e) {
            tmp_defaultBackgroundColor = -1;
        }
        if (tmp_defaultBackgroundColor < 0) {
            JOptionPane.showMessageDialog(frame, getTranslation("Incorrect default background color setting"));
            return;
        }


        int tmp_fontColor;
        try {
            tmp_fontColor = Integer.parseInt(textFieldFontColor.getText(), 16);
        } catch (Exception e) {
            tmp_fontColor = -1;
        }
        if (tmp_fontColor < 0) {
            JOptionPane.showMessageDialog(frame, "Unchosen label font color is set incorrectly");
            return;
        }


        int tmp_searchBarColor;
        try {
            tmp_searchBarColor = Integer.parseInt(textFieldSearchBarColor.getText(), 16);
        } catch (Exception e) {
            tmp_searchBarColor = -1;
        }
        if (tmp_searchBarColor < 0) {
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
        allSettings.put("language", language);
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
                    checkBoxAddToStartup.setSelected(true);
                    JOptionPane.showMessageDialog(frame, getTranslation("Delete startup failed, please try to run as administrator"));
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
