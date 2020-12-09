package FileEngine.frames;

import FileEngine.SQLiteConfig.SQLiteUtil;
import FileEngine.checkHotkey.CheckHotKeyUtil;
import FileEngine.configs.AllConfigs;
import FileEngine.download.DownloadUtil;
import FileEngine.moveFiles.MoveDesktopFiles;
import FileEngine.pluginSystem.Plugin;
import FileEngine.pluginSystem.PluginUtil;
import FileEngine.search.SearchUtil;
import FileEngine.threadPool.CachedThreadPool;
import FileEngine.translate.TranslateUtil;
import com.alibaba.fastjson.JSONObject;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;


public class SettingsFrame {
    private final Set<String> cacheSet = ConcurrentHashMap.newKeySet();
    private static volatile int tmp_copyPathKeyCode;
    private static volatile int tmp_runAsAdminKeyCode;
    private static volatile int tmp_openLastFolderKeyCode;
    private static volatile boolean isStartupChanged = false;
    private static volatile boolean isUpdateButtonPluginString = false;
    private static ImageIcon frameIcon;
    private final JFrame frame = new JFrame("Settings");
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
    private JLabel labelPlaceHolder10;
    private JLabel placeHolder1;
    private JLabel placeHolder2;
    private JLabel placeHolder4;
    private JLabel placeHolder6;
    private JLabel placeHolder7;
    private JComboBox<String> chooseUpdateAddress;
    private JLabel chooseUpdateAddressLabel;
    private JLabel labelPlaceHolderWhatever2;
    private JPanel tabCache;
    private JLabel labelCacheSettings;
    private JLabel labelCacheTip;
    private JList<Object> listCache;
    private JLabel labelPlaceHolderWhatever3;
    private JButton buttonDeleteCache;
    private JScrollPane cacheScrollPane;
    private JLabel labelVacuumTip2;
    private JLabel labelCacheTip2;
    private JButton buttonDeleteAllCache;
    private JLabel labelSearchBarFontColor;
    private JLabel labelSharp9;
    private JTextField textFieldSearchBarFontColor;
    private JLabel SearchBarFontColorChooser;
    private JLabel labelBorderColor;
    private JLabel labelSharp2;
    private JTextField textFieldBorderColor;
    private JLabel borderColorChooser;
    private JLabel labelCurrentCacheNum;
    private JLabel labelUninstallPluginTip;
    private JLabel labelUninstallPluginTip2;
    private JCheckBox checkBoxIsShowTipOnCreatingLnk;
    private JLabel labelPlaceHolder15;
    private JLabel labelPlaceHolder12;

    private static class SettingsFrameBuilder {
        private static final SettingsFrame instance = new SettingsFrame();
    }

    public static SettingsFrame getInstance() {
        return SettingsFrameBuilder.instance;
    }

    private void addWindowCloseListener() {
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                String errors = saveChanges(false);
                if (!errors.isEmpty()) {
                    int ret = JOptionPane.showConfirmDialog(null,
                            TranslateUtil.getInstance().getTranslation("Errors") + ":\n" + errors + "\n" +
                            TranslateUtil.getInstance().getTranslation("Failed to save settings, do you still close the window"));
                    if (ret == JOptionPane.YES_OPTION) {
                        hideFrame();
                    }
                } else {
                    hideFrame();
                }
            }
        });
    }

    private void addCheckBoxStartupListener() {
        checkBoxAddToStartup.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                isStartupChanged = true;
            }
        });
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
            if (isConfirmed == JOptionPane.YES_OPTION) {
                CachedThreadPool.getInstance().executeTask(MoveDesktopFiles::start);
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
            if ("update".equalsIgnoreCase(name) || "clearbin".equalsIgnoreCase(name) ||
                    "help".equalsIgnoreCase(name) || "version".equalsIgnoreCase(name) || isRepeatCommand(name)) {
                JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Conflict with existing commands"));
                return;
            }
            if (name.length() == 1) {
                int ret = JOptionPane.showConfirmDialog(frame,
                        TranslateUtil.getInstance().getTranslation("The identifier you entered is too short, continue") + "?");
                if (ret != JOptionPane.OK_OPTION) {
                    return;
                }
            }
            String cmd;
            JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Please select the location of the executable file (a folder is also acceptable)"));
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
            int returnValue = fileChooser.showDialog(new Label(), TranslateUtil.getInstance().getTranslation("Choose"));
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                cmd = fileChooser.getSelectedFile().getAbsolutePath();
                AllConfigs.getCmdSet().add(":" + name + ";" + cmd);
                listCmds.setListData(AllConfigs.getCmdSet().toArray());
            }
        });
    }

    private void addButtonDelCMDListener() {
        buttonDelCmd.addActionListener(e -> {
            String del = (String) listCmds.getSelectedValue();
            if (del != null) {
                AllConfigs.getCmdSet().remove(del);
                listCmds.setListData(AllConfigs.getCmdSet().toArray());
            }

        });
    }

    private void addButtonSaveListener() {
        buttonSave.addActionListener(e -> saveChanges(true));
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
            AllConfigs.DownloadStatus status = DownloadUtil.getInstance().getDownloadStatus(AllConfigs.FILE_NAME);
            if (status == AllConfigs.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                //取消下载
                String fileName = AllConfigs.FILE_NAME;
                DownloadUtil instance = DownloadUtil.getInstance();
                instance.cancelDownload(fileName);
                //复位button
                buttonCheckUpdate.setText(TranslateUtil.getInstance().getTranslation("Check for update"));
                buttonCheckUpdate.setEnabled(true);
            } else if (status == AllConfigs.DownloadStatus.DOWNLOAD_DONE) {
                buttonCheckUpdate.setEnabled(false);
            } else {
                //开始下载
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

                if (Double.parseDouble(latestVersion) > Double.parseDouble(AllConfigs.version)) {
                    String description = updateInfo.getString("description");
                    int result = JOptionPane.showConfirmDialog(frame,
                            TranslateUtil.getInstance().getTranslation(
                                    "New Version available") + latestVersion + "," +
                                    TranslateUtil.getInstance().getTranslation("Whether to update") + "\n" +
                                    TranslateUtil.getInstance().getTranslation("update content") + "\n" + description);
                    if (result == JOptionPane.YES_OPTION) {
                        //开始更新,下载更新文件到tmp
                        String urlChoose;
                        String fileName;
                        urlChoose = "url64";
                        fileName = AllConfigs.FILE_NAME;
                        DownloadUtil download = DownloadUtil.getInstance();
                        download.downLoadFromUrl(updateInfo.getString(urlChoose), fileName, AllConfigs.getTmp().getAbsolutePath());
                        //更新button为取消
                        buttonCheckUpdate.setText(TranslateUtil.getInstance().getTranslation("Cancel"));
                    }
                } else {
                    JOptionPane.showMessageDialog(frame,
                            TranslateUtil.getInstance().getTranslation("Latest version:") + latestVersion + "\n" +
                                    TranslateUtil.getInstance().getTranslation("The current version is the latest"));
                }
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
            textFieldFontColorWithCoverage.setText(Integer.toHexString(AllConfigs.defaultFontColorWithCoverage));
            textFieldSearchBarColor.setText(Integer.toHexString(AllConfigs.defaultSearchbarColor));
            textFieldLabelColor.setText(Integer.toHexString(AllConfigs.defaultLabelColor));
            textFieldBackgroundDefault.setText(Integer.toHexString(AllConfigs.defaultWindowBackgroundColor));
            textFieldFontColor.setText(Integer.toHexString(AllConfigs.defaultFontColor));
            textFieldSearchBarFontColor.setText(Integer.toHexString(AllConfigs.defaultSearchbarFontColor));
            textFieldBorderColor.setText(Integer.toHexString(AllConfigs.defaultBorderColor));
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
                textFieldLabelColor.setText(parseColorHex(color));
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
                textFieldFontColorWithCoverage.setText(parseColorHex(color));
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
                textFieldBackgroundDefault.setText(parseColorHex(color));
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
                textFieldFontColor.setText(parseColorHex(color));
                FontColorChooser.setBackground(color);
                FontColorChooser.setForeground(color);
            }
        });

        SearchBarFontColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, TranslateUtil.getInstance().getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldSearchBarFontColor.setText(parseColorHex(color));
                SearchBarFontColorChooser.setBackground(color);
                SearchBarFontColorChooser.setForeground(color);
            }
        });

        borderColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, TranslateUtil.getInstance().getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldBorderColor.setText(parseColorHex(color));
                borderColorChooser.setBackground(color);
                borderColorChooser.setForeground(color);
            }
        });

        searchBarColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(null, TranslateUtil.getInstance().getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldSearchBarColor.setText(parseColorHex(color));
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
                    String pluginIdentifier = PluginUtil.getInstance().getIdentifierByName(pluginName);
                    Plugin plugin = PluginUtil.getInstance().getPluginByIdentifier(pluginIdentifier);
                    int apiVersion;
                    ImageIcon icon = plugin.getPluginIcon();
                    String description = plugin.getDescription();
                    String officialSite = plugin.getOfficialSite();
                    String version = plugin.getVersion();
                    String author = plugin.getAuthor();
                    apiVersion = plugin.getApiVersion();

                    labelPluginVersion.setText(TranslateUtil.getInstance().getTranslation("Version") + ":" + version);
                    labelApiVersion.setText("API " + TranslateUtil.getInstance().getTranslation("Version") + ":" + apiVersion);
                    PluginIconLabel.setIcon(icon);
                    PluginNamelabel.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                    textAreaDescription.setText(description);
                    labelAuthor.setText(TranslateUtil.getInstance().getTranslation("Author") + ":" + author);
                    labelOfficialSite.setText("<html><a href='" + officialSite + "'><font size=\"4\">" + pluginName + "</font></a></html>");
                    buttonUpdatePlugin.setVisible(true);
                    if (PluginUtil.getInstance().isPluginsNotLatest(pluginName)) {
                        isUpdateButtonPluginString = true;
                        Color color = new Color(51,122,183);
                        buttonUpdatePlugin.setText(TranslateUtil.getInstance().getTranslation("Update"));
                        buttonUpdatePlugin.setBackground(color);
                    }
                }
            }
        });
    }

    private String parseColorHex(Color color) {
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
        return rgb.toString();
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

    private void addButtonDeleteCacheListener() {
        buttonDeleteCache.addActionListener(e -> {
            String cache = (String) listCache.getSelectedValue();
            if (cache != null) {
                SearchUtil.getInstance().removeFileFromCache(cache);
                cacheSet.remove(cache);
                AllConfigs.decrementCacheNum();
                listCache.setListData(cacheSet.toArray());
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

    private void clearDatabase() {
        String column;
        for (int i = 0; i <= 40; ++i) {
            column = "list" + i;
            clearDatabase(column);
        }
        SearchUtil.getInstance().executeImmediately();
    }

    private void clearDatabase(String column) {
        File file;
        try(PreparedStatement pStmt = SQLiteUtil.getPreparedStatement(column);
            ResultSet resultSet = pStmt.executeQuery()) {
            while (resultSet.next()) {
                String record = resultSet.getString("PATH");
                file = new File(record);
                if (!file.exists()) {
                    if (AllConfigs.isDebug()) {
                        System.err.println("正在删除" + record);
                    }
                    SearchUtil.getInstance().removeFileFromDatabase(record);
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    private void addButtonVacuumListener() {
        buttonVacuum.addActionListener(e -> {
            int ret = JOptionPane.showConfirmDialog(frame, TranslateUtil.getInstance().getTranslation("Confirm whether to start optimizing the database?"));
            if (JOptionPane.YES_OPTION == ret) {
                int status = SearchUtil.getInstance().getStatus();
                if (status == SearchUtil.NORMAL) {
                    if (AllConfigs.isDebug()) {
                        System.out.println("开始优化");
                    }
                    SearchUtil.getInstance().setStatus(SearchUtil.VACUUM);
                    CachedThreadPool.getInstance().executeTask(() -> {
                        //执行VACUUM命令
                        try (PreparedStatement stmt = SQLiteUtil.getPreparedStatement("VACUUM;")) {
                            clearDatabase();
                            stmt.execute();
                        } catch (Exception ex) {
                            if (AllConfigs.isDebug()) {
                                ex.printStackTrace();
                            }
                        } finally {
                            if (AllConfigs.isDebug()) {
                                System.out.println("结束优化");
                            }
                            SearchUtil.getInstance().setStatus(SearchUtil.NORMAL);
                        }
                    });
                    CachedThreadPool.getInstance().executeTask(() -> {
                        //实时显示VACUUM状态
                        try {
                            SearchUtil instance = SearchUtil.getInstance();
                            while (instance.getStatus() == SearchUtil.VACUUM) {
                                labelVacuumStatus.setText(TranslateUtil.getInstance().getTranslation("Optimizing..."));
                                TimeUnit.MILLISECONDS.sleep(50);
                            }
                            labelVacuumStatus.setText(TranslateUtil.getInstance().getTranslation("Optimized"));
                            TimeUnit.SECONDS.sleep(3);
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

    private void addButtonDeleteAllCacheListener() {
        buttonDeleteAllCache.addActionListener(e -> {
            int ret = JOptionPane.showConfirmDialog(frame,
                    TranslateUtil.getInstance().getTranslation("The operation is irreversible. Are you sure you want to clear the cache?"));
            if (JOptionPane.YES_OPTION == ret) {
                for (String each : cacheSet) {
                    SearchUtil.getInstance().removeFileFromCache(each);
                }
                cacheSet.clear();
                AllConfigs.resetCacheNumToZero();
                listCache.setListData(cacheSet.toArray());
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
        AtomicLong startCheckTime = new AtomicLong(0L);
        AtomicBoolean isVersionLatest = new AtomicBoolean(true);
        AtomicBoolean isSkipConfirm = new AtomicBoolean(false);

        buttonUpdatePlugin.addActionListener(e -> {
            startCheckTime.set(0L);
            String pluginName = (String) listPlugins.getSelectedValue();
            String pluginIdentifier = PluginUtil.getInstance().getIdentifierByName(pluginName);
            Plugin plugin = PluginUtil.getInstance().getPluginByIdentifier(pluginIdentifier);
            String pluginFullName = pluginName + ".jar";
            //检查是否已经开始下载
            AllConfigs.DownloadStatus downloadStatus = DownloadUtil.getInstance().getDownloadStatus(pluginFullName);
            if (downloadStatus == AllConfigs.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                //取消下载
                DownloadUtil instance = DownloadUtil.getInstance();
                instance.cancelDownload(pluginFullName);
                buttonUpdatePlugin.setEnabled(true);
            } else if (downloadStatus == AllConfigs.DownloadStatus.DOWNLOAD_DONE) {
                buttonUpdatePlugin.setEnabled(false);
                PluginUtil.getInstance().removeFromPluginsCanUpdate(pluginName);
            } else {
                if (!PluginUtil.getInstance().isPluginsNotLatest(pluginName)) {
                    Thread checkUpdateThread = new Thread(() -> {
                        startCheckTime.set(System.currentTimeMillis());
                        try {
                            isVersionLatest.set(plugin.isLatest());
                            if (!Thread.interrupted()) {
                                startCheckTime.set(0x100L); //表示检查成功
                            }
                        } catch (Exception exception) {
                            exception.printStackTrace();
                            startCheckTime.set(0xFFFL);  //表示检查失败
                        }
                    });
                    CachedThreadPool.getInstance().executeTask(checkUpdateThread);
                    //等待获取插件更新信息
                    try {
                        while (startCheckTime.get() != 0x100L) {
                            TimeUnit.MILLISECONDS.sleep(200);
                            if ((System.currentTimeMillis() - startCheckTime.get() > 5000L && startCheckTime.get() != 0x100L) || startCheckTime.get() == 0xFFFL) {
                                checkUpdateThread.interrupt();
                                JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Check update failed"));
                                return;
                            }
                            if (!AllConfigs.isNotMainExit()) {
                                return;
                            }
                        }
                    } catch (InterruptedException e1) {
                        e1.printStackTrace();
                    }
                } else {
                    isVersionLatest.set(false);
                    isSkipConfirm.set(true);
                }
                if (isVersionLatest.get()) {
                    JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("The current Version is the latest."));
                } else {
                    if (isSkipConfirm.get()) {
                        //直接开始下载
                        String url = plugin.getUpdateURL();
                        DownloadUtil.getInstance().downLoadFromUrl(url, pluginFullName, "tmp/pluginsUpdate");
                    } else {
                        PluginUtil.getInstance().addPluginsCanUpdate(pluginName);
                        int ret = JOptionPane.showConfirmDialog(frame, TranslateUtil.getInstance().getTranslation("New version available, do you want to update?"));
                        if (ret == JOptionPane.YES_OPTION) {
                            //开始下载
                            String url = plugin.getUpdateURL();
                            DownloadUtil.getInstance().downLoadFromUrl(url, pluginFullName, "tmp/pluginsUpdate");
                        }
                    }
                }
            }
        });
    }

    private void setLabelGui() {
        labelAboutGithub.setText("<html><a href='https://github.com/XUANXUQAQ/File-Engine'><font size=\"4\">File-Engine</font></a></html>");
        labelWebLookAndFeel.setText("1.FlatLaf");
        labelFastJson.setText("2.FastJson");
        labelJna.setText("3.Java-Native-Access");
        labelSQLite.setText("4.SQLite-JDBC");
        labelPluginNum.setText(String.valueOf(PluginUtil.getInstance().getInstalledPluginNum()));
        ImageIcon imageIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        labelIcon.setIcon(imageIcon);
        labelVersion.setText(TranslateUtil.getInstance().getTranslation("Current Version:") + AllConfigs.version);
        labelCurrentCacheNum.setText(TranslateUtil.getInstance().getTranslation("Current Caches Num:") + AllConfigs.getCacheNum());
    }

    private void setTextFieldAndTextAreaGui() {
        textFieldBackgroundDefault.setText(Integer.toHexString(AllConfigs.getDefaultBackgroundColor()));
        textFieldLabelColor.setText(Integer.toHexString(AllConfigs.getLabelColor()));
        textFieldFontColorWithCoverage.setText(Integer.toHexString(AllConfigs.getLabelFontColorWithCoverage()));
        textFieldTransparency.setText(String.valueOf(AllConfigs.getTransparency()));
        textFieldBorderColor.setText(Integer.toHexString(AllConfigs.getBorderColor()));
        textFieldFontColor.setText(Integer.toHexString(AllConfigs.getLabelFontColor()));
        textFieldSearchBarFontColor.setText(Integer.toHexString(AllConfigs.getSearchBarFontColor()));
        textFieldCacheNum.setText(String.valueOf(AllConfigs.getCacheNumLimit()));
        textFieldSearchDepth.setText(String.valueOf(AllConfigs.getSearchDepth()));
        textFieldHotkey.setText(AllConfigs.getHotkey());
        textFieldPriorityFolder.setText(AllConfigs.getPriorityFolder());
        textFieldUpdateInterval.setText(String.valueOf(AllConfigs.getUpdateTimeLimit()));
        textFieldSearchBarColor.setText(Integer.toHexString(AllConfigs.getSearchBarColor()));
        textFieldAddress.setText(AllConfigs.getProxyAddress());
        textFieldPort.setText(String.valueOf(AllConfigs.getProxyPort()));
        textFieldUserName.setText(AllConfigs.getProxyUserName());
        textFieldPassword.setText(AllConfigs.getProxyPassword());
        textAreaIgnorePath.setText(AllConfigs.getIgnorePath().replaceAll(",", ",\n"));
        if (AllConfigs.getRunAsAdminKeyCode() == 17) {
            textFieldRunAsAdminHotKey.setText("Ctrl + Enter");
        } else if (AllConfigs.getRunAsAdminKeyCode() == 16) {
            textFieldRunAsAdminHotKey.setText("Shift + Enter");
        } else if (AllConfigs.getRunAsAdminKeyCode() == 18) {
            textFieldRunAsAdminHotKey.setText("Alt + Enter");
        }
        if (AllConfigs.getOpenLastFolderKeyCode() == 17) {
            textFieldOpenLastFolder.setText("Ctrl + Enter");
        } else if (AllConfigs.getOpenLastFolderKeyCode() == 16) {
            textFieldOpenLastFolder.setText("Shift + Enter");
        } else if (AllConfigs.getOpenLastFolderKeyCode() == 18) {
            textFieldOpenLastFolder.setText("Alt + Enter");
        }
        if (AllConfigs.getCopyPathKeyCode() == 17) {
            textFieldCopyPath.setText("Ctrl + Enter");
        } else if (AllConfigs.getCopyPathKeyCode() == 16) {
            textFieldCopyPath.setText("Shift + Enter");
        } else if (AllConfigs.getCopyPathKeyCode() == 18) {
            textFieldCopyPath.setText("Alt + Enter");
        }
    }

    private void setColorChooserGui() {
        Color tmp_searchBarColor = new Color(AllConfigs.getSearchBarColor());
        searchBarColorChooser.setBackground(tmp_searchBarColor);
        searchBarColorChooser.setForeground(tmp_searchBarColor);

        Color tmp_defaultBackgroundColor = new Color(AllConfigs.getDefaultBackgroundColor());
        defaultBackgroundChooser.setBackground(tmp_defaultBackgroundColor);
        defaultBackgroundChooser.setForeground(tmp_defaultBackgroundColor);

        Color tmp_labelColor = new Color(AllConfigs.getLabelColor());
        labelColorChooser.setBackground(tmp_labelColor);
        labelColorChooser.setForeground(tmp_labelColor);

        Color tmp_fontColorWithCoverage = new Color(AllConfigs.getLabelFontColorWithCoverage());
        FontColorWithCoverageChooser.setBackground(tmp_fontColorWithCoverage);
        FontColorWithCoverageChooser.setForeground(tmp_fontColorWithCoverage);

        Color tmp_fontColor = new Color(AllConfigs.getLabelFontColor());
        FontColorChooser.setBackground(tmp_fontColor);
        FontColorChooser.setForeground(tmp_fontColor);

        Color tmp_searchBarFontColor = new Color(AllConfigs.getSearchBarFontColor());
        SearchBarFontColorChooser.setBackground(tmp_searchBarFontColor);
        SearchBarFontColorChooser.setForeground(tmp_searchBarFontColor);

        Color tmp_borderColor = new Color(AllConfigs.getBorderColor());
        borderColorChooser.setBackground(tmp_borderColor);
        borderColorChooser.setForeground(tmp_borderColor);
    }

    private void setCheckBoxGui() {
        checkBoxLoseFocus.setSelected(AllConfigs.isLoseFocusClose());
        checkBoxAddToStartup.setSelected(AllConfigs.hasStartup());
        checkBoxAdmin.setSelected(AllConfigs.isDefaultAdmin());
        checkBoxIsShowTipOnCreatingLnk.setSelected(AllConfigs.isShowTipOnCreatingLnk());
    }

    private void setListGui() {
        listCmds.setListData(AllConfigs.getCmdSet().toArray());
        listLanguage.setListData(TranslateUtil.getInstance().getLanguageSet().toArray());
        listLanguage.setSelectedValue(TranslateUtil.getInstance().getLanguage(), true);
        Object[] plugins = PluginUtil.getInstance().getPluginNameArray();
        listPlugins.setListData(plugins);
        listCache.setListData(cacheSet.toArray());
    }

    private void initGUI() {
        //设置窗口显示
        setLabelGui();
        setListGui();
        setColorChooserGui();
        setTextFieldAndTextAreaGui();
        setCheckBoxGui();

        Color color = new Color(43,123,80);

        buttonUpdatePlugin.setVisible(false);
        buttonUpdatePlugin.setBackground(color);

        if (AllConfigs.getProxyType() == AllConfigs.ProxyType.PROXY_DIRECT) {
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
        chooseUpdateAddress.setSelectedItem(AllConfigs.getUpdateAddress());
    }

    private void initCacheArray() {
        String eachLine;
        try (PreparedStatement statement = SQLiteUtil.getPreparedStatement("SELECT PATH FROM cache;");
             ResultSet resultSet = statement.executeQuery()) {
            while (resultSet.next()) {
                eachLine = resultSet.getString("PATH");
                cacheSet.add(eachLine);
            }
        } catch (Exception throwables) {
            throwables.printStackTrace();
        }
    }

    private void selectProxyType() {
        if (AllConfigs.getProxyType() == AllConfigs.ProxyType.PROXY_SOCKS) {
            radioButtonProxyTypeSocks5.setSelected(true);
        } else {
            radioButtonProxyTypeHttp.setSelected(true);
        }
    }

    private SettingsFrame() {
        frame.setUndecorated(true);
        frame.getRootPane().setWindowDecorationStyle(JRootPane.FRAME);

        frameIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        ButtonGroup proxyButtonGroup = new ButtonGroup();
        proxyButtonGroup.add(radioButtonNoProxy);
        proxyButtonGroup.add(radioButtonUseProxy);

        ButtonGroup proxyTypeButtonGroup = new ButtonGroup();
        proxyTypeButtonGroup.add(radioButtonProxyTypeHttp);
        proxyTypeButtonGroup.add(radioButtonProxyTypeSocks5);

        tmp_openLastFolderKeyCode = AllConfigs.getOpenLastFolderKeyCode();
        tmp_runAsAdminKeyCode = AllConfigs.getRunAsAdminKeyCode();
        tmp_copyPathKeyCode = AllConfigs.getCopyPathKeyCode();

        addUpdateAddressToComboBox();

        initCmdSetSettings();

        initCacheArray();

        translate();

        addWindowCloseListener();

        addCheckBoxStartupListener();

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

        addButtonDeleteCacheListener();

        addButtonDeleteAllCacheListener();

        initGUI();

        initThreadPool();
    }

    public boolean isCacheExist(String cache) {
        return cacheSet.contains(cache);
    }

    public void addCache(String cache) {
        cacheSet.add(cache);
    }

    private void addUpdateAddressToComboBox() {
        //todo 添加更新服务器地址
        chooseUpdateAddress.addItem("jsdelivr CDN");
        chooseUpdateAddress.addItem("GitHub");
        chooseUpdateAddress.addItem("GitHack");
        chooseUpdateAddress.addItem("Gitee");
    }

    private void checkDownloadTask(JLabel label, JButton button, String fileName, String originButtonString, String updateSignalFileName) throws IOException {
        //设置进度显示线程
        double progress;
        if (DownloadUtil.getInstance().getDownloadStatus(fileName) != AllConfigs.DownloadStatus.DOWNLOAD_NO_TASK) {
            progress = DownloadUtil.getInstance().getDownloadProgress(fileName);
            label.setText(TranslateUtil.getInstance().getTranslation("Downloading:") + (int) (progress * 100) + "%");

            AllConfigs.DownloadStatus downloadingStatus = DownloadUtil.getInstance().getDownloadStatus(fileName);
            if (downloadingStatus == AllConfigs.DownloadStatus.DOWNLOAD_DONE) {
                //下载完成，禁用按钮
                label.setText(TranslateUtil.getInstance().getTranslation("Downloaded"));
                button.setText(TranslateUtil.getInstance().getTranslation("Downloaded"));
                label.setEnabled(false);
                File updatePluginSign = new File("user/" + updateSignalFileName);
                if (!updatePluginSign.exists()) {
                    updatePluginSign.createNewFile();
                }
            } else if (downloadingStatus == AllConfigs.DownloadStatus.DOWNLOAD_ERROR) {
                //下载错误，重置button
                label.setText(TranslateUtil.getInstance().getTranslation("Download failed"));
                button.setText(TranslateUtil.getInstance().getTranslation(originButtonString));
                button.setEnabled(true);
            } else if (downloadingStatus == AllConfigs.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                //正在下载
                button.setText(TranslateUtil.getInstance().getTranslation("Cancel"));
            } else if (downloadingStatus == AllConfigs.DownloadStatus.DOWNLOAD_INTERRUPTED) {
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
    }

    private void addShowDownloadProgressTask(JLabel label, JButton button, String fileName) {
        try {
            String originString = button.getText();
            while (AllConfigs.isNotMainExit()) {
                checkDownloadTask(label, button, fileName, originString, "update");
                TimeUnit.MILLISECONDS.sleep(200);
            }
        } catch (InterruptedException | IOException ignored) {
        }
    }

    private void initThreadPool() {
        CachedThreadPool.getInstance().executeTask(() ->
                addShowDownloadProgressTask(labelDownloadProgress, buttonCheckUpdate, AllConfigs.FILE_NAME));

        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                String fileName;
                String originString = buttonUpdatePlugin.getText();
                while (AllConfigs.isNotMainExit()) {
                    if (isUpdateButtonPluginString) {
                        isUpdateButtonPluginString = false;
                        originString = buttonUpdatePlugin.getText();
                    } else {
                        fileName = (String) listPlugins.getSelectedValue();
                        checkDownloadTask(labelProgress, buttonUpdatePlugin, fileName + ".jar", originString, "updatePlugin");
                    }
                    TimeUnit.MILLISECONDS.sleep(200);
                }
            } catch (InterruptedException | IOException ignored) {
            }
        });
    }

    private void translateTabbedPane() {
        tabbedPane.setTitleAt(0, TranslateUtil.getInstance().getTranslation("General"));
        tabbedPane.setTitleAt(1, TranslateUtil.getInstance().getTranslation("Search settings"));
        tabbedPane.setTitleAt(2, TranslateUtil.getInstance().getTranslation("Search bar settings"));
        tabbedPane.setTitleAt(3, TranslateUtil.getInstance().getTranslation("Cache"));
        tabbedPane.setTitleAt(4, TranslateUtil.getInstance().getTranslation("Proxy settings"));
        tabbedPane.setTitleAt(5, TranslateUtil.getInstance().getTranslation("Plugins"));
        tabbedPane.setTitleAt(6, TranslateUtil.getInstance().getTranslation("Hotkey settings"));
        tabbedPane.setTitleAt(7, TranslateUtil.getInstance().getTranslation("Language settings"));
        tabbedPane.setTitleAt(8, TranslateUtil.getInstance().getTranslation("My commands"));
        tabbedPane.setTitleAt(9, TranslateUtil.getInstance().getTranslation("Color settings"));
        tabbedPane.setTitleAt(10, TranslateUtil.getInstance().getTranslation("About"));
    }

    private void translateLabels() {
        labelMaxCacheNum.setText(TranslateUtil.getInstance().getTranslation("Set the maximum number of caches:"));
        labelUpdateInterval.setText(TranslateUtil.getInstance().getTranslation("File update detection interval:"));
        labelSecond.setText(TranslateUtil.getInstance().getTranslation("Seconds"));
        labelSearchDepth.setText(TranslateUtil.getInstance().getTranslation("Search depth (too large may affect performance):"));
        labeltipPriorityFolder.setText(TranslateUtil.getInstance().getTranslation("Priority search folder location (double-click to clear):"));
        labelConstIgnorePathTip.setText(TranslateUtil.getInstance().getTranslation("Separate different paths with commas, and ignore C:\\Windows by default"));
        labelSetIgnorePathTip.setText(TranslateUtil.getInstance().getTranslation("Set ignore folder:"));
        labelTransparency.setText(TranslateUtil.getInstance().getTranslation("Search bar transparency:"));
        labelOpenSearchBarHotKey.setText(TranslateUtil.getInstance().getTranslation("Open search bar:"));
        labelRunAsAdminHotKey.setText(TranslateUtil.getInstance().getTranslation("Run as administrator:"));
        labelOpenFolderHotKey.setText(TranslateUtil.getInstance().getTranslation("Open the parent folder:"));
        labelCopyPathHotKey.setText(TranslateUtil.getInstance().getTranslation("Copy path:"));
        labelCmdTip2.setText(TranslateUtil.getInstance().getTranslation("You can add custom commands here. After adding, " +
                "you can enter \": + your set identifier\" in the search box to quickly open"));
        labelColorTip.setText(TranslateUtil.getInstance().getTranslation("Please enter the hexadecimal value of RGB color"));
        labelSearchBarColor.setText(TranslateUtil.getInstance().getTranslation("Search bar Color:"));
        labelLabelColor.setText(TranslateUtil.getInstance().getTranslation("Chosen label color:"));
        labelFontColor.setText(TranslateUtil.getInstance().getTranslation("Chosen label font Color:"));
        labelDefaultColor.setText(TranslateUtil.getInstance().getTranslation("Default background Color:"));
        labelNotChosenFontColor.setText(TranslateUtil.getInstance().getTranslation("Unchosen label font Color:"));
        labelGitHubTip.setText(TranslateUtil.getInstance().getTranslation("This is an open source software,GitHub:"));
        labelGithubIssue.setText(TranslateUtil.getInstance().getTranslation("If you find a bug or have some suggestions, welcome to GitHub for feedback"));
        labelDescription.setText(TranslateUtil.getInstance().getTranslation("Thanks for the following project"));
        labelTranslationTip.setText(TranslateUtil.getInstance().getTranslation("The translation might not be 100% accurate"));
        labelLanguageChooseTip.setText(TranslateUtil.getInstance().getTranslation("Choose a language"));
        labelVersion.setText(TranslateUtil.getInstance().getTranslation("Current Version:") + AllConfigs.version);
        labelInstalledPluginNum.setText(TranslateUtil.getInstance().getTranslation("Installed plugins num:"));
        labelVacuumTip.setText(TranslateUtil.getInstance().getTranslation("Click to organize the database and reduce the size of the database,"));
        labelVacuumTip2.setText(TranslateUtil.getInstance().getTranslation("but it will consume a lot of time."));
        labelAddress.setText(TranslateUtil.getInstance().getTranslation("Address"));
        labelPort.setText(TranslateUtil.getInstance().getTranslation("Port"));
        labelUserName.setText(TranslateUtil.getInstance().getTranslation("User name"));
        labelPassword.setText(TranslateUtil.getInstance().getTranslation("Password"));
        labelProxyTip.setText(TranslateUtil.getInstance().getTranslation("If you need a proxy to access the Internet, You can add a proxy here."));
        labelCacheSettings.setText(TranslateUtil.getInstance().getTranslation("Cache Settings"));
        labelCacheTip.setText(TranslateUtil.getInstance().getTranslation("You can edit the saved caches here"));
        labelCacheTip2.setText(TranslateUtil.getInstance().getTranslation("The cache is automatically generated " +
                "by the software and will be displayed first when searching."));
        labelSearchBarFontColor.setText(TranslateUtil.getInstance().getTranslation("SearchBar Font Color:"));
        labelBorderColor.setText(TranslateUtil.getInstance().getTranslation("Border Color:"));
        labelCurrentCacheNum.setText(TranslateUtil.getInstance().getTranslation("Current Caches Num:") + AllConfigs.getCacheNum());
        labelUninstallPluginTip.setText(TranslateUtil.getInstance().getTranslation("If you need to delete a plug-in, just delete it under the \"plugins\" folder in the software directory."));
        labelUninstallPluginTip2.setText(TranslateUtil.getInstance().getTranslation("Tip:"));
        chooseUpdateAddressLabel.setText(TranslateUtil.getInstance().getTranslation("Choose update address"));
    }

    private void translateCheckBoxs() {
        checkBoxAddToStartup.setText(TranslateUtil.getInstance().getTranslation("Add to startup"));
        checkBoxLoseFocus.setText(TranslateUtil.getInstance().getTranslation("Close search bar when focus lost"));
        checkBoxAdmin.setText(TranslateUtil.getInstance().getTranslation("Open other programs as an administrator " +
                "(provided that the software has privileges)"));
        checkBoxIsShowTipOnCreatingLnk.setText(TranslateUtil.getInstance().getTranslation("Show tip on creating shortcut"));
    }

    private void translateButtons() {
        buttonSaveAndRemoveDesktop.setText(TranslateUtil.getInstance().getTranslation("Backup and remove all desktop files"));
        ButtonPriorityFolder.setText(TranslateUtil.getInstance().getTranslation("Choose"));
        buttonChooseFile.setText(TranslateUtil.getInstance().getTranslation("Choose"));
        buttonAddCMD.setText(TranslateUtil.getInstance().getTranslation("Add"));
        buttonDelCmd.setText(TranslateUtil.getInstance().getTranslation("Delete"));
        buttonResetColor.setText(TranslateUtil.getInstance().getTranslation("Reset to default"));
        buttonCheckUpdate.setText(TranslateUtil.getInstance().getTranslation("Check for update"));
        buttonSave.setText(TranslateUtil.getInstance().getTranslation("Save"));
        buttonUpdatePlugin.setText(TranslateUtil.getInstance().getTranslation("Check for update"));
        buttonPluginMarket.setText(TranslateUtil.getInstance().getTranslation("Plugin Market"));
        buttonDeleteCache.setText(TranslateUtil.getInstance().getTranslation("Delete cache"));
        buttonDeleteAllCache.setText(TranslateUtil.getInstance().getTranslation("Delete all"));
        buttonVacuum.setText(TranslateUtil.getInstance().getTranslation("Optimize database"));
    }

    private void translateRadioButtons() {
        radioButtonNoProxy.setText(TranslateUtil.getInstance().getTranslation("No proxy"));
        radioButtonUseProxy.setText(TranslateUtil.getInstance().getTranslation("Configure proxy"));
    }

    private void translate() {
        translateTabbedPane();
        translateLabels();
        translateCheckBoxs();
        translateButtons();
        translateRadioButtons();
        frame.setTitle(TranslateUtil.getInstance().getTranslation("Settings"));
    }

    private boolean isRepeatCommand(String name) {
        name = ":" + name;
        for (String each : AllConfigs.getCmdSet()) {
            if (each.substring(0, each.indexOf(";")).equals(name)) {
                return true;
            }
        }
        return false;
    }

    private static String getUpdateUrl() {
        //todo 添加更新服务器地址
        switch (AllConfigs.getUpdateAddress()) {
            case "jsdelivr CDN":
                return "https://cdn.jsdelivr.net/gh/XUANXUQAQ/File-Engine-Version/version.json";
            case "GitHub":
                return "https://raw.githubusercontent.com/XUANXUQAQ/File-Engine-Version/master/version.json";
            case "GitHack":
                return "https://raw.githack.com/XUANXUQAQ/File-Engine-Version/master/version.json";
            case "Gitee":
                return "https://gitee.com/XUANXUQAQ/file-engine-version/raw/master/version.json";
            default:
                return null;
        }
    }

    public static JSONObject getUpdateInfo() throws IOException, InterruptedException {
        DownloadUtil downloadUtil = DownloadUtil.getInstance();
        AllConfigs.DownloadStatus downloadStatus = downloadUtil.getDownloadStatus("version.json");
        if (downloadStatus != AllConfigs.DownloadStatus.DOWNLOAD_DOWNLOADING) {
            String url = getUpdateUrl();
            if (url != null) {
                downloadUtil.downLoadFromUrl(url,
                        "version.json", "tmp");
                int count = 0;
                boolean isError = false;
                //wait for task
                while (downloadUtil.getDownloadStatus("version.json") != AllConfigs.DownloadStatus.DOWNLOAD_DONE) {
                    count++;
                    if (count >= 3) {
                        isError = true;
                        break;
                    }
                    TimeUnit.SECONDS.sleep(1);
                }
                if (isError) {
                    throw new IOException("Download failed.");
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
        }
        return null;
    }

    private void initCmdSetSettings() {
        //获取所有自定义命令
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("user/cmds.txt"), StandardCharsets.UTF_8))) {
            String each;
            while ((each = br.readLine()) != null) {
                AllConfigs.addToCmdSet(each);
            }
        } catch (IOException ignored) {
        }
    }

    public void hideFrame() {
        frame.setVisible(false);
    }


    protected boolean isSettingsFrameVisible() {
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
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.setResizable(false);

        tabbedPane.setSelectedIndex(0);
        frame.setLocationRelativeTo(null);
        float transparency = AllConfigs.getTransparency();
        frame.setOpacity(transparency < 0.6f ? 0.95f : transparency);
        frame.setVisible(true);
    }

    private void checkUpdateTimeLimit(StringBuilder strBuilder) {
        int updateTimeLimitTemp;
        try {
            updateTimeLimitTemp = Integer.parseInt(textFieldUpdateInterval.getText());
        } catch (Exception e1) {
            updateTimeLimitTemp = -1; // 输入不正确
        }
        if (updateTimeLimitTemp > 3600 || updateTimeLimitTemp <= 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("The file index update setting is wrong, please change")).append("\n");
        }
    }

    private void checkCacheNumLimit(StringBuilder strBuilder) {
        int cacheNumLimitTemp;
        try {
            cacheNumLimitTemp = Integer.parseInt(textFieldCacheNum.getText());
        } catch (Exception e1) {
            cacheNumLimitTemp = -1;
        }
        if (cacheNumLimitTemp > 10000 || cacheNumLimitTemp <= 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("The cache capacity is set incorrectly, please change")).append("\n");
        }
    }

    private void checkSearchDepth(StringBuilder strBuilder) {
        int searchDepthTemp;
        try {
            searchDepthTemp = Integer.parseInt(textFieldSearchDepth.getText());
        } catch (Exception e1) {
            searchDepthTemp = -1;
        }
        if (searchDepthTemp > 10 || searchDepthTemp <= 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Search depth setting is wrong, please change")).append("\n");
        }
    }

    private void checkHotKey(StringBuilder strBuilder) {
        String tmp_hotkey = textFieldHotkey.getText();
        if (tmp_hotkey.length() < 5) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Hotkey setting is wrong, please change")).append("\n");
        } else {
            if (!CheckHotKeyUtil.getInstance().isHotkeyAvailable(tmp_hotkey)) {
                strBuilder.append(TranslateUtil.getInstance().getTranslation("Hotkey setting is wrong, please change")).append("\n");
            }
        }
        if (tmp_openLastFolderKeyCode == tmp_runAsAdminKeyCode || tmp_openLastFolderKeyCode == tmp_copyPathKeyCode || tmp_runAsAdminKeyCode == tmp_copyPathKeyCode) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("HotKey conflict")).append("\n");
        }
    }

    private void checkTransparency(StringBuilder strBuilder) {
        float transparencyTemp;
        try {
            transparencyTemp = Float.parseFloat(textFieldTransparency.getText());
        } catch (Exception e) {
            transparencyTemp = -1f;
        }
        if (transparencyTemp > 1 || transparencyTemp <= 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Transparency setting error")).append("\n");
        }
    }

    private void checkLabelColor(StringBuilder strBuilder) {
        int tmp_labelColor;
        try {
            tmp_labelColor = Integer.parseInt(textFieldLabelColor.getText(), 16);
        } catch (Exception e) {
            tmp_labelColor = -1;
        }
        if (tmp_labelColor < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Chosen label color is set incorrectly")).append("\n");
        }
    }

    private void checkLabelFontColorWithCoverage(StringBuilder strBuilder) {
        int tmp_fontColorWithCoverage;
        try {
            tmp_fontColorWithCoverage = Integer.parseInt(textFieldFontColorWithCoverage.getText(), 16);
        } catch (Exception e) {
            tmp_fontColorWithCoverage = -1;
        }
        if (tmp_fontColorWithCoverage < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Chosen label font color is set incorrectly")).append("\n");
        }
    }

    private void checkDefaultBackgroundColor(StringBuilder strBuilder) {
        int tmp_defaultBackgroundColor;
        try {
            tmp_defaultBackgroundColor = Integer.parseInt(textFieldBackgroundDefault.getText(), 16);
        } catch (Exception e) {
            tmp_defaultBackgroundColor = -1;
        }
        if (tmp_defaultBackgroundColor < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Incorrect default background color setting")).append("\n");
        }
    }

    private void checkBorderColor(StringBuilder stringBuilder) {
        int tmp_borderColor;
        try {
            tmp_borderColor = Integer.parseInt(textFieldBorderColor.getText(), 16);
        }catch (Exception e) {
            tmp_borderColor = -1;
        }
        if (tmp_borderColor < 0) {
            stringBuilder.append(TranslateUtil.getInstance().getTranslation("Border color is set incorrectly")).append("\n");
        }
    }

    private void checkLabelFontColor(StringBuilder strBuilder) {
        int tmp_labelFontColor;
        try {
            tmp_labelFontColor = Integer.parseInt(textFieldFontColor.getText(), 16);
        } catch (Exception e) {
            tmp_labelFontColor = -1;
        }
        if (tmp_labelFontColor < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Unchosen label font color is set incorrectly")).append("\n");
        }
    }

    private void checkSearchBarColor(StringBuilder strBuilder) {
        int tmp_searchBarColor;
        try {
            tmp_searchBarColor = Integer.parseInt(textFieldSearchBarColor.getText(), 16);
        } catch (Exception e) {
            tmp_searchBarColor = -1;
        }
        if (tmp_searchBarColor < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("The color of the search bar is set incorrectly")).append("\n");
        }
    }

    private void checkSearchBarFontColor(StringBuilder strBuilder) {
        int tmp_searchBarFontColor;
        try {
            tmp_searchBarFontColor = Integer.parseInt(textFieldSearchBarFontColor.getText(), 16);
        } catch (Exception e) {
            tmp_searchBarFontColor = -1;
        }
        if (tmp_searchBarFontColor < 0) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("The font color of the search bar is set incorrectly")).append("\n");
        }
    }

    private void checkProxy(StringBuilder strBuilder) {
        try {
            Integer.parseInt(textFieldPort.getText());
        } catch (Exception e) {
            strBuilder.append(TranslateUtil.getInstance().getTranslation("Proxy port is set incorrectly.")).append("\n");
        }
    }

    private String saveChanges(boolean isPopErrorWindow) {
        StringBuilder strBuilder = new StringBuilder();

        checkProxy(strBuilder);
        checkSearchBarColor(strBuilder);
        checkSearchBarFontColor(strBuilder);
        checkLabelColor(strBuilder);
        checkLabelFontColor(strBuilder);
        checkLabelFontColorWithCoverage(strBuilder);
        checkBorderColor(strBuilder);
        checkDefaultBackgroundColor(strBuilder);
        checkTransparency(strBuilder);
        checkHotKey(strBuilder);
        checkSearchDepth(strBuilder);
        checkCacheNumLimit(strBuilder);
        checkUpdateTimeLimit(strBuilder);

        String ignorePathTemp;
        ignorePathTemp = textAreaIgnorePath.getText();
        ignorePathTemp = ignorePathTemp.replaceAll("\n", "");

        String errors = strBuilder.toString();
        if (!errors.isEmpty()) {
            if (isPopErrorWindow) {
                JOptionPane.showMessageDialog(frame, errors);
            }
            return errors;
        }

        //重新显示翻译GUI
        if (!listLanguage.getSelectedValue().equals(TranslateUtil.getInstance().getLanguage())) {
            TranslateUtil.getInstance().setLanguage((String) listLanguage.getSelectedValue());
            translate();
        }

        //所有配置均正确
        //使所有配置生效
        String tmp_proxyAddress = textFieldAddress.getText();
        String tmp_proxyUserName = textFieldUserName.getText();
        String tmp_proxyPassword = textFieldPassword.getText();

        setStartup(checkBoxAddToStartup.isSelected());
        AllConfigs.allowChangeSettings();

        if (radioButtonProxyTypeSocks5.isSelected()) {
            AllConfigs.setProxyType(AllConfigs.ProxyType.PROXY_SOCKS);
        } else if (radioButtonProxyTypeHttp.isSelected()) {
            AllConfigs.setProxyType(AllConfigs.ProxyType.PROXY_HTTP);
        }
        if (radioButtonNoProxy.isSelected()) {
            AllConfigs.setProxyType(AllConfigs.ProxyType.PROXY_DIRECT);
        }
        AllConfigs.setUpdateAddress((String) chooseUpdateAddress.getSelectedItem());
        AllConfigs.setPriorityFolder(textFieldPriorityFolder.getText());
        AllConfigs.setHotkey(textFieldHotkey.getText());
        AllConfigs.setCacheNumLimit(Integer.parseInt(textFieldCacheNum.getText()));
        AllConfigs.setUpdateTimeLimit(Integer.parseInt(textFieldUpdateInterval.getText()));
        AllConfigs.setIgnorePath(ignorePathTemp);
        AllConfigs.setSearchDepth(Integer.parseInt(textFieldSearchDepth.getText()));
        AllConfigs.setIsDefaultAdmin(checkBoxAdmin.isSelected());
        AllConfigs.setIsLoseFocusClose(checkBoxLoseFocus.isSelected());
        AllConfigs.setIsShowTipCreatingLnk(checkBoxIsShowTipOnCreatingLnk.isSelected());
        AllConfigs.setTransparency(Float.parseFloat(textFieldTransparency.getText()));
        AllConfigs.setLabelColor(Integer.parseInt(textFieldLabelColor.getText(), 16));
        AllConfigs.setBorderColor(Integer.parseInt(textFieldBorderColor.getText(), 16));
        AllConfigs.setDefaultBackgroundColor(Integer.parseInt(textFieldBackgroundDefault.getText(), 16));
        AllConfigs.setSearchBarColor(Integer.parseInt(textFieldSearchBarColor.getText(), 16));
        AllConfigs.setLabelFontColorWithCoverage(Integer.parseInt(textFieldFontColorWithCoverage.getText(), 16));
        AllConfigs.setLabelFontColor(Integer.parseInt(textFieldFontColor.getText(), 16));
        AllConfigs.setSearchBarFontColor(Integer.parseInt(textFieldSearchBarFontColor.getText(), 16));
        AllConfigs.setProxyAddress(tmp_proxyAddress);
        AllConfigs.setProxyPort(Integer.parseInt(textFieldPort.getText()));
        AllConfigs.setProxyUserName(tmp_proxyUserName);
        AllConfigs.setProxyPassword(tmp_proxyPassword);
        AllConfigs.setOpenLastFolderKeyCode(tmp_openLastFolderKeyCode);
        AllConfigs.setRunAsAdminKeyCode(tmp_runAsAdminKeyCode);
        AllConfigs.setCopyPathKeyCode(tmp_copyPathKeyCode);

        AllConfigs.denyChangeSettings();

        AllConfigs.setAllSettings();

        AllConfigs.saveAllSettings();

        Color tmp_color = new Color(AllConfigs.getLabelColor());
        labelColorChooser.setBackground(tmp_color);
        labelColorChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getDefaultBackgroundColor());
        defaultBackgroundChooser.setBackground(tmp_color);
        defaultBackgroundChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getLabelFontColorWithCoverage());
        FontColorWithCoverageChooser.setBackground(tmp_color);
        FontColorWithCoverageChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getLabelFontColor());
        FontColorChooser.setBackground(tmp_color);
        FontColorChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getSearchBarFontColor());
        SearchBarFontColorChooser.setBackground(tmp_color);
        SearchBarFontColorChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getBorderColor());
        borderColorChooser.setBackground(tmp_color);
        borderColorChooser.setForeground(tmp_color);

        PluginUtil.getInstance().setCurrentTheme(AllConfigs.getDefaultBackgroundColor(), AllConfigs.getLabelColor(), AllConfigs.getBorderColor());

        //保存自定义命令
        StringBuilder strb = new StringBuilder();
        for (String each : AllConfigs.getCmdSet()) {
            strb.append(each);
            strb.append("\n");
        }
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/cmds.txt"), StandardCharsets.UTF_8))) {
            bw.write(strb.toString());
        } catch (IOException ignored) {
        }
        hideFrame();
        return "";
    }

    private void setStartup(boolean b) {
        if (isStartupChanged) {
            isStartupChanged = false;
            if (b) {
                String command = "cmd.exe /c schtasks /create /ru \"administrators\" /rl HIGHEST /sc ONLOGON /tn \"File-Engine\" /tr ";
                File FileEngine = new File(AllConfigs.FILE_NAME);
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
                    StringBuilder result = new StringBuilder();
                    try (BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()))) {
                        String line;
                        while ((line = outPut.readLine()) != null) {
                            result.append(line);
                        }
                    }
                    if (!result.toString().isEmpty()) {
                        checkBoxAddToStartup.setSelected(true);
                        JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Delete startup failed, please try to run as administrator"));
                    }
                } catch (IOException | InterruptedException ignored) {
                }
            }
        }
    }
}
