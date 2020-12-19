package FileEngine.frames;

import FileEngine.IsDebug;
import FileEngine.database.SQLiteUtil;
import FileEngine.checkHotkey.CheckHotKeyUtil;
import FileEngine.configs.AllConfigs;
import FileEngine.configs.Enums;
import FileEngine.download.DownloadUtil;
import FileEngine.eventHandler.EventUtil;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventHandler;
import FileEngine.eventHandler.impl.frame.pluginMarket.ShowPluginMarket;
import FileEngine.eventHandler.impl.frame.settingsFrame.HideSettingsFrameEvent;
import FileEngine.eventHandler.impl.configs.SaveConfigsEvent;
import FileEngine.eventHandler.impl.database.DeleteFromCacheEvent;
import FileEngine.eventHandler.impl.database.OptimiseDatabaseEvent;
import FileEngine.eventHandler.impl.download.StartDownloadEvent;
import FileEngine.eventHandler.impl.download.StopDownloadEvent;
import FileEngine.eventHandler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import FileEngine.eventHandler.impl.plugin.AddPluginsCanUpdateEvent;
import FileEngine.eventHandler.impl.plugin.RemoveFromPluginsCanUpdateEvent;
import FileEngine.moveFiles.MoveDesktopFiles;
import FileEngine.pluginSystem.Plugin;
import FileEngine.pluginSystem.PluginUtil;
import FileEngine.r.R;
import FileEngine.database.DatabaseUtil;
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
import java.util.ArrayList;
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
    private JLabel labelPlaceHolder;
    private JList<Object> listSwingThemes;
    private JScrollPane paneSwingThemes;
    private JButton buttonChangeTheme;
    private JLabel labelRemoveDesktop;
    private JLabel labelPlaceHolder2;
    private JLabel labelPlaceHolder3;

    private static volatile SettingsFrame instance = null;


    public static SettingsFrame getInstance() {
        if (instance == null) {
            synchronized (SettingsFrame.class) {
                if (instance == null) {
                    instance = new SettingsFrame();
                }
            }
        }
        return instance;
    }

    private void addWindowCloseListener() {
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                String errors = saveChanges();
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
                AllConfigs.getInstance().getCmdSet().add(":" + name + ";" + cmd);
                listCmds.setListData(AllConfigs.getInstance().getCmdSet().toArray());
            }
        });
    }

    private void addButtonDelCMDListener() {
        buttonDelCmd.addActionListener(e -> {
            String del = (String) listCmds.getSelectedValue();
            if (del != null) {
                AllConfigs.getInstance().getCmdSet().remove(del);
                listCmds.setListData(AllConfigs.getInstance().getCmdSet().toArray());
            }

        });
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
            Enums.DownloadStatus status = DownloadUtil.getInstance().getDownloadStatus(AllConfigs.FILE_NAME);
            if (status == Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                //取消下载
                String fileName = AllConfigs.FILE_NAME;
                EventUtil.getInstance().putTask(new StopDownloadEvent(fileName));
                //复位button
                buttonCheckUpdate.setText(TranslateUtil.getInstance().getTranslation("Check for update"));
                buttonCheckUpdate.setEnabled(true);
            } else if (status == Enums.DownloadStatus.DOWNLOAD_DONE) {
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
                        EventUtil.getInstance().putTask(new StartDownloadEvent(
                                updateInfo.getString(urlChoose), fileName, AllConfigs.getInstance().getTmp().getAbsolutePath()));

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

    private void addButtonChangeThemeListener() {
        //移除显示theme框，改为弹出窗口
        tabGeneral.remove(paneSwingThemes);
        buttonChangeTheme.addActionListener(e -> JOptionPane.showMessageDialog(frame, paneSwingThemes, "Theme", JOptionPane.PLAIN_MESSAGE));
    }

    private void addButtonDeleteCacheListener() {
        buttonDeleteCache.addActionListener(e -> {
            String cache = (String) listCache.getSelectedValue();
            if (cache != null) {
                EventUtil.getInstance().putTask(new DeleteFromCacheEvent(cache));
                cacheSet.remove(cache);
                AllConfigs.getInstance().decrementCacheNum();
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

    private void addButtonVacuumListener() {
        buttonVacuum.addActionListener(e -> {
            int ret = JOptionPane.showConfirmDialog(frame, TranslateUtil.getInstance().getTranslation("Confirm whether to start optimizing the database?"));
            if (JOptionPane.YES_OPTION == ret) {
                Enums.DatabaseStatus status = DatabaseUtil.getInstance().getStatus();
                if (status == Enums.DatabaseStatus.NORMAL) {
                    if (IsDebug.isDebug()) {
                        System.out.println("开始优化");
                    }
                    EventUtil.getInstance().putTask(new OptimiseDatabaseEvent());
                    CachedThreadPool.getInstance().executeTask(() -> {
                        //实时显示VACUUM状态
                        try {
                            DatabaseUtil instance = DatabaseUtil.getInstance();
                            while (instance.getStatus() == Enums.DatabaseStatus.VACUUM) {
                                labelVacuumStatus.setText(TranslateUtil.getInstance().getTranslation("Optimizing..."));
                                TimeUnit.MILLISECONDS.sleep(50);
                            }
                            labelVacuumStatus.setText(TranslateUtil.getInstance().getTranslation("Optimized"));
                            TimeUnit.SECONDS.sleep(3);
                            labelVacuumStatus.setText("");
                        } catch (InterruptedException ignored) {
                        }
                    });
                } else if (status == Enums.DatabaseStatus.MANUAL_UPDATE) {
                    JOptionPane.showMessageDialog(frame, TranslateUtil.getInstance().getTranslation("Database is not usable yet, please wait..."));
                } else if (status == Enums.DatabaseStatus.VACUUM) {
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
                    EventUtil.getInstance().putTask(new DeleteFromCacheEvent(each));
                }
                cacheSet.clear();
                AllConfigs.getInstance().resetCacheNumToZero();
                listCache.setListData(cacheSet.toArray());
            }
        });
    }

    private void addButtonViewPluginMarketListener() {
        buttonPluginMarket.addActionListener(e -> {
            EventUtil.getInstance().putTask(new ShowPluginMarket());
        });
    }

    private void addSwingThemePreviewListener() {
        listSwingThemes.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                String swingTheme = (String) listSwingThemes.getSelectedValue();
                AllConfigs.getInstance().setSwingPreview(swingTheme);
            }
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
            Enums.DownloadStatus downloadStatus = DownloadUtil.getInstance().getDownloadStatus(pluginFullName);
            if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                //取消下载
                EventUtil.getInstance().putTask(new StopDownloadEvent(pluginFullName));
                buttonUpdatePlugin.setEnabled(true);
            } else if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_DONE) {
                buttonUpdatePlugin.setEnabled(false);
                EventUtil.getInstance().putTask(new RemoveFromPluginsCanUpdateEvent(pluginName));
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
                            if (!EventUtil.getInstance().isNotMainExit()) {
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
                        EventUtil.getInstance().putTask(new StartDownloadEvent(
                                url, pluginFullName, new File(AllConfigs.getInstance().getTmp(), "pluginsUpdate").getAbsolutePath()));
                    } else {
                        EventUtil.getInstance().putTask(new AddPluginsCanUpdateEvent(pluginName));
                        int ret = JOptionPane.showConfirmDialog(frame, TranslateUtil.getInstance().getTranslation("New version available, do you want to update?"));
                        if (ret == JOptionPane.YES_OPTION) {
                            //开始下载
                            String url = plugin.getUpdateURL();
                            EventUtil.getInstance().putTask(new StartDownloadEvent(
                                    url, pluginFullName, new File(AllConfigs.getInstance().getTmp(), "pluginsUpdate").getAbsolutePath()));
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
        labelCurrentCacheNum.setText(TranslateUtil.getInstance().getTranslation("Current Caches Num:") + AllConfigs.getInstance().getCacheNum());
    }

    private void setTextFieldAndTextAreaGui() {
        textFieldBackgroundDefault.setText(Integer.toHexString(AllConfigs.getInstance().getDefaultBackgroundColor()));
        textFieldLabelColor.setText(Integer.toHexString(AllConfigs.getInstance().getLabelColor()));
        textFieldFontColorWithCoverage.setText(Integer.toHexString(AllConfigs.getInstance().getLabelFontColorWithCoverage()));
        textFieldTransparency.setText(String.valueOf(AllConfigs.getInstance().getTransparency()));
        textFieldBorderColor.setText(Integer.toHexString(AllConfigs.getInstance().getBorderColor()));
        textFieldFontColor.setText(Integer.toHexString(AllConfigs.getInstance().getLabelFontColor()));
        textFieldSearchBarFontColor.setText(Integer.toHexString(AllConfigs.getInstance().getSearchBarFontColor()));
        textFieldCacheNum.setText(String.valueOf(AllConfigs.getInstance().getCacheNumLimit()));
        textFieldSearchDepth.setText(String.valueOf(AllConfigs.getInstance().getSearchDepth()));
        textFieldHotkey.setText(AllConfigs.getInstance().getHotkey());
        textFieldPriorityFolder.setText(AllConfigs.getInstance().getPriorityFolder());
        textFieldUpdateInterval.setText(String.valueOf(AllConfigs.getInstance().getUpdateTimeLimit()));
        textFieldSearchBarColor.setText(Integer.toHexString(AllConfigs.getInstance().getSearchBarColor()));
        textFieldAddress.setText(AllConfigs.getInstance().getProxyAddress());
        textFieldPort.setText(String.valueOf(AllConfigs.getInstance().getProxyPort()));
        textFieldUserName.setText(AllConfigs.getInstance().getProxyUserName());
        textFieldPassword.setText(AllConfigs.getInstance().getProxyPassword());
        textAreaIgnorePath.setText(AllConfigs.getInstance().getIgnorePath().replaceAll(",", ",\n"));
        if (AllConfigs.getInstance().getRunAsAdminKeyCode() == 17) {
            textFieldRunAsAdminHotKey.setText("Ctrl + Enter");
        } else if (AllConfigs.getInstance().getRunAsAdminKeyCode() == 16) {
            textFieldRunAsAdminHotKey.setText("Shift + Enter");
        } else if (AllConfigs.getInstance().getRunAsAdminKeyCode() == 18) {
            textFieldRunAsAdminHotKey.setText("Alt + Enter");
        }
        if (AllConfigs.getInstance().getOpenLastFolderKeyCode() == 17) {
            textFieldOpenLastFolder.setText("Ctrl + Enter");
        } else if (AllConfigs.getInstance().getOpenLastFolderKeyCode() == 16) {
            textFieldOpenLastFolder.setText("Shift + Enter");
        } else if (AllConfigs.getInstance().getOpenLastFolderKeyCode() == 18) {
            textFieldOpenLastFolder.setText("Alt + Enter");
        }
        if (AllConfigs.getInstance().getCopyPathKeyCode() == 17) {
            textFieldCopyPath.setText("Ctrl + Enter");
        } else if (AllConfigs.getInstance().getCopyPathKeyCode() == 16) {
            textFieldCopyPath.setText("Shift + Enter");
        } else if (AllConfigs.getInstance().getCopyPathKeyCode() == 18) {
            textFieldCopyPath.setText("Alt + Enter");
        }
    }

    private void setColorChooserGui() {
        Color tmp_searchBarColor = new Color(AllConfigs.getInstance().getSearchBarColor());
        searchBarColorChooser.setBackground(tmp_searchBarColor);
        searchBarColorChooser.setForeground(tmp_searchBarColor);

        Color tmp_defaultBackgroundColor = new Color(AllConfigs.getInstance().getDefaultBackgroundColor());
        defaultBackgroundChooser.setBackground(tmp_defaultBackgroundColor);
        defaultBackgroundChooser.setForeground(tmp_defaultBackgroundColor);

        Color tmp_labelColor = new Color(AllConfigs.getInstance().getLabelColor());
        labelColorChooser.setBackground(tmp_labelColor);
        labelColorChooser.setForeground(tmp_labelColor);

        Color tmp_fontColorWithCoverage = new Color(AllConfigs.getInstance().getLabelFontColorWithCoverage());
        FontColorWithCoverageChooser.setBackground(tmp_fontColorWithCoverage);
        FontColorWithCoverageChooser.setForeground(tmp_fontColorWithCoverage);

        Color tmp_fontColor = new Color(AllConfigs.getInstance().getLabelFontColor());
        FontColorChooser.setBackground(tmp_fontColor);
        FontColorChooser.setForeground(tmp_fontColor);

        Color tmp_searchBarFontColor = new Color(AllConfigs.getInstance().getSearchBarFontColor());
        SearchBarFontColorChooser.setBackground(tmp_searchBarFontColor);
        SearchBarFontColorChooser.setForeground(tmp_searchBarFontColor);

        Color tmp_borderColor = new Color(AllConfigs.getInstance().getBorderColor());
        borderColorChooser.setBackground(tmp_borderColor);
        borderColorChooser.setForeground(tmp_borderColor);
    }

    private void setCheckBoxGui() {
        checkBoxLoseFocus.setSelected(AllConfigs.getInstance().isLoseFocusClose());
        checkBoxAddToStartup.setSelected(AllConfigs.getInstance().hasStartup());
        checkBoxAdmin.setSelected(AllConfigs.getInstance().isDefaultAdmin());
        checkBoxIsShowTipOnCreatingLnk.setSelected(AllConfigs.getInstance().isShowTipOnCreatingLnk());
    }

    private void setListGui() {
        listCmds.setListData(AllConfigs.getInstance().getCmdSet().toArray());
        listLanguage.setListData(TranslateUtil.getInstance().getLanguageSet().toArray());
        listLanguage.setSelectedValue(TranslateUtil.getInstance().getLanguage(), true);
        Object[] plugins = PluginUtil.getInstance().getPluginNameArray();
        listPlugins.setListData(plugins);
        listCache.setListData(cacheSet.toArray());
        ArrayList<String> list = new ArrayList<>();
        for (Enums.SwingThemes each : Enums.SwingThemes.values()) {
            list.add(each.toString());
        }
        listSwingThemes.setListData(list.toArray());
        listSwingThemes.setSelectedValue(AllConfigs.getInstance().getSwingTheme(), true);
    }

    private void initGUI() {
        //设置窗口显示
        setLabelGui();
        setListGui();
        setColorChooserGui();
        setTextFieldAndTextAreaGui();
        setCheckBoxGui();

        buttonUpdatePlugin.setVisible(false);

        if (AllConfigs.getInstance().getProxyType() == Enums.ProxyType.PROXY_DIRECT) {
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
        chooseUpdateAddress.setSelectedItem(AllConfigs.getInstance().getUpdateAddress());
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
        if (AllConfigs.getInstance().getProxyType() == Enums.ProxyType.PROXY_SOCKS) {
            radioButtonProxyTypeSocks5.setSelected(true);
        } else {
            radioButtonProxyTypeHttp.setSelected(true);
        }
    }

    private SettingsFrame() {
        R.getInstance().addComponent("settings-frame", frame);
        frame.setUndecorated(true);
        frame.getRootPane().setWindowDecorationStyle(JRootPane.FRAME);

        tabbedPane.setBackground(new Color(0,0,0,0));

        frameIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        ButtonGroup proxyButtonGroup = new ButtonGroup();
        proxyButtonGroup.add(radioButtonNoProxy);
        proxyButtonGroup.add(radioButtonUseProxy);

        ButtonGroup proxyTypeButtonGroup = new ButtonGroup();
        proxyTypeButtonGroup.add(radioButtonProxyTypeHttp);
        proxyTypeButtonGroup.add(radioButtonProxyTypeSocks5);

        tmp_openLastFolderKeyCode = AllConfigs.getInstance().getOpenLastFolderKeyCode();
        tmp_runAsAdminKeyCode = AllConfigs.getInstance().getRunAsAdminKeyCode();
        tmp_copyPathKeyCode = AllConfigs.getInstance().getCopyPathKeyCode();

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

        addGitHubLabelListener();

        addCheckForUpdateButtonListener();

        addTextFieldCopyPathListener();

        addResetColorButtonListener();

        addColorChooserLabelListener();

        addListPluginMouseListener();

        addButtonPluginUpdateCheckListener();

        addButtonViewPluginMarketListener();

        addSwingThemePreviewListener();

        addPluginOfficialSiteListener();

        addButtonVacuumListener();

        addButtonProxyListener();

        addButtonDeleteCacheListener();

        addButtonChangeThemeListener();

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
        if (DownloadUtil.getInstance().getDownloadStatus(fileName) != Enums.DownloadStatus.DOWNLOAD_NO_TASK) {
            progress = DownloadUtil.getInstance().getDownloadProgress(fileName);
            label.setText(TranslateUtil.getInstance().getTranslation("Downloading:") + (int) (progress * 100) + "%");

            Enums.DownloadStatus downloadingStatus = DownloadUtil.getInstance().getDownloadStatus(fileName);
            if (downloadingStatus == Enums.DownloadStatus.DOWNLOAD_DONE) {
                //下载完成，禁用按钮
                label.setText(TranslateUtil.getInstance().getTranslation("Downloaded"));
                button.setText(TranslateUtil.getInstance().getTranslation("Downloaded"));
                label.setEnabled(false);
                File updatePluginSign = new File("user/" + updateSignalFileName);
                if (!updatePluginSign.exists()) {
                    updatePluginSign.createNewFile();
                }
            } else if (downloadingStatus == Enums.DownloadStatus.DOWNLOAD_ERROR) {
                //下载错误，重置button
                label.setText(TranslateUtil.getInstance().getTranslation("Download failed"));
                button.setText(TranslateUtil.getInstance().getTranslation(originButtonString));
                button.setEnabled(true);
            } else if (downloadingStatus == Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                //正在下载
                button.setText(TranslateUtil.getInstance().getTranslation("Cancel"));
            } else if (downloadingStatus == Enums.DownloadStatus.DOWNLOAD_INTERRUPTED) {
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
            while (EventUtil.getInstance().isNotMainExit()) {
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
                while (EventUtil.getInstance().isNotMainExit()) {
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
        labelCurrentCacheNum.setText(TranslateUtil.getInstance().getTranslation("Current Caches Num:") + AllConfigs.getInstance().getCacheNum());
        labelUninstallPluginTip.setText(TranslateUtil.getInstance().getTranslation("If you need to delete a plug-in, just delete it under the \"plugins\" folder in the software directory."));
        labelUninstallPluginTip2.setText(TranslateUtil.getInstance().getTranslation("Tip:"));
        chooseUpdateAddressLabel.setText(TranslateUtil.getInstance().getTranslation("Choose update address"));
        labelRemoveDesktop.setText(TranslateUtil.getInstance().getTranslation("Backup and remove all desktop files") + ":");
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
        buttonUpdatePlugin.setText(TranslateUtil.getInstance().getTranslation("Check for update"));
        buttonPluginMarket.setText(TranslateUtil.getInstance().getTranslation("Plugin Market"));
        buttonDeleteCache.setText(TranslateUtil.getInstance().getTranslation("Delete cache"));
        buttonDeleteAllCache.setText(TranslateUtil.getInstance().getTranslation("Delete all"));
        buttonChangeTheme.setText(TranslateUtil.getInstance().getTranslation("change theme"));
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
        for (String each : AllConfigs.getInstance().getCmdSet()) {
            if (each.substring(0, each.indexOf(";")).equals(name)) {
                return true;
            }
        }
        return false;
    }

    private static String getUpdateUrl() {
        //todo 添加更新服务器地址
        switch (AllConfigs.getInstance().getUpdateAddress()) {
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
        Enums.DownloadStatus downloadStatus = downloadUtil.getDownloadStatus("version.json");
        if (downloadStatus != Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
            String url = getUpdateUrl();
            if (url != null) {
                EventUtil.getInstance().putTask(new StartDownloadEvent(
                        url, "version.json", AllConfigs.getInstance().getTmp().getAbsolutePath()));
                int count = 0;
                boolean isError = false;
                //wait for task
                while (downloadUtil.getDownloadStatus("version.json") != Enums.DownloadStatus.DOWNLOAD_DONE) {
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
                AllConfigs.getInstance().addToCmdSet(each);
            }
        } catch (IOException ignored) {
        }
    }

    private void hideFrame() {
        frame.setVisible(false);
    }

    public static void registerEventHandler() {
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.register(ShowSettingsFrameEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().showWindow();
            }
        });
        eventUtil.register(HideSettingsFrameEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().hideFrame();
            }
        });
    }

    protected boolean isSettingsFrameVisible() {
        return frame.isVisible();
    }

    private void showWindow() {
        initGUI();
        frame.setResizable(true);
        int width = Integer.parseInt(TranslateUtil.getInstance().getFrameWidth());
        int height = Integer.parseInt(TranslateUtil.getInstance().getFrameHeight());

        panel.setSize(width, height);
        frame.setSize(width, height);
        frame.setContentPane(getInstance().panel);
        frame.setIconImage(frameIcon.getImage());
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.setResizable(false);

        tabbedPane.setSelectedIndex(0);
        frame.setLocationRelativeTo(null);
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

    private String saveChanges() {
        StringBuilder errorsStrb = new StringBuilder();

        checkProxy(errorsStrb);
        checkSearchBarColor(errorsStrb);
        checkSearchBarFontColor(errorsStrb);
        checkLabelColor(errorsStrb);
        checkLabelFontColor(errorsStrb);
        checkLabelFontColorWithCoverage(errorsStrb);
        checkBorderColor(errorsStrb);
        checkDefaultBackgroundColor(errorsStrb);
        checkTransparency(errorsStrb);
        checkHotKey(errorsStrb);
        checkSearchDepth(errorsStrb);
        checkCacheNumLimit(errorsStrb);
        checkUpdateTimeLimit(errorsStrb);

        String ignorePathTemp = textAreaIgnorePath.getText().replaceAll("\n", "");

        String swingTheme = (String) listSwingThemes.getSelectedValue();

        String errors = errorsStrb.toString();
        if (!errors.isEmpty()) {
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
        AllConfigs.getInstance().allowChangeSettings();

        if (radioButtonProxyTypeSocks5.isSelected()) {
            AllConfigs.getInstance().setProxyType(Enums.ProxyType.PROXY_SOCKS);
        } else if (radioButtonProxyTypeHttp.isSelected()) {
            AllConfigs.getInstance().setProxyType(Enums.ProxyType.PROXY_HTTP);
        }
        if (radioButtonNoProxy.isSelected()) {
            AllConfigs.getInstance().setProxyType(Enums.ProxyType.PROXY_DIRECT);
        }
        AllConfigs.getInstance().setUpdateAddress((String) chooseUpdateAddress.getSelectedItem());
        AllConfigs.getInstance().setPriorityFolder(textFieldPriorityFolder.getText());
        AllConfigs.getInstance().setHotkey(textFieldHotkey.getText());
        AllConfigs.getInstance().setCacheNumLimit(Integer.parseInt(textFieldCacheNum.getText()));
        AllConfigs.getInstance().setUpdateTimeLimit(Integer.parseInt(textFieldUpdateInterval.getText()));
        AllConfigs.getInstance().setIgnorePath(ignorePathTemp);
        AllConfigs.getInstance().setSearchDepth(Integer.parseInt(textFieldSearchDepth.getText()));
        AllConfigs.getInstance().setIsDefaultAdmin(checkBoxAdmin.isSelected());
        AllConfigs.getInstance().setIsLoseFocusClose(checkBoxLoseFocus.isSelected());
        AllConfigs.getInstance().setIsShowTipCreatingLnk(checkBoxIsShowTipOnCreatingLnk.isSelected());
        AllConfigs.getInstance().setTransparency(Float.parseFloat(textFieldTransparency.getText()));
        AllConfigs.getInstance().setLabelColor(Integer.parseInt(textFieldLabelColor.getText(), 16));
        AllConfigs.getInstance().setBorderColor(Integer.parseInt(textFieldBorderColor.getText(), 16));
        AllConfigs.getInstance().setDefaultBackgroundColor(Integer.parseInt(textFieldBackgroundDefault.getText(), 16));
        AllConfigs.getInstance().setSearchBarColor(Integer.parseInt(textFieldSearchBarColor.getText(), 16));
        AllConfigs.getInstance().setLabelFontColorWithCoverage(Integer.parseInt(textFieldFontColorWithCoverage.getText(), 16));
        AllConfigs.getInstance().setLabelFontColor(Integer.parseInt(textFieldFontColor.getText(), 16));
        AllConfigs.getInstance().setSearchBarFontColor(Integer.parseInt(textFieldSearchBarFontColor.getText(), 16));
        AllConfigs.getInstance().setProxyAddress(tmp_proxyAddress);
        AllConfigs.getInstance().setProxyPort(Integer.parseInt(textFieldPort.getText()));
        AllConfigs.getInstance().setProxyUserName(tmp_proxyUserName);
        AllConfigs.getInstance().setProxyPassword(tmp_proxyPassword);
        AllConfigs.getInstance().setOpenLastFolderKeyCode(tmp_openLastFolderKeyCode);
        AllConfigs.getInstance().setRunAsAdminKeyCode(tmp_runAsAdminKeyCode);
        AllConfigs.getInstance().setCopyPathKeyCode(tmp_copyPathKeyCode);
        AllConfigs.getInstance().setSwingTheme(swingTheme);

        AllConfigs.getInstance().denyChangeSettings();

        EventUtil.getInstance().putTask(new SaveConfigsEvent());

        Color tmp_color = new Color(AllConfigs.getInstance().getLabelColor());
        labelColorChooser.setBackground(tmp_color);
        labelColorChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getInstance().getDefaultBackgroundColor());
        defaultBackgroundChooser.setBackground(tmp_color);
        defaultBackgroundChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getInstance().getLabelFontColorWithCoverage());
        FontColorWithCoverageChooser.setBackground(tmp_color);
        FontColorWithCoverageChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getInstance().getLabelFontColor());
        FontColorChooser.setBackground(tmp_color);
        FontColorChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getInstance().getSearchBarFontColor());
        SearchBarFontColorChooser.setBackground(tmp_color);
        SearchBarFontColorChooser.setForeground(tmp_color);
        tmp_color = new Color(AllConfigs.getInstance().getBorderColor());
        borderColorChooser.setBackground(tmp_color);
        borderColorChooser.setForeground(tmp_color);

        //保存自定义命令
        StringBuilder strb = new StringBuilder();
        for (String each : AllConfigs.getInstance().getCmdSet()) {
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
