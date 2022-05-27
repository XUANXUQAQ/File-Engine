package file.engine.frames;

import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.ConfigEntity;
import file.engine.configs.Constants;
import file.engine.dllInterface.IsLocalDisk;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.SetSwingLaf;
import file.engine.event.handler.impl.configs.AddCmdEvent;
import file.engine.event.handler.impl.configs.DeleteCmdEvent;
import file.engine.event.handler.impl.configs.SaveConfigsEvent;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.database.*;
import file.engine.event.handler.impl.download.IsTaskDoneBeforeEvent;
import file.engine.event.handler.impl.download.StartDownloadEvent;
import file.engine.event.handler.impl.download.StopDownloadEvent;
import file.engine.event.handler.impl.frame.pluginMarket.ShowPluginMarket;
import file.engine.event.handler.impl.frame.searchBar.HideSearchBarEvent;
import file.engine.event.handler.impl.frame.searchBar.PreviewSearchBarEvent;
import file.engine.event.handler.impl.frame.searchBar.StartPreviewEvent;
import file.engine.event.handler.impl.frame.searchBar.StopPreviewEvent;
import file.engine.event.handler.impl.frame.settingsFrame.*;
import file.engine.event.handler.impl.hotkey.CheckHotKeyAvailableEvent;
import file.engine.event.handler.impl.hotkey.ResponseCtrlEvent;
import file.engine.event.handler.impl.plugin.AddPluginsCanUpdateEvent;
import file.engine.event.handler.impl.plugin.GetPluginByNameEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.services.DatabaseService;
import file.engine.services.TranslateService;
import file.engine.services.download.DownloadManager;
import file.engine.services.plugin.system.Plugin;
import file.engine.services.plugin.system.PluginService;
import file.engine.utils.*;
import file.engine.utils.file.MoveDesktopFiles;
import file.engine.utils.system.properties.IsDebug;
import file.engine.utils.system.properties.IsPreview;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileSystemView;
import javax.swing.table.DefaultTableModel;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeNode;
import javax.swing.tree.TreePath;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.List;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static file.engine.utils.ColorUtil.*;
import static file.engine.utils.StartupUtil.hasStartup;


public class SettingsFrame {
    private final Set<String> cacheSet = new ConcurrentSkipListSet<>();
    private static volatile int tmp_copyPathKeyCode;
    private static volatile int tmp_runAsAdminKeyCode;
    private static volatile int tmp_openLastFolderKeyCode;
    private static final ImageIcon frameIcon = new ImageIcon(Objects.requireNonNull(SettingsFrame.class.getResource("/icons/frame.png")));
    private static final JFrame frame = new JFrame("Settings");
    private static final TranslateService TRANSLATE_SERVICE = TranslateService.getInstance();
    private static final EventManagement eventManagement = EventManagement.getInstance();
    private static final AllConfigs allConfigs = AllConfigs.getInstance();
    private static final CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
    private final HashMap<TabNameAndTitle, Component> tabComponentNameMap = new HashMap<>();
    private final HashMap<String, Integer> suffixMap = new HashMap<>();
    private final Set<Component> excludeComponent = ConcurrentHashMap.newKeySet();
    private final HashSet<String> diskSet = new HashSet<>();
    private JTextField textFieldUpdateInterval;
    private JTextField textFieldCacheNum;
    private JTextArea textAreaIgnorePath;
    private JCheckBox checkBoxAddToStartup;
    private JLabel labelSetIgnorePathTip;
    private JLabel labelUpdateInterval;
    private JLabel labelOpenSearchBarHotKey;
    private JLabel labelMaxCacheNum;
    private JLabel labelSecond;
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
    private JLabel labelColorTip;
    private JTextField textFieldBackgroundDefault;
    private JTextField textFieldFontColorWithCoverage;
    private JTextField textFieldLabelColor;
    private JLabel labelLabelColor;
    private JLabel labelFontColor;
    private JLabel labelDefaultColor;
    private JButton buttonResetColor;
    private JTextField textFieldFontColor;
    private JLabel labelNotChosenFontColor;
    private JLabel labelColorChooser;
    private JLabel FontColorWithCoverageChooser;
    private JLabel defaultBackgroundChooser;
    private JLabel FontColorChooser;
    private JLabel labelSearchBarColor;
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
    private JPanel tabPlugin;
    private JLabel labelInstalledPluginNum;
    private JLabel labelPluginNum;
    private JPanel PluginListPanel;
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
    private JRadioButton radioButtonProxyTypeHttp;
    private JRadioButton radioButtonProxyTypeSocks5;
    private JLabel labelProxyTip;
    private JLabel labelPlaceHolder8;
    private JLabel labelVacuumStatus;
    private JLabel labelApiVersion;
    private JComboBox<String> chooseUpdateAddress;
    private JLabel chooseUpdateAddressLabel;
    private JLabel labelPlaceHolderWhatever2;
    private JPanel tabCache;
    private JLabel labelCacheSettings;
    private JLabel labelCacheTip;
    private JList<Object> listCache;
    private JButton buttonDeleteCache;
    private JScrollPane cacheScrollPane;
    private JLabel labelVacuumTip2;
    private JLabel labelCacheTip2;
    private JButton buttonDeleteAllCache;
    private JLabel labelSearchBarFontColor;
    private JTextField textFieldSearchBarFontColor;
    private JLabel SearchBarFontColorChooser;
    private JLabel labelBorderColor;
    private JTextField textFieldBorderColor;
    private JLabel borderColorChooser;
    private JLabel labelCurrentCacheNum;
    private JLabel labelUninstallPluginTip;
    private JLabel labelUninstallPluginTip2;
    private JCheckBox checkBoxIsShowTipOnCreatingLnk;
    private JLabel labelPlaceHolder12;
    private JList<Object> listSwingThemes;
    private JScrollPane paneSwingThemes;
    private JButton buttonChangeTheme;
    private JScrollPane paneListPlugins;
    private JScrollPane paneListLanguage;
    private JButton buttonPreviewColor;
    private JButton buttonClosePreview;
    private JTextField textFieldSearchCommands;
    private JLabel labelSearchCommand;
    private JLabel placeholderN;
    private JLabel placeholderl;
    private JLabel placeholdersearch0;
    private JLabel labelTinyPinyin;
    private JLabel labelLombok;
    private JPanel tabModifyPriority;
    private JScrollPane suffixScrollpane;
    private JButton buttonAddSuffix;
    private JButton buttonDeleteSuffix;
    private JTable tableSuffix;
    private JButton buttonDeleteAllSuffix;
    private JLabel labelSuffixTip;
    private JLabel labelPlaceHolder;
    private JCheckBox checkBoxResponseCtrl;
    private JTree treeSettings;
    private JLabel placeholder1;
    private JLabel placeholder2;
    private JLabel placeholder3;
    private JLabel placeholderInterface1;
    private JLabel placeholderInterface2;
    private JLabel placeholderInterface3;
    private JLabel placeholderInterface4;
    private JLabel placeholderPlugin1;
    private JLabel placeholderPlugin2;
    private JLabel placeholderPlugins3;
    private JLabel placeholderHotkey1;
    private JLabel placeholderPlugin5;
    private JLabel placeholderGeneral;
    private JLabel placeholderSearch;
    private JLabel placeholderGeneral1;
    private JPanel tabIndex;
    private JLabel labelIndexTip;
    private JLabel labelIndexChooseDisk;
    private JButton buttonAddNewDisk;
    private JButton buttonRebuildIndex;
    private JLabel placeholderIndex1;
    private JButton buttonDeleteDisk;
    private JScrollPane scrollPaneDisks;
    private JList<Object> listDisks;
    private JLabel labelTipNTFSTip;
    private JLabel labelLocalDiskTip;
    private JCheckBox checkBoxCheckUpdate;
    private JLabel labelBorderType;
    private JLabel labelBorderThickness;
    private JTextField textFieldBorderThickness;
    private JComboBox<Object> comboBoxBorderType;
    private JCheckBox checkBoxIsAttachExplorer;
    private JLabel labelZip;
    private JLabel labelRoundRadius;
    private JTextField textFieldRoundRadius;
    private JSplitPane splitPane;
    private JPanel leftPanel;
    private JPanel rightPanel;
    private JLabel labelSearchSettingsPlaceholder;
    private JLabel labelpriorityPlaceholder;
    private JLabel labelCachePlaceholder;
    private JLabel labelLanguagePlaceholder;
    private JLabel labelCommandsPlaceholder;
    private JLabel labelSearchSettingsPlaceholder2;
    private JTextField textFieldSearchCache;


    private static volatile SettingsFrame instance = null;


    private static SettingsFrame getInstance() {
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
                    int ret = JOptionPane.showConfirmDialog(null, TRANSLATE_SERVICE.getTranslation("Errors") + ":\n" + errors + "\n" + TRANSLATE_SERVICE.getTranslation("Failed to save settings, do you still close the window"));
                    if (ret == JOptionPane.YES_OPTION) {
                        hideFrame();
                    }
                } else {
                    hideFrame();
                }
            }
        });
    }

    /**
     * 添加到开机启动监听器
     */
    private void addCheckBoxStartupListener() {
        checkBoxAddToStartup.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                setStartup(!checkBoxAddToStartup.isSelected());
            }
        });
    }

    /**
     * 清空桌面按钮监听器
     */
    private void addButtonRemoveDesktopListener() {
        buttonSaveAndRemoveDesktop.addActionListener(e -> {
            String currentFolder = new File("").getAbsolutePath();
            if (currentFolder.equals(FileSystemView.getFileSystemView().getHomeDirectory().getAbsolutePath()) || "C:\\Users\\Public\\Desktop".equals(currentFolder)) {
                JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("The program is detected on the desktop and cannot be moved"));
                return;
            }
            int isConfirmed = JOptionPane.showConfirmDialog(frame, TRANSLATE_SERVICE.getTranslation("Whether to remove and backup all files on the desktop," + "they will be in the program's Files folder, which may take a few minutes"));
            if (isConfirmed == JOptionPane.YES_OPTION) {
                Future<Boolean> future = cachedThreadPoolUtil.executeTask(MoveDesktopFiles::start, false);
                try {
                    if (future == null) {
                        return;
                    }
                    if (!future.get()) {
                        JOptionPane.showMessageDialog(null, TRANSLATE_SERVICE.getTranslation("Files with the same name are detected, please move them by yourself"));
                    }
                } catch (InterruptedException | ExecutionException exception) {
                    exception.printStackTrace();
                }
            }
        });
    }

    private void addFileChooserButtonListener() {
        buttonChooseFile.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), TRANSLATE_SERVICE.getTranslation("Choose"));
            File file = fileChooser.getSelectedFile();
            if (file != null && returnValue == JFileChooser.APPROVE_OPTION) {
                textAreaIgnorePath.append(file.getAbsolutePath() + ",\n");
            }
        });
    }

    /**
     * 快捷键响应
     */
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

    /**
     * 选择优先文件夹
     */
    private void addPriorityFileChooserListener() {
        ButtonPriorityFolder.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), TRANSLATE_SERVICE.getTranslation("Choose"));
            File file = fileChooser.getSelectedFile();
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                textFieldPriorityFolder.setText(file.getAbsolutePath());
            }
        });
    }

    /**
     * 双击有限文件夹显示textField后清空
     */
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

    /**
     * 打开所在文件夹监听器
     */
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

    /**
     * 添加自定义命令监听器
     */
    private void addButtonCMDListener() {
        buttonAddCMD.addActionListener(e -> {
            String name = JOptionPane.showInputDialog(TRANSLATE_SERVICE.getTranslation("Please enter the ID of the command, then you can enter \": identifier\" in the search box to execute the command directly"));
            if (name == null || name.isEmpty()) {
                //未输入
                return;
            }
            if ("update".equalsIgnoreCase(name) || "clearbin".equalsIgnoreCase(name) || "help".equalsIgnoreCase(name) || "version".equalsIgnoreCase(name) || isRepeatCommand(name)) {
                JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Conflict with existing commands"));
                return;
            }
            if (name.length() == 1) {
                int ret = JOptionPane.showConfirmDialog(frame, TRANSLATE_SERVICE.getTranslation("The identifier you entered is too short, continue") + "?");
                if (ret != JOptionPane.OK_OPTION) {
                    return;
                }
            }
            String cmd;
            JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Please select the location of the executable file (a folder is also acceptable)"));
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
            int returnValue = fileChooser.showDialog(new Label(), TRANSLATE_SERVICE.getTranslation("Choose"));
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                cmd = fileChooser.getSelectedFile().getAbsolutePath();
                eventManagement.putEvent(new AddCmdEvent(":" + name + ";" + cmd), event -> listCmds.setListData(allConfigs.getCmdSet().toArray()), event -> listCmds.setListData(allConfigs.getCmdSet().toArray()));
            }
        });
    }

    /**
     * 删除自定义命令监听器
     */
    private void addButtonDelCMDListener() {
        buttonDelCmd.addActionListener(e -> {
            String del = (String) listCmds.getSelectedValue();
            if (del != null) {
                eventManagement.putEvent(new DeleteCmdEvent(del), event -> listCmds.setListData(allConfigs.getCmdSet().toArray()), event -> listCmds.setListData(allConfigs.getCmdSet().toArray()));
            }
        });
    }

    /**
     * 点击打开github
     */
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

    /**
     * 点击检查更新
     */
    private void addCheckForUpdateButtonListener() {
        var downloadManager = new Object() {
            DownloadManager newJar;
            DownloadManager newLauncher;
        };

        buttonCheckUpdate.addActionListener(e -> {
            if (downloadManager.newJar != null && downloadManager.newJar.getDownloadStatus() == Constants.Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                eventManagement.putEvent(new StopDownloadEvent(downloadManager.newJar));
                eventManagement.putEvent(new StopDownloadEvent(downloadManager.newLauncher));
            } else {
                //开始下载
                Map<String, Object> updateInfo;
                String latestVersion;
                try {
                    updateInfo = allConfigs.getUpdateInfo();
                    if (updateInfo != null) {
                        latestVersion = (String) updateInfo.get("version");
                    } else {
                        throw new IOException("failed");
                    }
                } catch (IOException e1) {
                    JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Check update failed"));
                    showManualDownloadDialog();
                    return;
                }
                if (Double.parseDouble(latestVersion) > Double.parseDouble(Constants.version) || IsPreview.isPreview() || IsDebug.isDebug()) {
                    String description = (String) updateInfo.get("description");
                    int result = JOptionPane.showConfirmDialog(frame, TRANSLATE_SERVICE.getTranslation("New Version available") + latestVersion + "," + TRANSLATE_SERVICE.getTranslation("Whether to update") + "\n" + TRANSLATE_SERVICE.getTranslation("update content") + "\n" + description);
                    if (result == JOptionPane.YES_OPTION) {
                        //开始更新,下载更新文件到tmp
                        downloadManager.newJar = new DownloadManager((String) updateInfo.get("url64"), Constants.FILE_NAME, new File("tmp").getAbsolutePath());
                        downloadManager.newLauncher = new DownloadManager((String) updateInfo.get("urlLauncher"), Constants.LAUNCH_WRAPPER_NAME, new File("tmp").getAbsolutePath());
                        eventManagement.putEvent(new StartDownloadEvent(downloadManager.newLauncher));
                        cachedThreadPoolUtil.executeTask(() -> {
                            try {
                                if (!downloadManager.newLauncher.waitFor(10 * 60 * 1000)) {
                                    return;
                                }
                                if (downloadManager.newLauncher.getDownloadStatus() == Constants.Enums.DownloadStatus.DOWNLOAD_DONE) {
                                    Files.createFile(Path.of("user/updateLauncher"));
                                }
                            } catch (IOException e1) {
                                e1.printStackTrace();
                            }
                        });
                        eventManagement.putEvent(new StartDownloadEvent(downloadManager.newJar));
                        cachedThreadPoolUtil.executeTask(() -> {
                            boolean isDownloadSuccess = SetDownloadProgress.setProgress(labelDownloadProgress, buttonCheckUpdate, downloadManager.newLauncher, () -> Constants.FILE_NAME.equals(downloadManager.newJar.fileName), new File("user/update"), "", null, null);
                            if (!isDownloadSuccess) {
                                showManualDownloadDialog();
                            }
                        });
                    }
                } else {
                    JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Latest version:") + latestVersion + "\n" + TRANSLATE_SERVICE.getTranslation("The current version is the latest"));
                }
            }
        });
    }

    /**
     * 显示手动下载弹窗
     */
    private void showManualDownloadDialog() {
        int ret = JOptionPane.showConfirmDialog(frame, TRANSLATE_SERVICE.getTranslation("Do you want to download it manually") + "?");
        if (ret == JOptionPane.YES_OPTION) {
            Desktop desktop;
            if (Desktop.isDesktopSupported()) {
                desktop = Desktop.getDesktop();
                try {
                    desktop.browse(new URI("https://github.com/XUANXUQAQ/File-Engine/releases/"));
                } catch (IOException | URISyntaxException ioException) {
                    ioException.printStackTrace();
                }
            }
        }
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

    /**
     * 重置颜色设置
     */
    private void addResetColorButtonListener() {
        buttonResetColor.addActionListener(e -> {
            textFieldFontColorWithCoverage.setText(toRGBHexString(Constants.DEFAULT_FONT_COLOR_WITH_COVERAGE));
            textFieldSearchBarColor.setText(toRGBHexString(Constants.DEFAULT_SEARCHBAR_COLOR));
            textFieldLabelColor.setText(toRGBHexString(Constants.DEFAULT_LABEL_COLOR));
            textFieldBackgroundDefault.setText(toRGBHexString(Constants.DEFAULT_WINDOW_BACKGROUND_COLOR));
            textFieldFontColor.setText(toRGBHexString(Constants.DEFAULT_FONT_COLOR));
            textFieldSearchBarFontColor.setText(toRGBHexString(Constants.DEFAULT_SEARCHBAR_FONT_COLOR));
            textFieldBorderColor.setText(toRGBHexString(Constants.DEFAULT_BORDER_COLOR));
        });
    }

    /**
     * 检查字符串是否可以解析成指定范围的数字
     *
     * @param str 字符串
     * @param min 最小值
     * @param max 最大值
     * @return boolean
     */
    private boolean canParseInteger(String str, int min, int max) {
        try {
            int ret = Integer.parseInt(str);
            if (min <= ret && ret <= max) {
                return true;
            } else {
                throw new Exception("parse failed");
            }
        } catch (Exception e) {
            return false;
        }
    }

    @SuppressWarnings("SameParameterValue")
    private boolean canParseDouble(String str, double min, double max) {
        try {
            double v = Double.parseDouble(str);
            if (min <= v && v <= max) {
                return true;
            }
            throw new Exception("parse failed");
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 获取保存颜色信息的textField的信息，失败为null
     *
     * @param textField textField
     * @return Color
     */
    private Color getColorFromTextFieldStr(JTextField textField) {
        String tmp;
        return canParseToRGB(tmp = textField.getText()) ? new Color(Integer.parseInt(tmp, 16)) : null;
    }

    /**
     * 在label上显示颜色
     *
     * @param color color
     * @param label JLabel
     */
    private void setColorChooserLabel(Color color, JLabel label) {
        if (color != null) {
            label.setBackground(color);
            label.setForeground(color);
        }
    }

    /**
     * 实时监测textField中保存的颜色信息，并尝试更新label
     */
    private void addColorChooserLabelListener() {
        cachedThreadPoolUtil.executeTask(() -> {
            try {
                Color labelColor;
                Color fontColorWithCoverage;
                Color defaultBackgroundColor;
                Color defaultFontColor;
                Color searchBarColor;
                Color searchBarFontColor;
                Color borderColor;
                while (eventManagement.notMainExit()) {
                    labelColor = getColorFromTextFieldStr(textFieldLabelColor);
                    fontColorWithCoverage = getColorFromTextFieldStr(textFieldFontColorWithCoverage);
                    defaultBackgroundColor = getColorFromTextFieldStr(textFieldBackgroundDefault);
                    defaultFontColor = getColorFromTextFieldStr(textFieldFontColor);
                    searchBarColor = getColorFromTextFieldStr(textFieldSearchBarColor);
                    searchBarFontColor = getColorFromTextFieldStr(textFieldSearchBarFontColor);
                    borderColor = getColorFromTextFieldStr(textFieldBorderColor);
                    setColorChooserLabel(labelColor, labelColorChooser);
                    setColorChooserLabel(fontColorWithCoverage, FontColorWithCoverageChooser);
                    setColorChooserLabel(defaultBackgroundColor, defaultBackgroundChooser);
                    setColorChooserLabel(defaultFontColor, FontColorChooser);
                    setColorChooserLabel(searchBarColor, searchBarColorChooser);
                    setColorChooserLabel(searchBarFontColor, SearchBarFontColorChooser);
                    setColorChooserLabel(borderColor, borderColorChooser);
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException ignored) {
            }
        });

        labelColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, TRANSLATE_SERVICE.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldLabelColor.setText(parseColorHex(color));
            }
        });
        FontColorWithCoverageChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, TRANSLATE_SERVICE.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldFontColorWithCoverage.setText(parseColorHex(color));
            }
        });
        defaultBackgroundChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, TRANSLATE_SERVICE.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldBackgroundDefault.setText(parseColorHex(color));
            }
        });
        FontColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, TRANSLATE_SERVICE.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldFontColor.setText(parseColorHex(color));
            }
        });

        SearchBarFontColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, TRANSLATE_SERVICE.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldSearchBarFontColor.setText(parseColorHex(color));
            }
        });

        borderColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, TRANSLATE_SERVICE.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldBorderColor.setText(parseColorHex(color));
            }
        });

        searchBarColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, TRANSLATE_SERVICE.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldSearchBarColor.setText(parseColorHex(color));
            }
        });
    }

    /**
     * 点击插件在右方显示详细信息
     */
    private void addListPluginMouseListener() {
        listPlugins.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                String pluginName = (String) listPlugins.getSelectedValue();
                if (pluginName != null) {
                    GetPluginByNameEvent getPluginByNameEvent = new GetPluginByNameEvent(pluginName);
                    eventManagement.putEvent(getPluginByNameEvent);
                    eventManagement.waitForEvent(getPluginByNameEvent);
                    Optional<PluginService.PluginInfo> pluginInfoOptional = getPluginByNameEvent.getReturnValue();
                    pluginInfoOptional.ifPresent(pluginInfo -> {
                        Plugin plugin = pluginInfo.plugin;
                        int apiVersion;
                        ImageIcon icon = plugin.getPluginIcon();
                        String description = plugin.getDescription();
                        String officialSite = plugin.getOfficialSite();
                        String version = plugin.getVersion();
                        String author = plugin.getAuthor();
                        apiVersion = plugin.getApiVersion();
                        labelPluginVersion.setText(TRANSLATE_SERVICE.getTranslation("Version") + ":" + version);
                        labelApiVersion.setText("API " + TRANSLATE_SERVICE.getTranslation("Version") + ":" + apiVersion);
                        PluginIconLabel.setIcon(icon);
                        PluginNamelabel.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                        textAreaDescription.setText(description);
                        labelAuthor.setText(TRANSLATE_SERVICE.getTranslation("Author") + ":" + author);
                        labelOfficialSite.setText("<html><a href='" + officialSite + "'><font size=\"4\">" + pluginName + "</font></a></html>");
                        labelProgress.setText("");
                        buttonUpdatePlugin.setVisible(true);
                    });
                    if (PluginService.getInstance().hasPluginNotLatest(pluginName)) {
                        IsTaskDoneBeforeEvent isTaskDoneBeforeEvent = new IsTaskDoneBeforeEvent(new DownloadManager(null, pluginName + ".jar", new File("tmp", "pluginsUpdate").getAbsolutePath()));
                        eventManagement.putEvent(isTaskDoneBeforeEvent);
                        eventManagement.waitForEvent(isTaskDoneBeforeEvent);
                        Optional<Boolean> returnValue = isTaskDoneBeforeEvent.getReturnValue();
                        returnValue.ifPresent((isTaskDone) -> {
                            if (isTaskDone) {
                                buttonUpdatePlugin.setEnabled(false);
                                buttonUpdatePlugin.setText(TRANSLATE_SERVICE.getTranslation("Downloaded"));
                            } else {
                                Color color = new Color(51, 122, 183);
                                buttonUpdatePlugin.setText(TRANSLATE_SERVICE.getTranslation("Update"));
                                buttonUpdatePlugin.setBackground(color);
                                buttonUpdatePlugin.setEnabled(true);
                            }
                        });
                    } else {
                        buttonUpdatePlugin.setText(TRANSLATE_SERVICE.getTranslation("Check for update"));
                        buttonUpdatePlugin.setEnabled(true);
                    }
                }
            }
        });
    }

    /**
     * 点击插件名打开浏览器到插件首页
     */
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

    /**
     * 显示更新主题弹窗
     */
    private void addButtonChangeThemeListener() {
        //移除显示theme框，改为弹出窗口
        buttonChangeTheme.addActionListener(e -> JOptionPane.showMessageDialog(frame, paneSwingThemes, TRANSLATE_SERVICE.getTranslation("Change Theme"), JOptionPane.PLAIN_MESSAGE));
    }

    private Object[] queryListData(String keyword) {
        return cacheSet.stream().filter(each -> keyword == null || keyword.isEmpty() || each.toLowerCase().contains(keyword.toLowerCase())).toArray();
    }

    private void addSearchCacheListener() {
        textFieldSearchCache.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                Object[] listData = queryListData(textFieldSearchCache.getText());
                listCache.setListData(listData);
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                Object[] listData = queryListData(textFieldSearchCache.getText());
                listCache.setListData(listData);
            }

            @Override
            public void changedUpdate(DocumentEvent e) {

            }
        });
    }

    private void addButtonDeleteCacheListener() {
        buttonDeleteCache.addActionListener(e -> {
            String cache = (String) listCache.getSelectedValue();
            if (cache != null) {
                eventManagement.putEvent(new DeleteFromCacheEvent(cache));
                cacheSet.remove(cache);
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

    /**
     * 点击执行数据库VACUUM
     */
    private void addButtonVacuumListener() {
        buttonVacuum.addActionListener(e -> {
            int ret = JOptionPane.showConfirmDialog(frame, TRANSLATE_SERVICE.getTranslation("Confirm whether to start optimizing the database?"));
            if (JOptionPane.YES_OPTION == ret) {
                Constants.Enums.DatabaseStatus status = DatabaseService.getInstance().getStatus();
                if (status == Constants.Enums.DatabaseStatus.NORMAL) {
                    if (IsDebug.isDebug()) {
                        System.out.println("开始优化");
                    }
                    eventManagement.putEvent(new OptimiseDatabaseEvent());
                    cachedThreadPoolUtil.executeTask(() -> {
                        //实时显示VACUUM状态
                        try {
                            DatabaseService instance = DatabaseService.getInstance();
                            while (instance.getStatus() == Constants.Enums.DatabaseStatus.VACUUM) {
                                labelVacuumStatus.setText(TRANSLATE_SERVICE.getTranslation("Optimizing..."));
                                TimeUnit.MILLISECONDS.sleep(50);
                            }
                            labelVacuumStatus.setText(TRANSLATE_SERVICE.getTranslation("Optimized"));
                            TimeUnit.SECONDS.sleep(3);
                            labelVacuumStatus.setText("");
                        } catch (InterruptedException ignored) {
                        }
                    });
                } else if (status == Constants.Enums.DatabaseStatus.MANUAL_UPDATE) {
                    JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Database is not usable yet, please wait..."));
                } else if (status == Constants.Enums.DatabaseStatus.VACUUM) {
                    JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Task is still running."));
                }
            }
        });
    }

    /**
     * 搜索自定义命令
     */
    private void addTextFieldSearchCommandsListener() {
        class search {
            final String searchText;
            final CachedThreadPoolUtil cachedThreadPoolUtil = CachedThreadPoolUtil.getInstance();
            final HashSet<String> cmdSetTmp = new HashSet<>();

            search(String searchText) {
                this.searchText = searchText;
            }

            void doSearch() {
                cachedThreadPoolUtil.executeTask(() -> {
                    String tmp;
                    if ((tmp = searchText) == null || tmp.isEmpty()) {
                        listCmds.setListData(allConfigs.getCmdSet().toArray());
                    } else {
                        for (String each : allConfigs.getCmdSet()) {
                            if (each.toLowerCase().contains(tmp.toLowerCase())) {
                                cmdSetTmp.add(each);
                            }
                        }
                        listCmds.setListData(cmdSetTmp.toArray());
                    }
                });
            }
        }

        textFieldSearchCommands.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                new search(textFieldSearchCommands.getText()).doSearch();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                new search(textFieldSearchCommands.getText()).doSearch();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
            }
        });
    }

    private static class PreviewStatus {
        private static boolean isPreview = false;
    }

    private void addButtonClosePreviewListener() {
        buttonClosePreview.addActionListener(e -> {
            eventManagement.putEvent(new StopPreviewEvent());
            eventManagement.putEvent(new HideSearchBarEvent());
            PreviewStatus.isPreview = false;
        });
    }

    private void addButtonPreviewListener() {
        buttonPreviewColor.addActionListener(e -> {
            PreviewStatus.isPreview = true;
            eventManagement.putEvent(new StartPreviewEvent());
            cachedThreadPoolUtil.executeTask(() -> {
                try {
                    String borderColor;
                    String searchBarColor;
                    String searchBarFontColor;
                    String labelColor;
                    String fontColorCoverage;
                    String fontColor;
                    String defaultBackgroundColor;
                    Constants.Enums.BorderType borderType;
                    String borderThickness;
                    while (PreviewStatus.isPreview && eventManagement.notMainExit()) {
                        borderColor = textFieldBorderColor.getText();
                        searchBarColor = textFieldSearchBarColor.getText();
                        searchBarFontColor = textFieldSearchBarFontColor.getText();
                        labelColor = textFieldLabelColor.getText();
                        fontColorCoverage = textFieldFontColorWithCoverage.getText();
                        fontColor = textFieldFontColor.getText();
                        defaultBackgroundColor = textFieldBackgroundDefault.getText();
                        borderThickness = textFieldBorderThickness.getText();
                        borderType = (Constants.Enums.BorderType) comboBoxBorderType.getSelectedItem();
                        if (canParseToRGB(borderColor) && canParseToRGB(searchBarColor) && canParseToRGB(searchBarFontColor) && canParseToRGB(labelColor) && canParseToRGB(fontColorCoverage) && canParseToRGB(fontColor) && canParseToRGB(defaultBackgroundColor) && canParseInteger(borderThickness, 1, 5)) {
                            eventManagement.putEvent(new PreviewSearchBarEvent(borderColor, searchBarColor, searchBarFontColor, labelColor, fontColorCoverage, fontColor, defaultBackgroundColor, borderType, borderThickness));
                        }
                        TimeUnit.SECONDS.sleep(1);
                    }
                } catch (InterruptedException ignored) {
                }
            });
        });
    }

    /**
     * 删除所有缓存
     */
    private void addButtonDeleteAllCacheListener() {
        buttonDeleteAllCache.addActionListener(e -> {
            int ret = JOptionPane.showConfirmDialog(frame, TRANSLATE_SERVICE.getTranslation("The operation is irreversible. Are you sure you want to clear the cache?"));
            if (JOptionPane.YES_OPTION == ret) {
                for (String each : cacheSet) {
                    eventManagement.putEvent(new DeleteFromCacheEvent(each));
                }
                cacheSet.clear();
                listCache.setListData(cacheSet.toArray());
            }
        });
    }

    private void addButtonViewPluginMarketListener() {
        buttonPluginMarket.addActionListener(e -> eventManagement.putEvent(new ShowPluginMarket()));
    }

    /**
     * 将diskSet中的字符串转为用逗号隔开的字符串
     *
     * @return string
     */
    private String parseDisk() {
        return String.join(",", diskSet);
    }

    @SuppressWarnings("SuspiciousMethodCalls")
    private void addButtonDeleteDiskListener() {
        buttonDeleteDisk.addActionListener(e -> {
            List<Object> selectedValuesList = listDisks.getSelectedValuesList();
            for (Object obj : selectedValuesList) {
                diskSet.remove(obj);
            }
            listDisks.setListData(diskSet.toArray());
        });
    }

    /**
     * 点击重建索引
     */
    private void addButtonRebuildListener() {
        buttonRebuildIndex.addActionListener(e -> {
            eventManagement.putEvent(new ShowTaskBarMessageEvent(TRANSLATE_SERVICE.getTranslation("Info"), TRANSLATE_SERVICE.getTranslation("Updating file index")));
            eventManagement.putEvent(new UpdateDatabaseEvent(false), event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(TranslateService.getInstance().getTranslation("Info"), TranslateService.getInstance().getTranslation("Search Done"))), event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(TranslateService.getInstance().getTranslation("Warning"), TranslateService.getInstance().getTranslation("Search Failed"))));
        });
    }

    /**
     * 点击添加硬盘
     */
    private void addButtonAddDiskListener() {
        buttonAddNewDisk.addActionListener(e -> {
            File[] disks = File.listRoots();
            ArrayList<String> arraylistDisks = new ArrayList<>();
            for (File each : disks) {
                if (IsLocalDisk.INSTANCE.isDiskNTFS(each.getAbsolutePath())) {
                    arraylistDisks.add(each.getAbsolutePath());
                }
            }
            JList<Object> listDisksTmp = new JList<>();
            listDisksTmp.setListData(arraylistDisks.toArray());
            JScrollPane pane = new JScrollPane(listDisksTmp);
            Dimension dimension = new Dimension(400, 200);
            pane.setPreferredSize(dimension);
            if (JOptionPane.showConfirmDialog(frame, pane, TRANSLATE_SERVICE.getTranslation("Select disk"), JOptionPane.YES_NO_OPTION) == JOptionPane.YES_OPTION) {
                List<Object> selectedValuesList = listDisksTmp.getSelectedValuesList();
                for (Object obj : selectedValuesList) {
                    diskSet.add((String) obj);
                }
                listDisks.setListData(diskSet.toArray());
            }
        });
    }

    private void addSwingThemePreviewListener() {
        listSwingThemes.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                String swingTheme = (String) listSwingThemes.getSelectedValue();
                eventManagement.putEvent(new SetSwingLaf(swingTheme));
            }
        });
    }

    private void addButtonDeleteAllSuffixListener() {
        buttonDeleteAllSuffix.addActionListener(e -> {
            int ret = JOptionPane.showConfirmDialog(frame, TRANSLATE_SERVICE.getTranslation("Are you sure to delete all the suffixes") + "?");
            if (ret == JOptionPane.YES_OPTION) {
                suffixMap.clear();
                suffixMap.put("defaultPriority", 0);
                eventManagement.putEvent(new ClearSuffixPriorityMapEvent());
                refreshPriorityTable();
            }
        });
    }

    private void showOnTabbedPane(String tabName) {
        String title = TRANSLATE_SERVICE.getTranslation(getTabTitle(tabName));
        showOnTabbedPane(tabName, title);
    }

    private void showOnTabbedPane(String tabName, String tabTitle) {
        tabbedPane.addTab(tabTitle, getTabComponent(tabName));
        tabbedPane.setSelectedIndex(0);
    }

    private String getTabTitle(String tabName) {
        for (TabNameAndTitle each : tabComponentNameMap.keySet()) {
            if (each.tabName.equals(tabName)) {
                return each.title;
            }
        }
        return "";
    }

    private void addTreeSettingsListener() {
        //todo 注册tab后在这里添加响应
        treeSettings.addTreeSelectionListener(e -> {
            tabbedPane.removeAll();
            DefaultMutableTreeNode note = (DefaultMutableTreeNode) treeSettings.getLastSelectedPathComponent();
            if (note != null) {
                String name = note.toString();
                if (TRANSLATE_SERVICE.getTranslation("General").equals(name)) {
                    showOnTabbedPane("tabGeneral");
                } else if (TRANSLATE_SERVICE.getTranslation("Interface").equals(name)) {
                    showOnTabbedPane("tabSearchBarSettings");
                } else if (TRANSLATE_SERVICE.getTranslation("Language").equals(name)) {
                    showOnTabbedPane("tabLanguage");
                } else if (TRANSLATE_SERVICE.getTranslation("Suffix priority").equals(name)) {
                    showOnTabbedPane("tabModifyPriority");
                } else if (TRANSLATE_SERVICE.getTranslation("Search settings").equals(name)) {
                    showOnTabbedPane("tabSearchSettings");
                } else if (TRANSLATE_SERVICE.getTranslation("Proxy settings").equals(name)) {
                    showOnTabbedPane("tabProxy");
                } else if (TRANSLATE_SERVICE.getTranslation("Hotkey settings").equals(name)) {
                    showOnTabbedPane("tabHotKey");
                } else if (TRANSLATE_SERVICE.getTranslation("Cache").equals(name)) {
                    showOnTabbedPane("tabCache");
                } else if (TRANSLATE_SERVICE.getTranslation("My commands").equals(name)) {
                    showOnTabbedPane("tabCommands");
                } else if (TRANSLATE_SERVICE.getTranslation("Plugins").equals(name)) {
                    showOnTabbedPane("tabPlugin");
                } else if (TRANSLATE_SERVICE.getTranslation("About").equals(name)) {
                    showOnTabbedPane("tabAbout");
                } else if (TRANSLATE_SERVICE.getTranslation("Index").equals(name)) {
                    showOnTabbedPane("tabIndex");
                } else {
                    showOnTabbedPane("tabGeneral");
                }
            } else {
                showOnTabbedPane("tabGeneral", TRANSLATE_SERVICE.getTranslation("General"));
            }
        });
    }

    /**
     * 刷新后缀优先级显示
     */
    private void refreshPriorityTable() {
        SwingUtilities.invokeLater(this::setTableGui);
    }

    private boolean isPriorityRepeat(int num) {
        for (int each : suffixMap.values()) {
            if (each == num) {
                return true;
            }
        }
        return false;
    }

    private boolean isSuffixRepeat(String suffix) {
        return suffixMap.containsKey(suffix);
    }

    /**
     * 检查后缀优先级的设置
     *
     * @param suffix            后缀
     * @param priority          优先级
     * @param errMsg            存储错误信息
     * @param isSuffixChanged   是否后缀修改
     * @param isPriorityChanged 是否优先级修改
     * @return true检查成功
     */
    private boolean checkSuffixAndPriority(String suffix, String priority, StringBuilder errMsg, boolean isSuffixChanged, boolean isPriorityChanged) {
        if (isSuffixChanged) {
            if (isSuffixRepeat(suffix)) {
                errMsg.append(TRANSLATE_SERVICE.getTranslation("Duplicate suffix, please check")).append("\n");
            }
        }
        if (isPriorityChanged) {
            try {
                int _p = Integer.parseInt(priority);
                if (_p <= 0) {
                    errMsg.append(TRANSLATE_SERVICE.getTranslation("Priority num must be positive")).append("\n");
                }
                if (isPriorityRepeat(_p)) {
                    errMsg.append(TRANSLATE_SERVICE.getTranslation("Duplicate priority num, please check")).append("\n");
                }
            } catch (NumberFormatException e) {
                errMsg.append(TRANSLATE_SERVICE.getTranslation("What you entered is not a number, please try again")).append("\n");
            }
        }
        return errMsg.toString().isEmpty();
    }

    private void addTableModifySuffixListener() {
        final String[] lastSuffix = new String[1];
        AtomicInteger lastNum = new AtomicInteger();
        AtomicBoolean dontTrigger = new AtomicBoolean(false);

        DefaultTableModel model = new DefaultTableModel() {
            @Override
            public boolean isCellEditable(int row, int column) {
                return row != 0;
            }
        };
        tableSuffix.setModel(model);
        tableSuffix.getTableHeader().setReorderingAllowed(false);

        tableSuffix.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                //保存还未被修改之前的值
                int row = tableSuffix.getSelectedRow();
                int column = tableSuffix.getSelectedColumn();
                if (row != -1 && column != -1) {
                    lastSuffix[0] = String.valueOf(tableSuffix.getValueAt(row, 0));
                    lastNum.set(Integer.parseInt(String.valueOf(tableSuffix.getValueAt(row, 1))));
                }
            }
        });

        tableSuffix.getModel().addTableModelListener(e -> {
            if (dontTrigger.get()) {
                return;
            }
            int currentRow = tableSuffix.getSelectedRow();
            if (currentRow == -1) {
                return;
            }
            class restoreUtil {
                void doRestoreNum() {
                    // 恢复
                    SwingUtilities.invokeLater(() -> {
                        dontTrigger.set(true);
                        tableSuffix.setValueAt(lastNum.get(), currentRow, 1);
                        dontTrigger.set(false);
                    });
                }

                void doRestoreSuffix() {
                    SwingUtilities.invokeLater(() -> {
                        dontTrigger.set(true);
                        tableSuffix.setValueAt(lastSuffix[0], currentRow, 0);
                        dontTrigger.set(false);
                    });
                }
            }
            restoreUtil util = new restoreUtil();

            int column = tableSuffix.getSelectedColumn();
            if (column == 0) {
                //当前修改的是后缀
                String suffix = String.valueOf(tableSuffix.getValueAt(currentRow, 0));
                if (lastSuffix[0].equals(suffix)) {
                    return;
                }
                String priorityNum = String.valueOf(tableSuffix.getValueAt(currentRow, 1));
                StringBuilder errMsg = new StringBuilder();
                if (checkSuffixAndPriority(suffix, priorityNum, errMsg, true, false)) {
                    int num = Integer.parseInt(priorityNum);
                    eventManagement.putEvent(new UpdateSuffixPriorityEvent(lastSuffix[0], suffix, num));
                    suffixMap.remove(lastSuffix[0]);
                    suffixMap.put(suffix, num);
                    refreshPriorityTable();
                } else {
                    // 恢复
                    util.doRestoreSuffix();
                    JOptionPane.showMessageDialog(frame, errMsg.toString());
                }
            } else {
                //当前修改的是优先级
                String priorityNum = String.valueOf(tableSuffix.getValueAt(currentRow, 1));
                if (String.valueOf(lastNum.get()).equals(priorityNum)) {
                    return;
                }
                String suffix = String.valueOf(tableSuffix.getValueAt(currentRow, 0));
                StringBuilder errMsg = new StringBuilder();
                if (checkSuffixAndPriority(suffix, priorityNum, errMsg, false, true)) {
                    int num = Integer.parseInt(priorityNum);
                    eventManagement.putEvent(new UpdateSuffixPriorityEvent(lastSuffix[0], suffix, num));
                    suffixMap.remove(lastSuffix[0]);
                    suffixMap.put(suffix, num);
                    refreshPriorityTable();
                } else {
                    util.doRestoreNum();
                    JOptionPane.showMessageDialog(frame, errMsg.toString());
                }
            }
        });
    }

    private String getCurrentSelectedTableVal() {
        int row = tableSuffix.getSelectedRow();
        int column = tableSuffix.getSelectedColumn();
        if (row == -1 || column == -1) {
            return "";
        }
        return String.valueOf(tableSuffix.getValueAt(row, 0));
    }

    private void addButtonDeleteSuffixListener() {
        buttonDeleteSuffix.addActionListener(e -> {
            String current = getCurrentSelectedTableVal();
            if (current.isEmpty() || "defaultPriority".equals(current)) {
                return;
            }
            int ret = JOptionPane.showConfirmDialog(frame, TRANSLATE_SERVICE.getTranslation("Do you sure want to delete this suffix") + "  --  " + current + "?");
            if (ret == JOptionPane.YES_OPTION) {
                int rowNum = tableSuffix.getSelectedRow();
                if (rowNum != -1) {
                    String suffix = (String) tableSuffix.getValueAt(rowNum, 0);
                    suffixMap.remove(suffix);
                    eventManagement.putEvent(new DeleteFromSuffixPriorityMapEvent(suffix));
                    refreshPriorityTable();
                }
            }
        });
    }

    private void addButtonAddSuffixListener() {
        JPanel panel = new JPanel();
        JLabel labelSuffix = new JLabel(TRANSLATE_SERVICE.getTranslation("Suffix") + ":");
        JLabel labelNum = new JLabel(TRANSLATE_SERVICE.getTranslation("Priority num") + ":");
        JTextField suffixName = new JTextField();
        JTextField priorityNum = new JTextField();

        Box suffixBox = new Box(BoxLayout.X_AXIS);
        suffixBox.add(labelSuffix);
        suffixBox.add(suffixName);

        Box numBox = new Box(BoxLayout.X_AXIS);
        numBox.add(labelNum);
        numBox.add(priorityNum);

        panel.add(suffixBox);
        panel.add(numBox);
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        buttonAddSuffix.addActionListener(e -> {
            int ret = JOptionPane.showConfirmDialog(frame, panel, TRANSLATE_SERVICE.getTranslation("Add"), JOptionPane.YES_NO_OPTION);
            if (ret == JOptionPane.YES_OPTION) {
                String suffix = suffixName.getText();
                String priorityNumTmp = priorityNum.getText();
                StringBuilder err = new StringBuilder();
                if (checkSuffixAndPriority(suffix, priorityNumTmp, err, true, true)) {
                    int num = Integer.parseInt(priorityNumTmp);
                    suffixMap.put(suffix, num);
                    eventManagement.putEvent(new AddToSuffixPriorityMapEvent(suffix, num));
                    refreshPriorityTable();
                } else {
                    JOptionPane.showMessageDialog(frame, err.toString());
                }
            }
        });
    }

    /**
     * 等待检查更新
     *
     * @param startCheckTime    开始检查时间
     * @param checkUpdateThread 检查线程
     *                          0x100L 表示检查成功
     *                          0xFFFL 表示检查失败
     */
    private void waitForCheckUpdateResult(AtomicLong startCheckTime, Thread checkUpdateThread) {
        try {
            while (startCheckTime.get() != 0x100L) {
                TimeUnit.MILLISECONDS.sleep(200);
                if ((System.currentTimeMillis() - startCheckTime.get() > 5000L && startCheckTime.get() != 0x100L) || startCheckTime.get() == 0xFFFL) {
                    checkUpdateThread.interrupt();
                    JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Check update failed"));
                    return;
                }
                if (!eventManagement.notMainExit()) {
                    return;
                }
            }
        } catch (InterruptedException e1) {
            e1.printStackTrace();
        }
    }

    private void addButtonPluginUpdateCheckListener() {
        AtomicLong startCheckTime = new AtomicLong(0L);
        AtomicBoolean isVersionLatest = new AtomicBoolean(true);
        AtomicBoolean isSkipConfirm = new AtomicBoolean(false);

        HashMap<String, DownloadManager> pluginInfoMap = new HashMap<>();
        PluginService pluginService = PluginService.getInstance();

        buttonUpdatePlugin.addActionListener(e -> {
            String pluginName = (String) listPlugins.getSelectedValue();
            var downloadManagerContainer = new Object() {
                DownloadManager downloadManager = pluginInfoMap.get(pluginName);
            };
            if (downloadManagerContainer.downloadManager != null && downloadManagerContainer.downloadManager.getDownloadStatus() == Constants.Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                eventManagement.putEvent(new StopDownloadEvent(downloadManagerContainer.downloadManager));
            } else {
                startCheckTime.set(0L);
                GetPluginByNameEvent getPluginByNameEvent = new GetPluginByNameEvent(pluginName);
                eventManagement.putEvent(getPluginByNameEvent);
                eventManagement.waitForEvent(getPluginByNameEvent);
                Optional<PluginService.PluginInfo> pluginInfoOptional = getPluginByNameEvent.getReturnValue();
                pluginInfoOptional.ifPresent(pluginInfo -> {
                    Plugin plugin = pluginInfo.plugin;
                    String pluginFullName = pluginName + ".jar";

                    if (pluginService.hasPluginNotLatest(pluginName)) {
                        //已经检查过
                        isVersionLatest.set(false);
                        isSkipConfirm.set(true);
                    } else {
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
                        cachedThreadPoolUtil.executeTask(checkUpdateThread);
                        //等待获取插件更新信息
                        waitForCheckUpdateResult(startCheckTime, checkUpdateThread);
                    }

                    if (isVersionLatest.get()) {
                        JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Latest version:") + plugin.getVersion() + "\n" + TRANSLATE_SERVICE.getTranslation("The current version is the latest"));
                        return;
                    }
                    if (!isSkipConfirm.get()) {
                        eventManagement.putEvent(new AddPluginsCanUpdateEvent(pluginName));
                        int ret = JOptionPane.showConfirmDialog(frame, TRANSLATE_SERVICE.getTranslation("New version available, do you want to update?"));
                        if (ret != JOptionPane.YES_OPTION) {
                            return;
                        }
                    }
                    //开始下载
                    String url = plugin.getUpdateURL();
                    downloadManagerContainer.downloadManager = new DownloadManager(url, pluginFullName, new File("tmp", "pluginsUpdate").getAbsolutePath());
                    eventManagement.putEvent(new StartDownloadEvent(downloadManagerContainer.downloadManager));
                    pluginInfoMap.put(pluginName, downloadManagerContainer.downloadManager);
                    DownloadManager finalDownloadManager = downloadManagerContainer.downloadManager;
                    cachedThreadPoolUtil.executeTask(() ->
                            SetDownloadProgress.setProgress(labelProgress,
                                    buttonUpdatePlugin,
                                    finalDownloadManager,
                                    () -> finalDownloadManager.fileName.equals(listPlugins.getSelectedValue() + ".jar"),
                                    new File("user/updatePlugin"),
                                    "",
                                    null,
                                    null));
                });
            }
        });
    }

    private void setLabelGui() {
        labelAboutGithub.setText("<html><a href='https://github.com/XUANXUQAQ/File-Engine'><font size=\"4\">File-Engine</font></a></html>");
        labelWebLookAndFeel.setText("1.JFormDesigner/FlatLaf");
        labelFastJson.setText("2.google/gson");
        labelJna.setText("3.java-native-access/jna");
        labelSQLite.setText("4.xerial/sqlite-jdbc");
        labelTinyPinyin.setText("5.promeG/TinyPinyin");
        labelLombok.setText("6.projectlombok/Lombok");
        labelZip.setText("7.kuba--/zip");
        labelPluginNum.setText(String.valueOf(PluginService.getInstance().getInstalledPluginNum()));
        ImageIcon imageIcon = new ImageIcon(Objects.requireNonNull(SettingsFrame.class.getResource("/icons/frame.png")));
        labelIcon.setIcon(imageIcon);
        labelVersion.setText(TRANSLATE_SERVICE.getTranslation("Current Version:") + Constants.version);
        labelCurrentCacheNum.setText(TRANSLATE_SERVICE.getTranslation("Current Caches Num:") + DatabaseService.getInstance().getCacheNum());
    }

    /**
     * 初始化textField的显示
     */
    private void setTextFieldAndTextAreaGui() {
        textFieldSearchCache.setText("");
        textFieldBackgroundDefault.setText(toRGBHexString(allConfigs.getDefaultBackgroundColor()));
        textFieldLabelColor.setText(toRGBHexString(allConfigs.getLabelColor()));
        textFieldFontColorWithCoverage.setText(toRGBHexString(allConfigs.getLabelFontColorWithCoverage()));
        textFieldTransparency.setText(String.valueOf(allConfigs.getOpacity()));
        textFieldBorderColor.setText(toRGBHexString(allConfigs.getBorderColor()));
        textFieldFontColor.setText(toRGBHexString(allConfigs.getLabelFontColor()));
        textFieldSearchBarFontColor.setText(toRGBHexString(allConfigs.getSearchBarFontColor()));
        textFieldCacheNum.setText(String.valueOf(allConfigs.getCacheNumLimit()));
        textFieldHotkey.setText(allConfigs.getHotkey());
        textFieldRoundRadius.setText(String.valueOf(allConfigs.getRoundRadius()));
        textFieldPriorityFolder.setText(allConfigs.getPriorityFolder());
        textFieldUpdateInterval.setText(String.valueOf(allConfigs.getUpdateTimeLimit()));
        textFieldSearchBarColor.setText(toRGBHexString(allConfigs.getSearchBarColor()));
        textFieldAddress.setText(allConfigs.getProxyAddress());
        textFieldPort.setText(String.valueOf(allConfigs.getProxyPort()));
        textFieldUserName.setText(allConfigs.getProxyUserName());
        textFieldPassword.setText(allConfigs.getProxyPassword());
        textFieldBorderThickness.setText(String.valueOf(allConfigs.getBorderThickness()));
        textAreaIgnorePath.setText(RegexUtil.comma.matcher(allConfigs.getIgnorePath()).replaceAll(",\n"));
        if (allConfigs.getRunAsAdminKeyCode() == 17) {
            textFieldRunAsAdminHotKey.setText("Ctrl + Enter");
        } else if (allConfigs.getRunAsAdminKeyCode() == 16) {
            textFieldRunAsAdminHotKey.setText("Shift + Enter");
        } else if (allConfigs.getRunAsAdminKeyCode() == 18) {
            textFieldRunAsAdminHotKey.setText("Alt + Enter");
        }
        if (allConfigs.getOpenLastFolderKeyCode() == 17) {
            textFieldOpenLastFolder.setText("Ctrl + Enter");
        } else if (allConfigs.getOpenLastFolderKeyCode() == 16) {
            textFieldOpenLastFolder.setText("Shift + Enter");
        } else if (allConfigs.getOpenLastFolderKeyCode() == 18) {
            textFieldOpenLastFolder.setText("Alt + Enter");
        }
        if (allConfigs.getCopyPathKeyCode() == 17) {
            textFieldCopyPath.setText("Ctrl + Enter");
        } else if (allConfigs.getCopyPathKeyCode() == 16) {
            textFieldCopyPath.setText("Shift + Enter");
        } else if (allConfigs.getCopyPathKeyCode() == 18) {
            textFieldCopyPath.setText("Alt + Enter");
        }
    }

    /**
     * 初始化颜色选择器的显示
     */
    private void setColorChooserGui() {
        Color tmp_searchBarColor = new Color(allConfigs.getSearchBarColor());
        searchBarColorChooser.setBackground(tmp_searchBarColor);
        searchBarColorChooser.setForeground(tmp_searchBarColor);

        Color tmp_defaultBackgroundColor = new Color(allConfigs.getDefaultBackgroundColor());
        defaultBackgroundChooser.setBackground(tmp_defaultBackgroundColor);
        defaultBackgroundChooser.setForeground(tmp_defaultBackgroundColor);

        Color tmp_labelColor = new Color(allConfigs.getLabelColor());
        labelColorChooser.setBackground(tmp_labelColor);
        labelColorChooser.setForeground(tmp_labelColor);

        Color tmp_fontColorWithCoverage = new Color(allConfigs.getLabelFontColorWithCoverage());
        FontColorWithCoverageChooser.setBackground(tmp_fontColorWithCoverage);
        FontColorWithCoverageChooser.setForeground(tmp_fontColorWithCoverage);

        Color tmp_fontColor = new Color(allConfigs.getLabelFontColor());
        FontColorChooser.setBackground(tmp_fontColor);
        FontColorChooser.setForeground(tmp_fontColor);

        Color tmp_searchBarFontColor = new Color(allConfigs.getSearchBarFontColor());
        SearchBarFontColorChooser.setBackground(tmp_searchBarFontColor);
        SearchBarFontColorChooser.setForeground(tmp_searchBarFontColor);

        Color tmp_borderColor = new Color(allConfigs.getBorderColor());
        borderColorChooser.setBackground(tmp_borderColor);
        borderColorChooser.setForeground(tmp_borderColor);
    }

    /**
     * 根据后缀优先级获取后缀
     *
     * @param suffixPriorityMap 优先级表
     * @param val               val
     * @return key
     */
    private String getSuffixByValue(HashMap<String, Integer> suffixPriorityMap, int val) {
        for (String each : suffixPriorityMap.keySet()) {
            if (suffixPriorityMap.get(each) == val) {
                return each;
            }
        }
        return "";
    }

    /**
     * 设置所有表的显示
     */
    private void setTableGui() {
        tableSuffix.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        DefaultTableModel tableModel = (DefaultTableModel) tableSuffix.getModel();    //获得表格模型
        tableModel.setRowCount(0);
        tableModel.setColumnIdentifiers(new String[]{TRANSLATE_SERVICE.getTranslation("suffix"), TRANSLATE_SERVICE.getTranslation("priority")});
        LinkedList<Integer> tmpKeySet = new LinkedList<>(suffixMap.values());
        tmpKeySet.sort(Integer::compare);
        for (int each : tmpKeySet) {
            String suffix = getSuffixByValue(suffixMap, each);
            if (suffix.isEmpty()) {
                continue;
            }
            String[] data = new String[2];
            data[0] = suffix;
            data[1] = String.valueOf(each);
            tableModel.addRow(data);
        }
    }

    /**
     * 初始化所有选择栏的显示
     */
    private void setCheckBoxGui() {
        checkBoxLoseFocus.setSelected(allConfigs.isLoseFocusClose());
        int startup = hasStartup();
        if (startup == 1) {
            eventManagement.putEvent(new ShowTaskBarMessageEvent(TRANSLATE_SERVICE.getTranslation("Warning"), TRANSLATE_SERVICE.getTranslation("The startup path is invalid")));
        }
        checkBoxAddToStartup.setSelected(startup == 0);
        checkBoxAdmin.setSelected(allConfigs.isDefaultAdmin());
        checkBoxIsShowTipOnCreatingLnk.setSelected(allConfigs.isShowTipOnCreatingLnk());
        checkBoxResponseCtrl.setSelected(allConfigs.isResponseCtrl());
        checkBoxCheckUpdate.setSelected(allConfigs.isCheckUpdateStartup());
        checkBoxIsAttachExplorer.setSelected(allConfigs.isAttachExplorer());
    }

    /**
     * 初始化所有列表的显示
     */
    private void setListGui() {
        listCmds.setListData(allConfigs.getCmdSet().toArray());
        listLanguage.setListData(TRANSLATE_SERVICE.getLanguageArray());
        listLanguage.setSelectedValue(TRANSLATE_SERVICE.getLanguage(), true);
        Object[] plugins = PluginService.getInstance().getPluginNameArray();
        listPlugins.setListData(plugins);
        listCache.setListData(cacheSet.toArray());
        ArrayList<String> list = new ArrayList<>();
        for (Constants.Enums.SwingThemes each : Constants.Enums.SwingThemes.values()) {
            list.add(each.toString());
        }
        listSwingThemes.setListData(list.toArray());
        listSwingThemes.setSelectedValue(allConfigs.getSwingTheme(), true);
        listDisks.setListData(diskSet.toArray());
    }

    /**
     * 初始化后缀名map
     */
    private void initSuffixMap() {
        try (Statement stmt = SQLiteUtil.getStatement("cache"); ResultSet resultSet = stmt.executeQuery("SELECT * FROM priority;")) {
            while (resultSet.next()) {
                suffixMap.put(resultSet.getString("SUFFIX"), resultSet.getInt("PRIORITY"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    /**
     * 获取tab中最长的字符串
     *
     * @return 最长的字符串
     */
    private String getLongestTitle() {
        String longest = "";
        String realTitle;
        for (TabNameAndTitle each : tabComponentNameMap.keySet()) {
            realTitle = TRANSLATE_SERVICE.getTranslation(each.title);
            if (longest.length() < realTitle.length()) {
                longest = realTitle;
            }
        }
        return longest;
    }

    /**
     * 重设gui大小
     */
    private void resizeGUI() {
        int fontSize = treeSettings.getFont().getSize() / 96 * 72;
        String longestTitle = getLongestTitle();
        int length = longestTitle.length();
        int width = length * fontSize;
        Dimension treeSize = new Dimension(width, -1);
        treeSettings.setMaximumSize(treeSize);
        treeSettings.setMinimumSize(treeSize);
        treeSettings.setPreferredSize(treeSize);
        Dimension tabbedPaneSize = new Dimension(Integer.parseInt(TRANSLATE_SERVICE.getFrameWidth()) - width, -1);
        tabbedPane.setMaximumSize(tabbedPaneSize);
        tabbedPane.setMinimumSize(tabbedPaneSize);
        tabbedPane.setPreferredSize(tabbedPaneSize);
    }

    /**
     * 初始化所有UI
     */
    private void initGUI() {
        //设置窗口显示
        setLabelGui();
        setListGui();
        setColorChooserGui();
        setTextFieldAndTextAreaGui();
        setCheckBoxGui();
        setTableGui();
        initTreeSettings();
        resizeGUI();

        tabbedPane.removeAll();
        tabbedPane.setBackground(new Color(0, 0, 0, 0));

        buttonUpdatePlugin.setVisible(false);

        if (allConfigs.getProxyType() == Constants.Enums.ProxyType.PROXY_DIRECT) {
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
        chooseUpdateAddress.setSelectedItem(allConfigs.getUpdateAddress());
        comboBoxBorderType.setSelectedItem(allConfigs.getBorderType());
    }

    /**
     * 初始化cache
     */
    private void initCacheArray() {
        String eachLine;
        try (Statement statement = SQLiteUtil.getStatement("cache"); ResultSet resultSet = statement.executeQuery("SELECT PATH FROM cache;")) {
            while (resultSet.next()) {
                eachLine = resultSet.getString("PATH");
                cacheSet.add(eachLine);
            }
        } catch (Exception throwables) {
            throwables.printStackTrace();
        }
    }

    /**
     * 初始化代理选择框
     */
    private void selectProxyType() {
        if (allConfigs.getProxyType() == Constants.Enums.ProxyType.PROXY_SOCKS) {
            radioButtonProxyTypeSocks5.setSelected(true);
        } else {
            radioButtonProxyTypeHttp.setSelected(true);
        }
    }

    private void initDiskSet() {
        String[] disks = RegexUtil.comma.split(allConfigs.getDisks());
        diskSet.addAll(Arrays.asList(disks));
    }

    private void initTabNameMap() {
        //todo 添加新tab后在这里注册
        tabComponentNameMap.put(new TabNameAndTitle("tabGeneral", "General"), tabGeneral);
        tabComponentNameMap.put(new TabNameAndTitle("tabSearchSettings", "Search settings"), tabSearchSettings);
        tabComponentNameMap.put(new TabNameAndTitle("tabSearchBarSettings", "Interface"), tabSearchBarSettings);
        tabComponentNameMap.put(new TabNameAndTitle("tabModifyPriority", "Modify suffix priority"), tabModifyPriority);
        tabComponentNameMap.put(new TabNameAndTitle("tabCache", "Cache"), tabCache);
        tabComponentNameMap.put(new TabNameAndTitle("tabProxy", "Proxy settings"), tabProxy);
        tabComponentNameMap.put(new TabNameAndTitle("tabPlugin", "Plugins"), tabPlugin);
        tabComponentNameMap.put(new TabNameAndTitle("tabHotKey", "Hotkey settings"), tabHotKey);
        tabComponentNameMap.put(new TabNameAndTitle("tabLanguage", "language"), tabLanguage);
        tabComponentNameMap.put(new TabNameAndTitle("tabCommands", "My commands"), tabCommands);
        tabComponentNameMap.put(new TabNameAndTitle("tabAbout", "About"), tabAbout);
        tabComponentNameMap.put(new TabNameAndTitle("tabIndex", "Index"), tabIndex);

        excludeComponent.addAll(tabComponentNameMap.values());
    }

    private void initTreeSettings() {
        //todo 添加新tab后在这里注册
        DefaultMutableTreeNode groupGeneral = new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("General"));
        groupGeneral.add(new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("Interface")));
        groupGeneral.add(new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("Language")));
        groupGeneral.add(new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("Cache")));

        DefaultMutableTreeNode groupSearchSettings = new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("Search settings"));
        groupSearchSettings.add(new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("Suffix priority")));
        groupSearchSettings.add(new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("My commands")));
        groupSearchSettings.add(new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("Index")));

        DefaultMutableTreeNode groupProxy = new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("Proxy settings"));

        DefaultMutableTreeNode groupHotkey = new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("Hotkey settings"));

        DefaultMutableTreeNode groupPlugin = new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("Plugins"));

        DefaultMutableTreeNode groupAbout = new DefaultMutableTreeNode(TRANSLATE_SERVICE.getTranslation("About"));

        DefaultMutableTreeNode root = new DefaultMutableTreeNode();
        root.add(groupGeneral);
        root.add(groupSearchSettings);
        root.add(groupProxy);
        root.add(groupHotkey);
        root.add(groupPlugin);
        root.add(groupAbout);
        treeSettings.setModel(new DefaultTreeModel(root));
        treeSettings.setRootVisible(false);
        expandAll(treeSettings, new TreePath(root), true);
    }

    /**
     * 展开所有设置
     *
     * @param tree   tree
     * @param parent parent
     * @param expand 是否展开
     */
    private void expandAll(JTree tree, TreePath parent, boolean expand) {
        TreeNode node = (TreeNode) parent.getLastPathComponent();
        if (node.getChildCount() >= 0) {
            for (Enumeration<?> e = node.children(); e.hasMoreElements(); ) {
                TreeNode n = (TreeNode) e.nextElement();
                TreePath path = parent.pathByAddingChild(n);
                expandAll(tree, path, expand);
            }
        }
        if (expand) {
            tree.expandPath(parent);
        } else {
            tree.collapsePath(parent);
        }
    }

    private SettingsFrame() {
        frame.setUndecorated(true);
        frame.getRootPane().setWindowDecorationStyle(JRootPane.FRAME);
        frame.setIconImage(frameIcon.getImage());

        initTabNameMap();

        initDiskSet();

        panel.remove(paneSwingThemes);
        excludeComponent.add(paneSwingThemes);

        ButtonGroup proxyButtonGroup = new ButtonGroup();
        proxyButtonGroup.add(radioButtonNoProxy);
        proxyButtonGroup.add(radioButtonUseProxy);

        ButtonGroup proxyTypeButtonGroup = new ButtonGroup();
        proxyTypeButtonGroup.add(radioButtonProxyTypeHttp);
        proxyTypeButtonGroup.add(radioButtonProxyTypeSocks5);

        tmp_openLastFolderKeyCode = allConfigs.getOpenLastFolderKeyCode();
        tmp_runAsAdminKeyCode = allConfigs.getRunAsAdminKeyCode();
        tmp_copyPathKeyCode = allConfigs.getCopyPathKeyCode();

        addUpdateAddressToComboBox();

        addBorderTypeToComboBox();

        initCacheArray();

        initSuffixMap();

        translate();

        addListeners();
    }

    /**
     * 添加所有监听器
     */
    private void addListeners() {
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
        addButtonAddDiskListener();
        addButtonDeleteDiskListener();
        addButtonRebuildListener();
        addSwingThemePreviewListener();
        addButtonAddSuffixListener();
        addButtonDeleteSuffixListener();
        addTableModifySuffixListener();
        addButtonDeleteAllSuffixListener();
        addPluginOfficialSiteListener();
        addButtonVacuumListener();
        addButtonProxyListener();
        addButtonDeleteCacheListener();
        addButtonChangeThemeListener();
        addSearchCacheListener();
        addButtonDeleteAllCacheListener();
        addButtonPreviewListener();
        addButtonClosePreviewListener();
        addTextFieldSearchCommandsListener();
        addTreeSettingsListener();
    }

    /**
     * 检查缓存是否存在
     *
     * @param cache cache
     * @return boolean
     */
    private boolean isCacheExist(String cache) {
        return cacheSet.contains(cache);
    }

    /**
     * 添加缓存到cacheSet
     *
     * @param cache cache
     */
    private void addCache(String cache) {
        cacheSet.add(cache);
    }

    private void addBorderTypeToComboBox() {
        Constants.Enums.BorderType[] borderTypes = Constants.Enums.BorderType.values();
        for (Constants.Enums.BorderType each : borderTypes) {
            comboBoxBorderType.addItem(each);
        }
    }

    private void addUpdateAddressToComboBox() {
        Set<String> updateAddresses = allConfigs.getAllUpdateAddress();
        for (String each : updateAddresses) {
            chooseUpdateAddress.addItem(each);
        }
    }

    private Component getTabComponent(String componentName) {
        for (TabNameAndTitle each : tabComponentNameMap.keySet()) {
            if (each.tabName.equals(componentName)) {
                return tabComponentNameMap.get(each);
            }
        }
        return tabGeneral;
    }

    private void translateLabels() {
        labelMaxCacheNum.setText(TRANSLATE_SERVICE.getTranslation("Set the maximum number of caches:"));
        labelUpdateInterval.setText(TRANSLATE_SERVICE.getTranslation("File update detection interval:"));
        labelSecond.setText(TRANSLATE_SERVICE.getTranslation("Seconds"));
        labeltipPriorityFolder.setText(TRANSLATE_SERVICE.getTranslation("Priority search folder location (double-click to clear):"));
        labelConstIgnorePathTip.setText(TRANSLATE_SERVICE.getTranslation("Separate different paths with commas, and ignore C:\\Windows by default"));
        labelSetIgnorePathTip.setText(TRANSLATE_SERVICE.getTranslation("Set ignore folder:"));
        labelTransparency.setText(TRANSLATE_SERVICE.getTranslation("Search bar opacity:"));
        labelOpenSearchBarHotKey.setText(TRANSLATE_SERVICE.getTranslation("Open search bar:"));
        labelRunAsAdminHotKey.setText(TRANSLATE_SERVICE.getTranslation("Run as administrator:"));
        labelOpenFolderHotKey.setText(TRANSLATE_SERVICE.getTranslation("Open the parent folder:"));
        labelCopyPathHotKey.setText(TRANSLATE_SERVICE.getTranslation("Copy path:"));
        labelCmdTip2.setText(TRANSLATE_SERVICE.getTranslation("You can add custom commands here. After adding, " + "you can enter \": + your set identifier\" in the search box to quickly open"));
        labelColorTip.setText(TRANSLATE_SERVICE.getTranslation("Please enter the hexadecimal value of RGB color"));
        labelSearchBarColor.setText(TRANSLATE_SERVICE.getTranslation("Search bar Color:"));
        labelLabelColor.setText(TRANSLATE_SERVICE.getTranslation("Chosen label color:"));
        labelFontColor.setText(TRANSLATE_SERVICE.getTranslation("Chosen label font Color:"));
        labelDefaultColor.setText(TRANSLATE_SERVICE.getTranslation("Default background Color:"));
        labelNotChosenFontColor.setText(TRANSLATE_SERVICE.getTranslation("Unchosen label font Color:"));
        labelGitHubTip.setText(TRANSLATE_SERVICE.getTranslation("This is an open source software,GitHub:"));
        labelGithubIssue.setText(TRANSLATE_SERVICE.getTranslation("If you find a bug or have some suggestions, welcome to GitHub for feedback"));
        labelDescription.setText(TRANSLATE_SERVICE.getTranslation("Thanks for the following project"));
        labelTranslationTip.setText(TRANSLATE_SERVICE.getTranslation("The translation might not be 100% accurate"));
        labelLanguageChooseTip.setText(TRANSLATE_SERVICE.getTranslation("Choose a language"));
        labelVersion.setText(TRANSLATE_SERVICE.getTranslation("Current Version:") + Constants.version);
        labelInstalledPluginNum.setText(TRANSLATE_SERVICE.getTranslation("Installed plugins num:"));
        labelVacuumTip.setText(TRANSLATE_SERVICE.getTranslation("Click to organize the database and reduce the size of the database"));
        labelVacuumTip2.setText(TRANSLATE_SERVICE.getTranslation("but it will consume a lot of time."));
        labelAddress.setText(TRANSLATE_SERVICE.getTranslation("Address"));
        labelPort.setText(TRANSLATE_SERVICE.getTranslation("Port"));
        labelUserName.setText(TRANSLATE_SERVICE.getTranslation("User name"));
        labelPassword.setText(TRANSLATE_SERVICE.getTranslation("Password"));
        labelProxyTip.setText(TRANSLATE_SERVICE.getTranslation("If you need a proxy to access the Internet, You can add a proxy here."));
        labelCacheSettings.setText(TRANSLATE_SERVICE.getTranslation("Cache Settings"));
        labelCacheTip.setText(TRANSLATE_SERVICE.getTranslation("You can edit the saved caches here"));
        labelCacheTip2.setText(TRANSLATE_SERVICE.getTranslation("The cache is automatically generated " + "by the software and will be displayed first when searching."));
        labelSearchBarFontColor.setText(TRANSLATE_SERVICE.getTranslation("SearchBar Font Color:"));
        labelBorderColor.setText(TRANSLATE_SERVICE.getTranslation("Border Color:"));
        labelCurrentCacheNum.setText(TRANSLATE_SERVICE.getTranslation("Current Caches Num:") + DatabaseService.getInstance().getCacheNum());
        labelUninstallPluginTip.setText(TRANSLATE_SERVICE.getTranslation("If you need to delete a plug-in, just delete it under the \"plugins\" folder in the software directory."));
        labelUninstallPluginTip2.setText(TRANSLATE_SERVICE.getTranslation("Tip:"));
        chooseUpdateAddressLabel.setText(TRANSLATE_SERVICE.getTranslation("Choose update address"));
        labelSearchCommand.setText(TRANSLATE_SERVICE.getTranslation("Search"));
        labelSuffixTip.setText(TRANSLATE_SERVICE.getTranslation("Modifying the suffix priority requires rebuilding the index (input \":update\") to take effect"));
        labelIndexTip.setText(TRANSLATE_SERVICE.getTranslation("You can rebuild the disk index here"));
        labelIndexChooseDisk.setText(TRANSLATE_SERVICE.getTranslation("The disks listed below will be indexed"));
        labelTipNTFSTip.setText(TRANSLATE_SERVICE.getTranslation("Only supports NTFS format disks"));
        labelLocalDiskTip.setText(TRANSLATE_SERVICE.getTranslation("Only the disks on this machine are listed below, click \"Add\" button to add a removable disk"));
        labelBorderThickness.setText(TRANSLATE_SERVICE.getTranslation("Border Thickness"));
        labelBorderType.setText(TRANSLATE_SERVICE.getTranslation("Border Type"));
        labelRoundRadius.setText(TRANSLATE_SERVICE.getTranslation("Round rectangle radius"));
    }

    private void translateCheckbox() {
        checkBoxAddToStartup.setText(TRANSLATE_SERVICE.getTranslation("Add to startup"));
        checkBoxLoseFocus.setText(TRANSLATE_SERVICE.getTranslation("Close search bar when focus lost"));
        checkBoxAdmin.setText(TRANSLATE_SERVICE.getTranslation("Open other programs as an administrator " + "(provided that the software has privileges)"));
        checkBoxIsShowTipOnCreatingLnk.setText(TRANSLATE_SERVICE.getTranslation("Show tip on creating shortcut"));
        checkBoxResponseCtrl.setText(TRANSLATE_SERVICE.getTranslation("Double-click \"Ctrl\" to open the search bar"));
        checkBoxCheckUpdate.setText(TRANSLATE_SERVICE.getTranslation("Check for update at startup"));
        checkBoxIsAttachExplorer.setText(TRANSLATE_SERVICE.getTranslation("Attach to explorer"));
    }

    private void translateButtons() {
        buttonSaveAndRemoveDesktop.setText(TRANSLATE_SERVICE.getTranslation("Clear desktop"));
        buttonSaveAndRemoveDesktop.setToolTipText(TRANSLATE_SERVICE.getTranslation("Backup and remove all desktop files"));
        ButtonPriorityFolder.setText(TRANSLATE_SERVICE.getTranslation("Choose"));
        buttonChooseFile.setText(TRANSLATE_SERVICE.getTranslation("Choose"));
        buttonAddCMD.setText(TRANSLATE_SERVICE.getTranslation("Add"));
        buttonDelCmd.setText(TRANSLATE_SERVICE.getTranslation("Delete"));
        buttonResetColor.setText(TRANSLATE_SERVICE.getTranslation("Reset to default"));
        buttonCheckUpdate.setText(TRANSLATE_SERVICE.getTranslation("Check for update"));
        buttonUpdatePlugin.setText(TRANSLATE_SERVICE.getTranslation("Check for update"));
        buttonPluginMarket.setText(TRANSLATE_SERVICE.getTranslation("Plugin Market"));
        buttonDeleteCache.setText(TRANSLATE_SERVICE.getTranslation("Delete cache"));
        buttonDeleteAllCache.setText(TRANSLATE_SERVICE.getTranslation("Delete all"));
        buttonChangeTheme.setText(TRANSLATE_SERVICE.getTranslation("change theme"));
        buttonVacuum.setText(TRANSLATE_SERVICE.getTranslation("Optimize database"));
        buttonPreviewColor.setText(TRANSLATE_SERVICE.getTranslation("Preview"));
        buttonClosePreview.setText(TRANSLATE_SERVICE.getTranslation("Close preview window"));
        buttonAddSuffix.setText(TRANSLATE_SERVICE.getTranslation("Add"));
        buttonDeleteSuffix.setText(TRANSLATE_SERVICE.getTranslation("Delete"));
        buttonDeleteAllSuffix.setText(TRANSLATE_SERVICE.getTranslation("Delete all"));
        buttonRebuildIndex.setText(TRANSLATE_SERVICE.getTranslation("Rebuild"));
        buttonAddNewDisk.setText(TRANSLATE_SERVICE.getTranslation("Add"));
        buttonDeleteDisk.setText(TRANSLATE_SERVICE.getTranslation("Delete"));
    }

    private void translateRadioButtons() {
        radioButtonNoProxy.setText(TRANSLATE_SERVICE.getTranslation("No proxy"));
        radioButtonUseProxy.setText(TRANSLATE_SERVICE.getTranslation("Configure proxy"));
    }

    private void translate() {
        translateLabels();
        translateCheckbox();
        translateButtons();
        translateRadioButtons();
        frame.setTitle(TRANSLATE_SERVICE.getTranslation("Settings"));
    }

    private boolean isRepeatCommand(String name) {
        name = ":" + name;
        for (String each : allConfigs.getCmdSet()) {
            if (each.substring(0, each.indexOf(";")).equals(name)) {
                return true;
            }
        }
        return false;
    }

    private void hideFrame() {
        SwingUtilities.invokeLater(() -> {
            frame.setVisible(false);
            for (Component each : tabComponentNameMap.values()) {
                tabbedPane.addTab("", each);
            }
        });
    }

    @EventRegister(registerClass = ShowSettingsFrameEvent.class)
    private static void showSettingsFrameEvent(Event event) {
        ShowSettingsFrameEvent showSettingsFrameEvent = (ShowSettingsFrameEvent) event;
        SettingsFrame settingsFrame = getInstance();
        if (showSettingsFrameEvent.showTabName == null) {
            settingsFrame.showWindow();
        } else {
            settingsFrame.showWindow(showSettingsFrameEvent.showTabName);
        }
    }

    @EventRegister(registerClass = HideSettingsFrameEvent.class)
    private static void hideSettingsFrameEvent(Event event) {
        getInstance().hideFrame();
    }

    @EventRegister(registerClass = IsCacheExistEvent.class)
    private static void isCacheExistEvent(Event event) {
        IsCacheExistEvent cacheExist = (IsCacheExistEvent) event;
        event.setReturnValue(getInstance().isCacheExist(cacheExist.cache));
    }

    @EventRegister(registerClass = AddCacheEvent.class)
    private static void addCacheEvent(Event event) {
        AddCacheEvent addCacheEvent = (AddCacheEvent) event;
        getInstance().addCache(addCacheEvent.cache);
    }

    @EventRegister(registerClass = GetExcludeComponentEvent.class)
    private static void getExcludeComponentEvent(Event event) {
        event.setReturnValue(getInstance().excludeComponent);
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void restartEvent(Event event) {
        getInstance().hideFrame();
    }

    private void showWindow() {
        showWindow("tabGeneral");
    }

    private void showWindow(String tabName) {
        if (frame.isVisible()) {
            return;
        }
        frame.setResizable(true);
        double dpi = DpiUtil.getDpi();
        int width, height;
        try {
            width = Integer.parseInt(TRANSLATE_SERVICE.getFrameWidth());
            height = Integer.parseInt(TRANSLATE_SERVICE.getFrameHeight());
        } catch (NumberFormatException e) {
            e.printStackTrace();
            width = (int) (1000 / dpi);
            height = (int) (600 / dpi);
        }
        frame.setContentPane(getInstance().panel);
        Dimension dimension = new Dimension(width, height);
        panel.setPreferredSize(dimension);
        frame.setSize(dimension);
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        initGUI();
        showOnTabbedPane(tabName);
        textAreaDescription.setForeground(tabbedPane.getForeground());
        frame.setVisible(true);
    }

    private void checkRoundRadius(StringBuilder stringBuilder) {
        if (!canParseDouble(textFieldRoundRadius.getText(), 0, 100)) {
            stringBuilder.append(TRANSLATE_SERVICE.getTranslation("Round rectangle radius is wrong, please change"));
        }
    }

    private void checkBorderThickness(StringBuilder stringBuilder) {
        if (!canParseInteger(textFieldBorderThickness.getText(), 1, 4)) {
            stringBuilder.append(TRANSLATE_SERVICE.getTranslation("Border thickness is too large, please change"));
        }
    }

    private void checkUpdateTimeLimit(StringBuilder strBuilder) {
        if (!canParseInteger(textFieldUpdateInterval.getText(), 1, 3600)) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("The file index update setting is wrong, please change")).append("\n");
        }
    }

    private void checkCacheNumLimit(StringBuilder strBuilder) {
        if (!canParseInteger(textFieldCacheNum.getText(), 1, 3600)) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("The cache capacity is set incorrectly, please change")).append("\n");
        }
    }

    private void checkHotKey(StringBuilder strBuilder) {
        String tmp_hotkey = textFieldHotkey.getText();
        if (tmp_hotkey.length() < 5) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("Hotkey setting is wrong, please change")).append("\n");
        } else {
            CheckHotKeyAvailableEvent checkHotKeyAvailableEvent = new CheckHotKeyAvailableEvent(tmp_hotkey);
            eventManagement.putEvent(checkHotKeyAvailableEvent);
            eventManagement.waitForEvent(checkHotKeyAvailableEvent);
            Optional<Boolean> isAvailable = checkHotKeyAvailableEvent.getReturnValue();
            isAvailable.ifPresent(ret -> {
                if (!ret) {
                    strBuilder.append(TRANSLATE_SERVICE.getTranslation("Hotkey setting is wrong, please change")).append("\n");
                }
            });
        }
        if (tmp_openLastFolderKeyCode == tmp_runAsAdminKeyCode || tmp_openLastFolderKeyCode == tmp_copyPathKeyCode || tmp_runAsAdminKeyCode == tmp_copyPathKeyCode) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("HotKey conflict")).append("\n");
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
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("Transparency setting error")).append("\n");
        }
    }

    private void checkLabelColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldLabelColor.getText())) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("Chosen label color is set incorrectly")).append("\n");
        }
    }

    private void checkLabelFontColorWithCoverage(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldFontColorWithCoverage.getText())) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("Chosen label font color is set incorrectly")).append("\n");
        }
    }

    private void checkDefaultBackgroundColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldBackgroundDefault.getText())) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("Incorrect default background color setting")).append("\n");
        }
    }

    private void checkBorderColor(StringBuilder stringBuilder) {
        if (!canParseToRGB(textFieldBorderColor.getText())) {
            stringBuilder.append(TRANSLATE_SERVICE.getTranslation("Border color is set incorrectly")).append("\n");
        }
    }

    private void checkLabelFontColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldFontColor.getText())) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("Unchosen label font color is set incorrectly")).append("\n");
        }
    }

    private void checkSearchBarColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldSearchBarColor.getText())) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("The color of the search bar is set incorrectly")).append("\n");
        }
    }

    private void checkSearchBarFontColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldSearchBarFontColor.getText())) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("The font color of the search bar is set incorrectly")).append("\n");
        }
    }

    private void checkProxy(StringBuilder strBuilder) {
        if (!canParseInteger(textFieldPort.getText(), 0, 65535)) {
            strBuilder.append(TRANSLATE_SERVICE.getTranslation("Proxy port is set incorrectly.")).append("\n");
        }
    }

    /**
     * 生成configuration
     *
     * @return ConfigEntity
     */
    private ConfigEntity getConfigEntity() {
        ConfigEntity configEntity = new ConfigEntity();
        String ignorePathTemp = RegexUtil.lineFeed.matcher(textAreaIgnorePath.getText()).replaceAll("");
        String swingTheme = (String) listSwingThemes.getSelectedValue();
        Constants.Enums.BorderType borderType = (Constants.Enums.BorderType) comboBoxBorderType.getSelectedItem();
        String tmp_proxyAddress = textFieldAddress.getText();
        String tmp_proxyUserName = textFieldUserName.getText();
        String tmp_proxyPassword = textFieldPassword.getText();
        if (radioButtonProxyTypeSocks5.isSelected()) {
            configEntity.setProxyType(Constants.Enums.ProxyType.PROXY_SOCKS);
        } else if (radioButtonProxyTypeHttp.isSelected()) {
            configEntity.setProxyType(Constants.Enums.ProxyType.PROXY_HTTP);
        }
        if (radioButtonNoProxy.isSelected()) {
            configEntity.setProxyType(Constants.Enums.ProxyType.PROXY_DIRECT);
        }
        configEntity.setUpdateAddress((String) chooseUpdateAddress.getSelectedItem());
        configEntity.setBorderThickness(Integer.parseInt(textFieldBorderThickness.getText()));
        if (borderType == null) {
            borderType = Constants.Enums.BorderType.AROUND;
        }
        configEntity.setBorderType(borderType.toString());
        configEntity.setPriorityFolder(textFieldPriorityFolder.getText());
        configEntity.setHotkey(textFieldHotkey.getText());
        configEntity.setCacheNumLimit(Integer.parseInt(textFieldCacheNum.getText()));
        configEntity.setUpdateTimeLimit(Integer.parseInt(textFieldUpdateInterval.getText()));
        configEntity.setIgnorePath(ignorePathTemp);
        configEntity.setDefaultAdmin(checkBoxAdmin.isSelected());
        configEntity.setLoseFocusClose(checkBoxLoseFocus.isSelected());
        configEntity.setShowTipCreatingLnk(checkBoxIsShowTipOnCreatingLnk.isSelected());
        configEntity.setTransparency(Float.parseFloat(textFieldTransparency.getText()));
        configEntity.setLabelColor(Integer.parseInt(textFieldLabelColor.getText(), 16));
        configEntity.setBorderColor(Integer.parseInt(textFieldBorderColor.getText(), 16));
        configEntity.setDefaultBackgroundColor(Integer.parseInt(textFieldBackgroundDefault.getText(), 16));
        configEntity.setSearchBarColor(Integer.parseInt(textFieldSearchBarColor.getText(), 16));
        configEntity.setFontColorWithCoverage(Integer.parseInt(textFieldFontColorWithCoverage.getText(), 16));
        configEntity.setFontColor(Integer.parseInt(textFieldFontColor.getText(), 16));
        configEntity.setSearchBarFontColor(Integer.parseInt(textFieldSearchBarFontColor.getText(), 16));
        configEntity.setRoundRadius(Double.parseDouble(textFieldRoundRadius.getText()));
        configEntity.setProxyAddress(tmp_proxyAddress);
        configEntity.setProxyPort(Integer.parseInt(textFieldPort.getText()));
        configEntity.setProxyUserName(tmp_proxyUserName);
        configEntity.setProxyPassword(tmp_proxyPassword);
        configEntity.setOpenLastFolderKeyCode(tmp_openLastFolderKeyCode);
        configEntity.setRunAsAdminKeyCode(tmp_runAsAdminKeyCode);
        configEntity.setCopyPathKeyCode(tmp_copyPathKeyCode);
        configEntity.setSwingTheme(swingTheme);
        configEntity.setLanguage(TRANSLATE_SERVICE.getLanguage());
        configEntity.setDoubleClickCtrlOpen(checkBoxResponseCtrl.isSelected());
        configEntity.setCheckUpdateStartup(checkBoxCheckUpdate.isSelected());
        configEntity.setDisks(parseDisk());
        configEntity.setAttachExplorer(checkBoxIsAttachExplorer.isSelected());
        return configEntity;
    }

    /**
     * 保存所有设置
     *
     * @return 出现的错误信息，若成功则为空
     */
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
        checkCacheNumLimit(errorsStrb);
        checkUpdateTimeLimit(errorsStrb);
        checkBorderThickness(errorsStrb);
        checkRoundRadius(errorsStrb);

        String errors = errorsStrb.toString();
        if (!errors.isEmpty()) {
            return errors;
        }

        //重新显示翻译GUI，采用等号比对内存地址，不是用equals
        if (!(listLanguage.getSelectedValue() == TRANSLATE_SERVICE.getLanguage())) {
            TRANSLATE_SERVICE.setLanguage((String) listLanguage.getSelectedValue());
            translate();
        }

        //所有配置均正确
        //使所有配置生效
        ConfigEntity configEntity = getConfigEntity();

        eventManagement.putEvent(new ResponseCtrlEvent(checkBoxResponseCtrl.isSelected()));

        SaveConfigsEvent event = new SaveConfigsEvent(configEntity);
        eventManagement.putEvent(event);
        eventManagement.waitForEvent(event);
        eventManagement.putEvent(new SetConfigsEvent());

        Color tmp_color = new Color(allConfigs.getLabelColor());
        labelColorChooser.setBackground(tmp_color);
        labelColorChooser.setForeground(tmp_color);
        tmp_color = new Color(allConfigs.getDefaultBackgroundColor());
        defaultBackgroundChooser.setBackground(tmp_color);
        defaultBackgroundChooser.setForeground(tmp_color);
        tmp_color = new Color(allConfigs.getLabelFontColorWithCoverage());
        FontColorWithCoverageChooser.setBackground(tmp_color);
        FontColorWithCoverageChooser.setForeground(tmp_color);
        tmp_color = new Color(allConfigs.getLabelFontColor());
        FontColorChooser.setBackground(tmp_color);
        FontColorChooser.setForeground(tmp_color);
        tmp_color = new Color(allConfigs.getSearchBarFontColor());
        SearchBarFontColorChooser.setBackground(tmp_color);
        SearchBarFontColorChooser.setForeground(tmp_color);
        tmp_color = new Color(allConfigs.getBorderColor());
        borderColorChooser.setBackground(tmp_color);
        borderColorChooser.setForeground(tmp_color);

        //保存自定义命令
        StringBuilder strb = new StringBuilder();
        for (String each : allConfigs.getCmdSet()) {
            strb.append(each);
            strb.append("\n");
        }
        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/cmds.txt"), StandardCharsets.UTF_8))) {
            bw.write(strb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        hideFrame();
        return "";
    }

    private void setStartup(boolean b) {
        if (b) {
            try {
                StartupUtil.deleteStartup();
                Process p = StartupUtil.addStartup();
                BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                String line;
                StringBuilder result = new StringBuilder();
                while ((line = outPut.readLine()) != null) {
                    result.append(line);
                }
                outPut.close();
                if (!result.toString().isEmpty()) {
                    checkBoxAddToStartup.setSelected(false);
                    JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Add to startup failed, please try to run as administrator") + "\n" + result);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            if (hasStartup() == 0) {
                try {
                    Process p = StartupUtil.deleteStartup();
                    StringBuilder result = new StringBuilder();
                    try (BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()))) {
                        String line;
                        while ((line = outPut.readLine()) != null) {
                            result.append(line);
                        }
                    }
                    if (!result.toString().isEmpty()) {
                        checkBoxAddToStartup.setSelected(true);
                        JOptionPane.showMessageDialog(frame, TRANSLATE_SERVICE.getTranslation("Delete startup failed, please try to run as administrator"));
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private static class TabNameAndTitle {
        private final String title;
        private final String tabName;

        private TabNameAndTitle(String tabName, String title) {
            this.tabName = tabName;
            this.title = title;
        }
    }
}
