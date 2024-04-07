package file.engine.frames;

import com.intellij.uiDesigner.core.GridConstraints;
import com.intellij.uiDesigner.core.GridLayoutManager;
import com.intellij.uiDesigner.core.Spacer;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.ConfigEntity;
import file.engine.configs.Constants;
import file.engine.configs.core.CoreConfigEntity;
import file.engine.dllInterface.IsLocalDisk;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.configs.AddCmdEvent;
import file.engine.event.handler.impl.configs.DeleteCmdEvent;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.configs.SetSwingLaf;
import file.engine.event.handler.impl.database.*;
import file.engine.event.handler.impl.download.StartDownloadEvent;
import file.engine.event.handler.impl.download.StopDownloadEvent;
import file.engine.event.handler.impl.frame.pluginMarket.ShowPluginMarket;
import file.engine.event.handler.impl.frame.searchBar.HideSearchBarEvent;
import file.engine.event.handler.impl.frame.searchBar.PreviewSearchBarEvent;
import file.engine.event.handler.impl.frame.searchBar.StartPreviewEvent;
import file.engine.event.handler.impl.frame.searchBar.StopPreviewEvent;
import file.engine.event.handler.impl.frame.settingsFrame.GetExcludeComponentEvent;
import file.engine.event.handler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import file.engine.event.handler.impl.hotkey.CheckHotKeyAvailableEvent;
import file.engine.event.handler.impl.plugin.AddPluginsCanUpdateEvent;
import file.engine.event.handler.impl.plugin.GetPluginByNameEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.frames.components.SetDownloadProgress;
import file.engine.services.DatabaseNativeService;
import file.engine.services.TranslateService;
import file.engine.services.download.DownloadManager;
import file.engine.services.plugin.system.Plugin;
import file.engine.services.plugin.system.PluginService;
import file.engine.utils.DpiUtil;
import file.engine.utils.RegexUtil;
import file.engine.utils.StartupUtil;
import file.engine.utils.ThreadPoolUtil;
import file.engine.utils.file.MoveDesktopFilesUtil;
import file.engine.utils.system.properties.IsDebug;
import file.engine.utils.system.properties.IsPreview;
import lombok.extern.slf4j.Slf4j;

import javax.swing.*;
import javax.swing.border.TitledBorder;
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
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static file.engine.utils.ColorUtil.*;
import static file.engine.utils.StartupUtil.hasStartup;

@Slf4j
public class SettingsFrame {
    private Set<String> cacheSet;
    private volatile boolean isFramePrepared = false;
    private boolean isSuffixChanged = false;
    private static volatile int tmp_copyPathKeyCode;
    private static volatile int tmp_runAsAdminKeyCode;
    private static volatile int tmp_openLastFolderKeyCode;
    private static final ImageIcon frameIcon = new ImageIcon(Objects.requireNonNull(SettingsFrame.class.getResource("/icons/frame.png")));
    private static final JFrame frame = new JFrame("Settings");
    private static final TranslateService translateService = TranslateService.getInstance();
    private static final EventManagement eventManagement = EventManagement.getInstance();
    private static final AllConfigs allConfigs = AllConfigs.getInstance();
    private static final ThreadPoolUtil threadPoolUtil = ThreadPoolUtil.getInstance();
    private final HashMap<TabNameAndTitle, Component> tabComponentNameMap = new HashMap<>();
    private Map<String, String> cudaDeviceMap = new HashMap<>();
    private Map<String, Integer> suffixMap;
    private final Set<Component> excludeComponent = ConcurrentHashMap.newKeySet();
    private final LinkedHashSet<String> diskSet = new LinkedHashSet<>();
    private final Set<String> downloadedPlugins = ConcurrentHashMap.newKeySet();
    private final DownloadManager[] downloadManager = new DownloadManager[2]; // 两个元素，第一个为File-Engine.jar的下载管理对象，第二个为File-Engine.exe的下载管理对象
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
    private JLabel labelHolder;
    private JLabel labelHolder2;
    private JLabel labelHolder3;
    private JLabel labelHolder4;
    private JLabel labelHolder5;
    private JLabel labelHolder6;
    private JCheckBox checkBoxEnableCuda;
    private JComboBox<Object> comboBoxCudaDevice;
    private JLabel labelSearchThread;
    private JComboBox<Object> comboBoxSearchThread;
    private JLabel labelBuildVersion;
    private JLabel labelProject1;
    private JLabel labelProject2;
    private JLabel labelProject3;
    private JLabel labelProject4;
    private JLabel labelProject5;
    private JLabel labelProject6;
    private JLabel labelProject7;
    private JLabel labelProject8;
    private JLabel labelProject9;
    private JLabel labelProject10;
    private JButton buttonPluginSettings;


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
                    int ret = JOptionPane.showConfirmDialog(null, translateService.getTranslation("Errors") + ":\n" + errors + "\n" + translateService.getTranslation("Failed to save settings, do you still close the window"));
                    if (ret == JOptionPane.YES_OPTION) {
                        hideFrame();
                    }
                } else {
                    if (isSuffixChanged) {
                        TranslateService translateService = TranslateService.INSTANCE;
                        eventManagement.putEvent(new ShowTaskBarMessageEvent(
                                translateService.getTranslation("Info"),
                                translateService.getTranslation("Updating file index")));
                        eventManagement.putEvent(new UpdateDatabaseEvent(true),
                                event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(
                                        translateService.getTranslation("Info"),
                                        translateService.getTranslation("Search Done"))),
                                event -> eventManagement.putEvent(new ShowTaskBarMessageEvent(
                                        translateService.getTranslation("Warning"),
                                        translateService.getTranslation("Search Failed"))));
                    }
                    hideFrame();
                }
            }
        });
    }

    private void addCheckBoxListener() {
        checkBoxLoseFocus.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                saveChanges();
            }
        });
        checkBoxAdmin.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                saveChanges();
            }
        });
        checkBoxIsShowTipOnCreatingLnk.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                saveChanges();
            }
        });
        checkBoxIsAttachExplorer.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                saveChanges();
            }
        });
        //添加到开机启动监听器
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
                JOptionPane.showMessageDialog(frame, translateService.getTranslation("The program is detected on the desktop and cannot be moved"));
                return;
            }
            int isConfirmed = JOptionPane.showConfirmDialog(frame, translateService.getTranslation("Whether to remove and backup all files on the desktop," + "they will be in the program's Files folder, which may take a few minutes"));
            if (isConfirmed == JOptionPane.YES_OPTION) {
                Future<Boolean> future = threadPoolUtil.executeTask(MoveDesktopFilesUtil::start, true);
                try {
                    if (future == null) {
                        return;
                    }
                    if (!future.get()) {
                        JOptionPane.showMessageDialog(null, translateService.getTranslation("Files with the same name are detected, please move them by yourself"));
                    }
                } catch (InterruptedException | ExecutionException exception) {
                    log.error(exception.getMessage(), exception);
                }
            }
        });
    }

    private void addFileChooserButtonListener() {
        buttonChooseFile.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), translateService.getTranslation("Choose"));
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
            final LinkedHashSet<String> hotkeys = new LinkedHashSet<>();
            final HashMap<Integer, String> availableHotkeyMap = new HashMap<>() {{
                put(KeyEvent.VK_CONTROL, "Ctrl");
                put(KeyEvent.VK_ALT, "Alt");
                put(KeyEvent.VK_SHIFT, "Shift");
                put(0x5B, "Win");
                put(KeyEvent.VK_F1, "F1");
                put(KeyEvent.VK_F2, "F2");
                put(KeyEvent.VK_F3, "F3");
                put(KeyEvent.VK_F4, "F4");
                put(KeyEvent.VK_F5, "F5");
                put(KeyEvent.VK_F6, "F6");
                put(KeyEvent.VK_F7, "F7");
                put(KeyEvent.VK_F8, "F8");
                put(KeyEvent.VK_F9, "F9");
                put(KeyEvent.VK_F10, "F10");
                put(KeyEvent.VK_F11, "F11");
                put(KeyEvent.VK_F12, "F12");
            }};

            @Override
            public void keyTyped(KeyEvent e) {
                // no use
            }

            @Override
            public void keyPressed(KeyEvent e) {
                final int key = e.getKeyCode();
                if (reset) {
                    textFieldHotkey.setText(null);
                    hotkeys.clear();
                    reset = false;
                }
                if (availableHotkeyMap.containsKey(key) || 'A' <= key && key <= 'Z') {
                    hotkeys.add(availableHotkeyMap.getOrDefault(key, String.valueOf((char) key)));
                }
                StringBuilder stringBuilder = new StringBuilder();
                String token = " + ";
                final int size = hotkeys.size();
                ArrayList<String> tmpHotkeys = new ArrayList<>(hotkeys);
                for (int i = 0; i < size - 1; ++i) {
                    stringBuilder.append(tmpHotkeys.get(i)).append(token);
                }
                stringBuilder.append(tmpHotkeys.get(size - 1));
                textFieldHotkey.setText(stringBuilder.toString());
            }

            @Override
            public void keyReleased(KeyEvent e) {
                reset = true;
            }
        });
    }

    private void addTextFieldProxyListener() {
        DocumentListener documentListener = new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                saveChanges();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                saveChanges();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
            }
        };
        textFieldAddress.getDocument().addDocumentListener(documentListener);
        textFieldPort.getDocument().addDocumentListener(documentListener);
        textFieldUserName.getDocument().addDocumentListener(documentListener);
        textFieldPassword.getDocument().addDocumentListener(documentListener);
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
            int returnValue = fileChooser.showDialog(new JLabel(), translateService.getTranslation("Choose"));
            File file = fileChooser.getSelectedFile();
            if (returnValue == JFileChooser.APPROVE_OPTION) {
                textFieldPriorityFolder.setText(file.getAbsolutePath());
            }
            saveChanges();
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
            String name = JOptionPane.showInputDialog(translateService.getTranslation("Please enter the ID of the command, then you can enter \": identifier\" in the search box to execute the command directly"));
            if (name == null || name.isEmpty()) {
                //未输入
                return;
            }
            if ("update".equalsIgnoreCase(name) || "clearbin".equalsIgnoreCase(name) || "help".equalsIgnoreCase(name) || "version".equalsIgnoreCase(name) || isRepeatCommand(name)) {
                JOptionPane.showMessageDialog(frame, translateService.getTranslation("Conflict with existing commands"));
                return;
            }
            if (name.length() == 1) {
                int ret = JOptionPane.showConfirmDialog(frame, translateService.getTranslation("The identifier you entered is too short, continue") + "?");
                if (ret != JOptionPane.OK_OPTION) {
                    return;
                }
            }
            String cmd;
            JOptionPane.showMessageDialog(frame, translateService.getTranslation("Please select the location of the executable file (a folder is also acceptable)"));
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
            int returnValue = fileChooser.showDialog(new Label(), translateService.getTranslation("Choose"));
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
        MouseAdapter aHrefMouseAdapter = new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Desktop desktop;
                if (Desktop.isDesktopSupported()) {
                    desktop = Desktop.getDesktop();
                    JLabel labelSource = (JLabel) e.getSource();
                    String text = labelSource.getText();
                    Pattern aHref = RegexUtil.getPattern("<a\\s+href\\s*=\\s*(\"|')?(.*?)[\"|'|>]", Pattern.CASE_INSENSITIVE);
                    Matcher matcher = aHref.matcher(text);
                    while (matcher.find()) {
                        String link = matcher.group(2).trim();
                        try {
                            URI uri = new URI(link);
                            desktop.browse(uri);
                        } catch (Exception ex) {
                            log.error(ex.getMessage(), ex);
                        }
                    }
                }
            }
        };
        labelAboutGithub.addMouseListener(aHrefMouseAdapter);
        labelProject1.addMouseListener(aHrefMouseAdapter);
        labelProject2.addMouseListener(aHrefMouseAdapter);
        labelProject3.addMouseListener(aHrefMouseAdapter);
        labelProject4.addMouseListener(aHrefMouseAdapter);
        labelProject5.addMouseListener(aHrefMouseAdapter);
        labelProject6.addMouseListener(aHrefMouseAdapter);
        labelProject7.addMouseListener(aHrefMouseAdapter);
        labelProject8.addMouseListener(aHrefMouseAdapter);
        labelProject9.addMouseListener(aHrefMouseAdapter);
        labelProject10.addMouseListener(aHrefMouseAdapter);
    }

    /**
     * 点击检查更新
     */
    private void addCheckForUpdateButtonListener() {
        buttonCheckUpdate.addActionListener(e -> {
            if (downloadManager[1] != null && downloadManager[1].getDownloadStatus() == Constants.Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                eventManagement.putEvent(new StopDownloadEvent(downloadManager[0]));
                eventManagement.putEvent(new StopDownloadEvent(downloadManager[1]));
            } else {
                checkForUpdate(downloadManager, true);
            }
        });
    }

    private void checkForUpdate(DownloadManager[] downloadManager, boolean isShowCheckUpdateDialogs) {
        //开始下载
        Map<String, Object> updateInfo;
        String latestVersion;
        try {
            updateInfo = allConfigs.getUpdateInfo();
            if (updateInfo != null && !updateInfo.isEmpty()) {
                latestVersion = (String) updateInfo.get("version");
            } else {
                throw new IOException("failed");
            }
        } catch (IOException e1) {
            if (isShowCheckUpdateDialogs) {
                JOptionPane.showMessageDialog(frame, translateService.getTranslation("Check update failed"));
                showManualDownloadDialog();
            }
            return;
        }
        if (Double.parseDouble(latestVersion) > Double.parseDouble(Constants.version) || IsPreview.isPreview() || IsDebug.isDebug()) {
            String description = (String) updateInfo.get("description");
            int result = JOptionPane.showConfirmDialog(frame, translateService.getTranslation("New Version available") + "  " +
                    latestVersion + ", " + translateService.getTranslation("Whether to update") + "\n" +
                    translateService.getTranslation("update content") + "\n" + description);
            if (result == JOptionPane.YES_OPTION) {
                //开始更新,下载更新文件到tmp
                downloadManager[0] = new DownloadManager((String) updateInfo.get("url64"), Constants.FILE_NAME, new File("tmp").getAbsolutePath());
                downloadManager[1] = new DownloadManager((String) updateInfo.get("urlLauncher"), Constants.LAUNCH_WRAPPER_NAME, new File("tmp").getAbsolutePath());
                //下载File-Engine.exe
                eventManagement.putEvent(new StartDownloadEvent(downloadManager[1]));
                threadPoolUtil.executeTask(() -> {
                    try {
                        if (!downloadManager[1].waitFor(10 * 60 * 1000)) {
                            return;
                        }
                        if (downloadManager[1].getDownloadStatus() == Constants.Enums.DownloadStatus.DOWNLOAD_DONE) {
                            Files.createFile(Path.of("user/updateLauncher"));
                        }
                    } catch (IOException e1) {
                        log.error(e1.getMessage(), e1);
                    }
                });
                //下载File-Engine.jar
                eventManagement.putEvent(new StartDownloadEvent(downloadManager[0]));
                threadPoolUtil.executeTask(() -> {
                    boolean isDownloadSuccess = SetDownloadProgress.setProgress(labelDownloadProgress,
                            buttonCheckUpdate,
                            downloadManager[1],
                            () -> Constants.FILE_NAME.equals(downloadManager[0].fileName),
                            () -> {
                                File updateSign = new File("user/update");
                                if (!updateSign.exists()) {
                                    try {
                                        if (updateSign.createNewFile()) {
                                            throw new RuntimeException("create user/update file failed.");
                                        }
                                    } catch (IOException ex) {
                                        throw new RuntimeException(ex);
                                    }
                                }
                                eventManagement.putEvent(new RestartEvent());
                            });
                    if (!isDownloadSuccess) {
                        showManualDownloadDialog();
                    }
                });
            }
        } else {
            if (isShowCheckUpdateDialogs) {
                JOptionPane.showMessageDialog(frame, translateService.getTranslation("Latest version:") + latestVersion + "\n" +
                        translateService.getTranslation("The current version is the latest"));
            }
        }
    }

    /**
     * 显示手动下载弹窗
     */
    private void showManualDownloadDialog() {
        int ret = JOptionPane.showConfirmDialog(frame, translateService.getTranslation("Do you want to download it manually") + "?");
        if (ret == JOptionPane.YES_OPTION) {
            Desktop desktop;
            if (Desktop.isDesktopSupported()) {
                desktop = Desktop.getDesktop();
                try {
                    desktop.browse(new URI("https://github.com/XUANXUQAQ/File-Engine/releases/"));
                } catch (IOException | URISyntaxException ioException) {
                    log.error(ioException.getMessage(), ioException);
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
            var defaultSearchBarColor = Constants.DefaultColors.getDefaultSearchBarColor();
            textFieldFontColorWithCoverage.setText(toRGBHexString(defaultSearchBarColor.DEFAULT_FONT_COLOR_WITH_COVERAGE));
            textFieldSearchBarColor.setText(toRGBHexString(defaultSearchBarColor.DEFAULT_SEARCHBAR_COLOR));
            textFieldLabelColor.setText(toRGBHexString(defaultSearchBarColor.DEFAULT_LABEL_COLOR));
            textFieldBackgroundDefault.setText(toRGBHexString(defaultSearchBarColor.DEFAULT_WINDOW_BACKGROUND_COLOR));
            textFieldFontColor.setText(toRGBHexString(defaultSearchBarColor.DEFAULT_FONT_COLOR));
            textFieldSearchBarFontColor.setText(toRGBHexString(defaultSearchBarColor.DEFAULT_SEARCHBAR_FONT_COLOR));
            textFieldBorderColor.setText(toRGBHexString(defaultSearchBarColor.DEFAULT_BORDER_COLOR));
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
    @SuppressWarnings("BooleanMethodIsAlwaysInverted")
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
    private boolean canParseFloat(String str, float min, float max) {
        try {
            float v = Float.parseFloat(str);
            if (min <= v && v <= max) {
                return true;
            }
            throw new Exception("parse failed");
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
        threadPoolUtil.executeTask(() -> {
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
                Color color = JColorChooser.showDialog(frame, translateService.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldLabelColor.setText(parseColorHex(color));
            }
        });
        FontColorWithCoverageChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, translateService.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldFontColorWithCoverage.setText(parseColorHex(color));
            }
        });
        defaultBackgroundChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, translateService.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldBackgroundDefault.setText(parseColorHex(color));
            }
        });
        FontColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, translateService.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldFontColor.setText(parseColorHex(color));
            }
        });

        SearchBarFontColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, translateService.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldSearchBarFontColor.setText(parseColorHex(color));
            }
        });

        borderColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, translateService.getTranslation("Choose Color"), null);
                if (color == null) {
                    return;
                }
                textFieldBorderColor.setText(parseColorHex(color));
            }
        });

        searchBarColorChooser.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Color color = JColorChooser.showDialog(frame, translateService.getTranslation("Choose Color"), null);
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
                        labelPluginVersion.setText(translateService.getTranslation("Version") + ":" + version);
                        labelApiVersion.setText("API " + translateService.getTranslation("Version") + ":" + apiVersion);
                        PluginIconLabel.setIcon(icon);
                        PluginNamelabel.setText("<html><body><font size=\"+1\">" + pluginName + "</body></html>");
                        textAreaDescription.setText(description);
                        labelAuthor.setText(translateService.getTranslation("Author") + ":" + author);
                        labelOfficialSite.setText("<html><a href='" + officialSite + "'><font size=\"4\">" + pluginName + "</font></a></html>");
                        labelProgress.setText("");
                        buttonUpdatePlugin.setVisible(true);
                        buttonPluginSettings.setVisible(true);
                    });
                    if (PluginService.getInstance().hasPluginNotLatest(pluginName)) {
                        boolean isTaskDone = downloadedPlugins.contains(pluginName);
                        if (isTaskDone) {
                            buttonUpdatePlugin.setEnabled(false);
                            buttonUpdatePlugin.setText(translateService.getTranslation("Downloaded"));
                        } else {
                            Color color = new Color(51, 122, 183);
                            buttonUpdatePlugin.setText(translateService.getTranslation("Update"));
                            buttonUpdatePlugin.setBackground(color);
                            buttonUpdatePlugin.setEnabled(true);
                        }
                    } else {
                        buttonUpdatePlugin.setText(translateService.getTranslation("Check for update"));
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
        buttonChangeTheme.addActionListener(e -> JOptionPane.showMessageDialog(frame, paneSwingThemes, translateService.getTranslation("Change Theme"), JOptionPane.PLAIN_MESSAGE));
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
            int ret = JOptionPane.showConfirmDialog(frame, translateService.getTranslation("Confirm whether to start optimizing the database?"));
            if (JOptionPane.YES_OPTION == ret) {
                Constants.Enums.DatabaseStatus status = DatabaseNativeService.getStatus();
                if (status == Constants.Enums.DatabaseStatus.NORMAL) {
                    if (IsDebug.isDebug()) {
                        log.info("开始优化");
                    }
                    eventManagement.putEvent(new OptimiseDatabaseEvent());
                    threadPoolUtil.executeTask(() -> {
                        //实时显示VACUUM状态
                        try {
                            while (DatabaseNativeService.getStatus() == Constants.Enums.DatabaseStatus.VACUUM) {
                                labelVacuumStatus.setText(translateService.getTranslation("Optimizing..."));
                                TimeUnit.MILLISECONDS.sleep(50);
                            }
                            labelVacuumStatus.setText(translateService.getTranslation("Optimized"));
                            TimeUnit.SECONDS.sleep(3);
                            labelVacuumStatus.setText("");
                        } catch (InterruptedException ignored) {
                        }
                    });
                } else if (status == Constants.Enums.DatabaseStatus.MANUAL_UPDATE) {
                    JOptionPane.showMessageDialog(frame, translateService.getTranslation("Database is not usable yet, please wait..."));
                } else if (status == Constants.Enums.DatabaseStatus.VACUUM) {
                    JOptionPane.showMessageDialog(frame, translateService.getTranslation("Task is still running."));
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
            final ThreadPoolUtil threadPoolUtil = ThreadPoolUtil.getInstance();
            final HashSet<String> cmdSetTmp = new HashSet<>();

            search(String searchText) {
                this.searchText = searchText;
            }

            void doSearch() {
                threadPoolUtil.executeTask(() -> {
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

    {
// GUI initializer generated by IntelliJ IDEA GUI Designer
// >>> IMPORTANT!! <<<
// DO NOT EDIT OR ADD ANY CODE HERE!
        $$$setupUI$$$();
    }

    /**
     * Method generated by IntelliJ IDEA GUI Designer
     * >>> IMPORTANT!! <<<
     * DO NOT edit this method OR call it in your code!
     *
     * @noinspection ALL
     */
    private void $$$setupUI$$$() {
        panel = new JPanel();
        panel.setLayout(new GridLayoutManager(2, 1, new Insets(0, 0, 0, 0), -1, -1));
        panel.setBackground(new Color(-1));
        panel.setDoubleBuffered(true);
        panel.setMinimumSize(new Dimension(850, 600));
        panel.setOpaque(false);
        panel.setPreferredSize(new Dimension(850, 600));
        panel.setRequestFocusEnabled(true);
        panel.setVerifyInputWhenFocusTarget(false);
        panel.setVisible(true);
        splitPane = new JSplitPane();
        splitPane.setAutoscrolls(true);
        splitPane.setContinuousLayout(true);
        splitPane.setDividerLocation(154);
        splitPane.setLastDividerLocation(154);
        splitPane.setOpaque(true);
        panel.add(splitPane, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, new Dimension(200, 200), null, 0, false));
        leftPanel = new JPanel();
        leftPanel.setLayout(new GridLayoutManager(1, 1, new Insets(0, 0, 0, 0), -1, -1));
        leftPanel.setOpaque(false);
        splitPane.setLeftComponent(leftPanel);
        treeSettings = new JTree();
        treeSettings.setRootVisible(false);
        leftPanel.add(treeSettings, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_WANT_GROW, null, new Dimension(150, -1), null, 0, false));
        rightPanel = new JPanel();
        rightPanel.setLayout(new GridLayoutManager(1, 1, new Insets(0, 0, 0, 0), -1, -1));
        rightPanel.setMinimumSize(new Dimension(0, 0));
        rightPanel.setOpaque(false);
        splitPane.setRightComponent(rightPanel);
        tabbedPane = new JTabbedPane();
        tabbedPane.setAutoscrolls(true);
        tabbedPane.setBackground(new Color(-1));
        tabbedPane.setDoubleBuffered(true);
        tabbedPane.setFocusable(false);
        tabbedPane.setOpaque(false);
        tabbedPane.setTabLayoutPolicy(0);
        tabbedPane.setTabPlacement(1);
        tabbedPane.setVisible(true);
        rightPanel.add(tabbedPane, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_VERTICAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, new Dimension(780, 589), new Dimension(780, 589), new Dimension(780, -1), 0, false));
        tabGeneral = new JPanel();
        tabGeneral.setLayout(new GridLayoutManager(9, 5, new Insets(0, 0, 0, 0), -1, -1));
        tabGeneral.setBackground(new Color(-1));
        tabGeneral.setMaximumSize(new Dimension(1000, 600));
        tabGeneral.setMinimumSize(new Dimension(850, 500));
        tabGeneral.setOpaque(false);
        tabbedPane.addTab("General", tabGeneral);
        tabGeneral.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        final Spacer spacer1 = new Spacer();
        tabGeneral.add(spacer1, new GridConstraints(8, 4, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, new Dimension(23, 14), null, 0, false));
        placeholder1 = new JLabel();
        placeholder1.setText("    ");
        tabGeneral.add(placeholder1, new GridConstraints(2, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(23, 17), null, 0, false));
        placeholder2 = new JLabel();
        placeholder2.setText("    ");
        tabGeneral.add(placeholder2, new GridConstraints(5, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(23, 17), null, 0, false));
        placeholder3 = new JLabel();
        placeholder3.setText("    ");
        tabGeneral.add(placeholder3, new GridConstraints(6, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(23, 17), null, 0, false));
        labelMaxCacheNum = new JLabel();
        labelMaxCacheNum.setText("Set the maximum number of caches:");
        tabGeneral.add(labelMaxCacheNum, new GridConstraints(4, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(100, 17), null, 0, false));
        placeholderGeneral = new JLabel();
        placeholderGeneral.setText("  ");
        tabGeneral.add(placeholderGeneral, new GridConstraints(7, 0, 1, 4, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(8, 20), null, 0, false));
        labelUpdateInterval = new JLabel();
        labelUpdateInterval.setText("File update detection interval:");
        tabGeneral.add(labelUpdateInterval, new GridConstraints(5, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(114, 17), null, 0, false));
        checkBoxAddToStartup = new JCheckBox();
        checkBoxAddToStartup.setOpaque(false);
        checkBoxAddToStartup.setText("Add to startup");
        tabGeneral.add(checkBoxAddToStartup, new GridConstraints(0, 0, 1, 4, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(108, 21), null, 0, false));
        placeholderGeneral1 = new JLabel();
        placeholderGeneral1.setText("   ");
        tabGeneral.add(placeholderGeneral1, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(12, 32), null, 0, false));
        textFieldUpdateInterval = new JTextField();
        textFieldUpdateInterval.setOpaque(false);
        tabGeneral.add(textFieldUpdateInterval, new GridConstraints(5, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(100, -1), null, 0, false));
        textFieldCacheNum = new JTextField();
        textFieldCacheNum.setOpaque(false);
        tabGeneral.add(textFieldCacheNum, new GridConstraints(4, 2, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(100, -1), null, 0, false));
        buttonChangeTheme = new JButton();
        buttonChangeTheme.setText("change theme");
        tabGeneral.add(buttonChangeTheme, new GridConstraints(2, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonSaveAndRemoveDesktop = new JButton();
        buttonSaveAndRemoveDesktop.setText("Clear desktop");
        tabGeneral.add(buttonSaveAndRemoveDesktop, new GridConstraints(3, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelSecond = new JLabel();
        labelSecond.setText("Seconds");
        tabGeneral.add(labelSecond, new GridConstraints(5, 3, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        checkBoxCheckUpdate = new JCheckBox();
        checkBoxCheckUpdate.setLabel("Check for update at startup");
        checkBoxCheckUpdate.setOpaque(false);
        checkBoxCheckUpdate.setText("Check for update at startup");
        tabGeneral.add(checkBoxCheckUpdate, new GridConstraints(1, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        tabSearchSettings = new JPanel();
        tabSearchSettings.setLayout(new GridLayoutManager(13, 6, new Insets(0, 0, 0, 0), -1, -1));
        tabSearchSettings.setBackground(new Color(-1));
        tabSearchSettings.setMaximumSize(new Dimension(1000, 600));
        tabSearchSettings.setMinimumSize(new Dimension(850, 500));
        tabSearchSettings.setOpaque(false);
        tabbedPane.addTab("Search settings", tabSearchSettings);
        tabSearchSettings.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        labelSetIgnorePathTip = new JLabel();
        labelSetIgnorePathTip.setText("Set ignore folder:");
        tabSearchSettings.add(labelSetIgnorePathTip, new GridConstraints(7, 0, 1, 6, GridConstraints.ANCHOR_NORTHWEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholdersearch0 = new JLabel();
        placeholdersearch0.setText("     ");
        tabSearchSettings.add(placeholdersearch0, new GridConstraints(3, 0, 1, 3, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labeltipPriorityFolder = new JLabel();
        labeltipPriorityFolder.setText("Priority search folder location (double-click to clear):");
        tabSearchSettings.add(labeltipPriorityFolder, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelPlaceHolder = new JLabel();
        labelPlaceHolder.setText("     ");
        tabSearchSettings.add(labelPlaceHolder, new GridConstraints(6, 0, 1, 3, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelConstIgnorePathTip = new JLabel();
        labelConstIgnorePathTip.setText("Separate different paths with commas, and ignore C:\\Windows by default");
        tabSearchSettings.add(labelConstIgnorePathTip, new GridConstraints(8, 0, 1, 3, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        scrollpaneIgnorePath = new JScrollPane();
        tabSearchSettings.add(scrollpaneIgnorePath, new GridConstraints(9, 0, 1, 4, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        textAreaIgnorePath = new JTextArea();
        textAreaIgnorePath.setForeground(new Color(-4473925));
        textAreaIgnorePath.setLineWrap(true);
        textAreaIgnorePath.setOpaque(false);
        textAreaIgnorePath.setWrapStyleWord(true);
        scrollpaneIgnorePath.setViewportView(textAreaIgnorePath);
        textFieldPriorityFolder = new JTextField();
        textFieldPriorityFolder.setEditable(false);
        textFieldPriorityFolder.setOpaque(false);
        tabSearchSettings.add(textFieldPriorityFolder, new GridConstraints(1, 1, 1, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        ButtonPriorityFolder = new JButton();
        ButtonPriorityFolder.setText("Choose");
        tabSearchSettings.add(ButtonPriorityFolder, new GridConstraints(1, 3, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelVacuumTip2 = new JLabel();
        labelVacuumTip2.setText("but it will consume a lot of time.");
        tabSearchSettings.add(labelVacuumTip2, new GridConstraints(5, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelVacuumStatus = new JLabel();
        labelVacuumStatus.setText("      ");
        tabSearchSettings.add(labelVacuumStatus, new GridConstraints(4, 3, 2, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonVacuum = new JButton();
        buttonVacuum.setText("SQLite Vacuum");
        tabSearchSettings.add(buttonVacuum, new GridConstraints(4, 2, 2, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelVacuumTip = new JLabel();
        labelVacuumTip.setText("Click to organize the database and reduce the size of the database");
        tabSearchSettings.add(labelVacuumTip, new GridConstraints(4, 0, 1, 2, GridConstraints.ANCHOR_SOUTHWEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonChooseFile = new JButton();
        buttonChooseFile.setText("Choose");
        tabSearchSettings.add(buttonChooseFile, new GridConstraints(9, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderSearch = new JLabel();
        placeholderSearch.setText("   ");
        tabSearchSettings.add(placeholderSearch, new GridConstraints(8, 5, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelSearchSettingsPlaceholder = new JLabel();
        labelSearchSettingsPlaceholder.setText("");
        tabSearchSettings.add(labelSearchSettingsPlaceholder, new GridConstraints(12, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelSearchSettingsPlaceholder2 = new JLabel();
        labelSearchSettingsPlaceholder2.setText("  ");
        tabSearchSettings.add(labelSearchSettingsPlaceholder2, new GridConstraints(11, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelHolder6 = new JLabel();
        labelHolder6.setText("  ");
        tabSearchSettings.add(labelHolder6, new GridConstraints(10, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        checkBoxEnableCuda = new JCheckBox();
        checkBoxEnableCuda.setOpaque(false);
        checkBoxEnableCuda.setText("Enable GPU acceleration");
        tabSearchSettings.add(checkBoxEnableCuda, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        comboBoxCudaDevice = new JComboBox();
        comboBoxCudaDevice.setOpaque(false);
        tabSearchSettings.add(comboBoxCudaDevice, new GridConstraints(0, 1, 1, 3, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelSearchThread = new JLabel();
        labelSearchThread.setText("Number of search threads");
        tabSearchSettings.add(labelSearchThread, new GridConstraints(2, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        comboBoxSearchThread = new JComboBox();
        comboBoxSearchThread.setOpaque(false);
        tabSearchSettings.add(comboBoxSearchThread, new GridConstraints(2, 1, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        tabSearchBarSettings = new JPanel();
        tabSearchBarSettings.setLayout(new GridLayoutManager(18, 9, new Insets(0, 0, 0, 0), -1, -1));
        tabSearchBarSettings.setBackground(new Color(-1));
        tabSearchBarSettings.setMaximumSize(new Dimension(1000, 600));
        tabSearchBarSettings.setMinimumSize(new Dimension(850, 500));
        tabSearchBarSettings.setOpaque(false);
        tabbedPane.addTab("Search bar settings", tabSearchBarSettings);
        tabSearchBarSettings.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        checkBoxLoseFocus = new JCheckBox();
        checkBoxLoseFocus.setOpaque(false);
        checkBoxLoseFocus.setText("Close search bar when focus lost");
        tabSearchBarSettings.add(checkBoxLoseFocus, new GridConstraints(1, 0, 1, 9, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        checkBoxAdmin = new JCheckBox();
        checkBoxAdmin.setOpaque(false);
        checkBoxAdmin.setText("Open other programs as an administrator (provided that the software has privileges)");
        tabSearchBarSettings.add(checkBoxAdmin, new GridConstraints(2, 0, 1, 9, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelPlaceHolderWhatever2 = new JLabel();
        labelPlaceHolderWhatever2.setText("");
        tabSearchBarSettings.add(labelPlaceHolderWhatever2, new GridConstraints(0, 0, 1, 5, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        checkBoxIsShowTipOnCreatingLnk = new JCheckBox();
        checkBoxIsShowTipOnCreatingLnk.setOpaque(false);
        checkBoxIsShowTipOnCreatingLnk.setText("Show tip on creating shortcut");
        tabSearchBarSettings.add(checkBoxIsShowTipOnCreatingLnk, new GridConstraints(3, 0, 1, 5, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelColorTip = new JLabel();
        labelColorTip.setText("Please enter the hexadecimal value of RGB color");
        tabSearchBarSettings.add(labelColorTip, new GridConstraints(9, 0, 1, 6, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelSearchBarFontColor = new JLabel();
        labelSearchBarFontColor.setText("SearchBar Font Color:");
        tabSearchBarSettings.add(labelSearchBarFontColor, new GridConstraints(12, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderInterface4 = new JLabel();
        placeholderInterface4.setText("   ");
        tabSearchBarSettings.add(placeholderInterface4, new GridConstraints(10, 5, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderInterface3 = new JLabel();
        placeholderInterface3.setText("   ");
        tabSearchBarSettings.add(placeholderInterface3, new GridConstraints(10, 6, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderInterface2 = new JLabel();
        placeholderInterface2.setText("   ");
        tabSearchBarSettings.add(placeholderInterface2, new GridConstraints(10, 7, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderInterface1 = new JLabel();
        placeholderInterface1.setText("   ");
        tabSearchBarSettings.add(placeholderInterface1, new GridConstraints(10, 8, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelTransparency = new JLabel();
        labelTransparency.setText("Search bar opacity : ");
        tabSearchBarSettings.add(labelTransparency, new GridConstraints(5, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelBorderColor = new JLabel();
        labelBorderColor.setText("Border Color:");
        tabSearchBarSettings.add(labelBorderColor, new GridConstraints(10, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelSearchBarColor = new JLabel();
        labelSearchBarColor.setText("Search bar Color:");
        tabSearchBarSettings.add(labelSearchBarColor, new GridConstraints(11, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelLabelColor = new JLabel();
        labelLabelColor.setText("Chosen label color:");
        tabSearchBarSettings.add(labelLabelColor, new GridConstraints(13, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelFontColor = new JLabel();
        labelFontColor.setText("Chosen label font Color:");
        tabSearchBarSettings.add(labelFontColor, new GridConstraints(14, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelNotChosenFontColor = new JLabel();
        labelNotChosenFontColor.setText("Unchosen label font Color:");
        tabSearchBarSettings.add(labelNotChosenFontColor, new GridConstraints(15, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelDefaultColor = new JLabel();
        labelDefaultColor.setText("Default background Color:");
        tabSearchBarSettings.add(labelDefaultColor, new GridConstraints(16, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldBorderColor = new JTextField();
        textFieldBorderColor.setOpaque(false);
        tabSearchBarSettings.add(textFieldBorderColor, new GridConstraints(10, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        textFieldSearchBarColor = new JTextField();
        textFieldSearchBarColor.setOpaque(false);
        tabSearchBarSettings.add(textFieldSearchBarColor, new GridConstraints(11, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        textFieldSearchBarFontColor = new JTextField();
        textFieldSearchBarFontColor.setOpaque(false);
        tabSearchBarSettings.add(textFieldSearchBarFontColor, new GridConstraints(12, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        textFieldLabelColor = new JTextField();
        textFieldLabelColor.setOpaque(false);
        tabSearchBarSettings.add(textFieldLabelColor, new GridConstraints(13, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        textFieldFontColorWithCoverage = new JTextField();
        textFieldFontColorWithCoverage.setOpaque(false);
        tabSearchBarSettings.add(textFieldFontColorWithCoverage, new GridConstraints(14, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        textFieldFontColor = new JTextField();
        textFieldFontColor.setOpaque(false);
        tabSearchBarSettings.add(textFieldFontColor, new GridConstraints(15, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        textFieldBackgroundDefault = new JTextField();
        textFieldBackgroundDefault.setOpaque(false);
        tabSearchBarSettings.add(textFieldBackgroundDefault, new GridConstraints(16, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        labelColorChooser = new JLabel();
        labelColorChooser.setOpaque(true);
        labelColorChooser.setText("preview");
        tabSearchBarSettings.add(labelColorChooser, new GridConstraints(13, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        FontColorWithCoverageChooser = new JLabel();
        FontColorWithCoverageChooser.setOpaque(true);
        FontColorWithCoverageChooser.setText("preview");
        tabSearchBarSettings.add(FontColorWithCoverageChooser, new GridConstraints(14, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        FontColorChooser = new JLabel();
        FontColorChooser.setOpaque(true);
        FontColorChooser.setText("preview");
        tabSearchBarSettings.add(FontColorChooser, new GridConstraints(15, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        defaultBackgroundChooser = new JLabel();
        defaultBackgroundChooser.setOpaque(true);
        defaultBackgroundChooser.setText("preview");
        tabSearchBarSettings.add(defaultBackgroundChooser, new GridConstraints(16, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonPreviewColor = new JButton();
        buttonPreviewColor.setText("Preview");
        tabSearchBarSettings.add(buttonPreviewColor, new GridConstraints(14, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonClosePreview = new JButton();
        buttonClosePreview.setText("Close preview window");
        tabSearchBarSettings.add(buttonClosePreview, new GridConstraints(15, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonResetColor = new JButton();
        buttonResetColor.setText("Reset to default");
        tabSearchBarSettings.add(buttonResetColor, new GridConstraints(16, 4, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        SearchBarFontColorChooser = new JLabel();
        SearchBarFontColorChooser.setOpaque(true);
        SearchBarFontColorChooser.setText("preview");
        tabSearchBarSettings.add(SearchBarFontColorChooser, new GridConstraints(12, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        searchBarColorChooser = new JLabel();
        searchBarColorChooser.setOpaque(true);
        searchBarColorChooser.setText("preview");
        tabSearchBarSettings.add(searchBarColorChooser, new GridConstraints(11, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        borderColorChooser = new JLabel();
        borderColorChooser.setOpaque(true);
        borderColorChooser.setText("preview");
        tabSearchBarSettings.add(borderColorChooser, new GridConstraints(10, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer2 = new Spacer();
        tabSearchBarSettings.add(spacer2, new GridConstraints(13, 4, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        final Spacer spacer3 = new Spacer();
        tabSearchBarSettings.add(spacer3, new GridConstraints(13, 3, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        final Spacer spacer4 = new Spacer();
        tabSearchBarSettings.add(spacer4, new GridConstraints(13, 5, 1, 4, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        textFieldTransparency = new JTextField();
        textFieldTransparency.setOpaque(false);
        tabSearchBarSettings.add(textFieldTransparency, new GridConstraints(5, 1, 1, 4, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        final Spacer spacer5 = new Spacer();
        tabSearchBarSettings.add(spacer5, new GridConstraints(17, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        checkBoxIsAttachExplorer = new JCheckBox();
        checkBoxIsAttachExplorer.setOpaque(false);
        checkBoxIsAttachExplorer.setText("Attach explorer");
        tabSearchBarSettings.add(checkBoxIsAttachExplorer, new GridConstraints(4, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelRoundRadius = new JLabel();
        labelRoundRadius.setText("Round rectangle radius:");
        tabSearchBarSettings.add(labelRoundRadius, new GridConstraints(6, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldRoundRadius = new JTextField();
        textFieldRoundRadius.setOpaque(false);
        tabSearchBarSettings.add(textFieldRoundRadius, new GridConstraints(6, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        labelBorderType = new JLabel();
        labelBorderType.setText("Border Type");
        tabSearchBarSettings.add(labelBorderType, new GridConstraints(7, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        comboBoxBorderType = new JComboBox();
        comboBoxBorderType.setOpaque(false);
        tabSearchBarSettings.add(comboBoxBorderType, new GridConstraints(7, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelBorderThickness = new JLabel();
        labelBorderThickness.setText("Border Thickness");
        tabSearchBarSettings.add(labelBorderThickness, new GridConstraints(8, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldBorderThickness = new JTextField();
        textFieldBorderThickness.setOpaque(false);
        tabSearchBarSettings.add(textFieldBorderThickness, new GridConstraints(8, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        tabModifyPriority = new JPanel();
        tabModifyPriority.setLayout(new GridLayoutManager(5, 4, new Insets(0, 0, 0, 0), -1, -1));
        tabbedPane.addTab("Modify suffix priority", tabModifyPriority);
        suffixScrollpane = new JScrollPane();
        tabModifyPriority.add(suffixScrollpane, new GridConstraints(1, 0, 3, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        suffixScrollpane.setBorder(BorderFactory.createTitledBorder(null, "", TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        tableSuffix = new JTable();
        suffixScrollpane.setViewportView(tableSuffix);
        buttonDeleteSuffix = new JButton();
        buttonDeleteSuffix.setText("Delete");
        tabModifyPriority.add(buttonDeleteSuffix, new GridConstraints(2, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonAddSuffix = new JButton();
        buttonAddSuffix.setText("Add");
        tabModifyPriority.add(buttonAddSuffix, new GridConstraints(1, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonDeleteAllSuffix = new JButton();
        buttonDeleteAllSuffix.setText("Delete all");
        tabModifyPriority.add(buttonDeleteAllSuffix, new GridConstraints(3, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelSuffixTip = new JLabel();
        labelSuffixTip.setText("Modifying the suffix priority requires rebuilding the index (input \":update\") to take effect");
        tabModifyPriority.add(labelSuffixTip, new GridConstraints(0, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer6 = new Spacer();
        tabModifyPriority.add(spacer6, new GridConstraints(1, 3, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        labelpriorityPlaceholder = new JLabel();
        labelpriorityPlaceholder.setText("");
        tabModifyPriority.add(labelpriorityPlaceholder, new GridConstraints(4, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        tabCache = new JPanel();
        tabCache.setLayout(new GridLayoutManager(10, 4, new Insets(0, 0, 0, 0), -1, -1));
        tabCache.setOpaque(false);
        tabbedPane.addTab("Cache", tabCache);
        tabCache.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        labelCacheSettings = new JLabel();
        labelCacheSettings.setText("Cache Settings");
        tabCache.add(labelCacheSettings, new GridConstraints(0, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelCacheTip = new JLabel();
        labelCacheTip.setText("You can edit the saved cacheshere.");
        tabCache.add(labelCacheTip, new GridConstraints(1, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelCacheTip2 = new JLabel();
        labelCacheTip2.setText("The cache is automatically generated by the software and will be displayed first when searching.");
        tabCache.add(labelCacheTip2, new GridConstraints(2, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        cacheScrollPane = new JScrollPane();
        tabCache.add(cacheScrollPane, new GridConstraints(5, 0, 3, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        listCache = new JList();
        listCache.setSelectionMode(0);
        cacheScrollPane.setViewportView(listCache);
        buttonDeleteCache = new JButton();
        buttonDeleteCache.setText("Delete cache");
        tabCache.add(buttonDeleteCache, new GridConstraints(5, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer7 = new Spacer();
        tabCache.add(spacer7, new GridConstraints(5, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        buttonDeleteAllCache = new JButton();
        buttonDeleteAllCache.setLabel("Delete all");
        buttonDeleteAllCache.setText("Delete all");
        tabCache.add(buttonDeleteAllCache, new GridConstraints(6, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer8 = new Spacer();
        tabCache.add(spacer8, new GridConstraints(7, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        labelCurrentCacheNum = new JLabel();
        labelCurrentCacheNum.setText("Current Caches Num:");
        tabCache.add(labelCurrentCacheNum, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderN = new JLabel();
        placeholderN.setText("");
        tabCache.add(placeholderN, new GridConstraints(5, 3, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer9 = new Spacer();
        tabCache.add(spacer9, new GridConstraints(3, 3, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        labelCachePlaceholder = new JLabel();
        labelCachePlaceholder.setText("  ");
        tabCache.add(labelCachePlaceholder, new GridConstraints(9, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldSearchCache = new JTextField();
        textFieldSearchCache.setOpaque(false);
        tabCache.add(textFieldSearchCache, new GridConstraints(4, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        labelHolder5 = new JLabel();
        labelHolder5.setText("   ");
        tabCache.add(labelHolder5, new GridConstraints(8, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        tabProxy = new JPanel();
        tabProxy.setLayout(new GridLayoutManager(9, 2, new Insets(0, 0, 0, 0), -1, -1));
        tabProxy.setOpaque(false);
        tabbedPane.addTab("Proxy settings", tabProxy);
        tabProxy.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        final Spacer spacer10 = new Spacer();
        tabProxy.add(spacer10, new GridConstraints(8, 0, 1, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        labelProxyTip = new JLabel();
        labelProxyTip.setText("If you need a proxy to access the Internet, You can add a proxy here.");
        tabProxy.add(labelProxyTip, new GridConstraints(0, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelPlaceHolder8 = new JLabel();
        labelPlaceHolder8.setText("      ");
        tabProxy.add(labelPlaceHolder8, new GridConstraints(2, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelAddress = new JLabel();
        labelAddress.setText("Address");
        tabProxy.add(labelAddress, new GridConstraints(4, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldAddress = new JTextField();
        textFieldAddress.setOpaque(false);
        tabProxy.add(textFieldAddress, new GridConstraints(4, 1, 1, 1, GridConstraints.ANCHOR_NORTHWEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        textFieldPort = new JTextField();
        textFieldPort.setOpaque(false);
        tabProxy.add(textFieldPort, new GridConstraints(5, 1, 1, 1, GridConstraints.ANCHOR_NORTHWEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        labelPort = new JLabel();
        labelPort.setText("Port");
        tabProxy.add(labelPort, new GridConstraints(5, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldUserName = new JTextField();
        textFieldUserName.setOpaque(false);
        tabProxy.add(textFieldUserName, new GridConstraints(6, 1, 1, 1, GridConstraints.ANCHOR_NORTHWEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        labelUserName = new JLabel();
        labelUserName.setText("User name");
        tabProxy.add(labelUserName, new GridConstraints(6, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldPassword = new JTextField();
        textFieldPassword.setOpaque(false);
        tabProxy.add(textFieldPassword, new GridConstraints(7, 1, 1, 1, GridConstraints.ANCHOR_NORTHWEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        labelPassword = new JLabel();
        labelPassword.setText("Password");
        tabProxy.add(labelPassword, new GridConstraints(7, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        radioButtonProxyTypeSocks5 = new JRadioButton();
        radioButtonProxyTypeSocks5.setOpaque(false);
        radioButtonProxyTypeSocks5.setSelected(false);
        radioButtonProxyTypeSocks5.setText("Socks5");
        tabProxy.add(radioButtonProxyTypeSocks5, new GridConstraints(3, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        radioButtonProxyTypeHttp = new JRadioButton();
        radioButtonProxyTypeHttp.setOpaque(false);
        radioButtonProxyTypeHttp.setText("Http");
        tabProxy.add(radioButtonProxyTypeHttp, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        radioButtonUseProxy = new JRadioButton();
        radioButtonUseProxy.setOpaque(false);
        radioButtonUseProxy.setText("Configure proxy");
        tabProxy.add(radioButtonUseProxy, new GridConstraints(1, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        radioButtonNoProxy = new JRadioButton();
        radioButtonNoProxy.setOpaque(false);
        radioButtonNoProxy.setText("No proxy");
        tabProxy.add(radioButtonNoProxy, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        tabPlugin = new JPanel();
        tabPlugin.setLayout(new GridBagLayout());
        tabPlugin.setMaximumSize(new Dimension(1000, 600));
        tabPlugin.setMinimumSize(new Dimension(850, 500));
        tabPlugin.setOpaque(false);
        tabbedPane.addTab("Plugins", tabPlugin);
        tabPlugin.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        PluginSettingsPanel = new JPanel();
        PluginSettingsPanel.setLayout(new GridLayoutManager(8, 8, new Insets(0, 0, 0, 0), -1, -1));
        PluginSettingsPanel.setOpaque(false);
        GridBagConstraints gbc;
        gbc = new GridBagConstraints();
        gbc.gridx = 1;
        gbc.gridy = 3;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        gbc.fill = GridBagConstraints.BOTH;
        tabPlugin.add(PluginSettingsPanel, gbc);
        PluginSettingsPanel.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        PluginIconLabel = new JLabel();
        PluginIconLabel.setText("");
        PluginSettingsPanel.add(PluginIconLabel, new GridConstraints(2, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        PluginNamelabel = new JLabel();
        PluginNamelabel.setText("");
        PluginSettingsPanel.add(PluginNamelabel, new GridConstraints(2, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelPluginVersion = new JLabel();
        labelPluginVersion.setText("");
        PluginSettingsPanel.add(labelPluginVersion, new GridConstraints(4, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        scrollPane = new JScrollPane();
        PluginSettingsPanel.add(scrollPane, new GridConstraints(5, 0, 1, 7, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        textAreaDescription = new JTextArea();
        textAreaDescription.setEditable(false);
        textAreaDescription.setEnabled(true);
        textAreaDescription.setForeground(new Color(-4473925));
        textAreaDescription.setLineWrap(true);
        textAreaDescription.setOpaque(false);
        textAreaDescription.setWrapStyleWord(true);
        scrollPane.setViewportView(textAreaDescription);
        labelAuthor = new JLabel();
        labelAuthor.setText("");
        PluginSettingsPanel.add(labelAuthor, new GridConstraints(4, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelOfficialSite = new JLabel();
        labelOfficialSite.setText("");
        PluginSettingsPanel.add(labelOfficialSite, new GridConstraints(4, 3, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelApiVersion = new JLabel();
        labelApiVersion.setText("");
        PluginSettingsPanel.add(labelApiVersion, new GridConstraints(4, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelPlaceHolder12 = new JLabel();
        labelPlaceHolder12.setText("");
        PluginSettingsPanel.add(labelPlaceHolder12, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelUninstallPluginTip = new JLabel();
        labelUninstallPluginTip.setText("If you need to delete a plug-in, just delete it under the \"plugins\" folder in the software directory.");
        PluginSettingsPanel.add(labelUninstallPluginTip, new GridConstraints(0, 0, 1, 6, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderPlugin2 = new JLabel();
        placeholderPlugin2.setText("   ");
        PluginSettingsPanel.add(placeholderPlugin2, new GridConstraints(7, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderPlugin1 = new JLabel();
        placeholderPlugin1.setText("         ");
        PluginSettingsPanel.add(placeholderPlugin1, new GridConstraints(5, 7, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderPlugins3 = new JLabel();
        placeholderPlugins3.setText("    ");
        PluginSettingsPanel.add(placeholderPlugins3, new GridConstraints(4, 6, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonUpdatePlugin = new JButton();
        buttonUpdatePlugin.setText("Check for Update");
        PluginSettingsPanel.add(buttonUpdatePlugin, new GridConstraints(2, 5, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProgress = new JLabel();
        labelProgress.setText("    ");
        PluginSettingsPanel.add(labelProgress, new GridConstraints(3, 5, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelHolder2 = new JLabel();
        labelHolder2.setText("   ");
        PluginSettingsPanel.add(labelHolder2, new GridConstraints(6, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonPluginSettings = new JButton();
        buttonPluginSettings.setText("Settings");
        PluginSettingsPanel.add(buttonPluginSettings, new GridConstraints(2, 4, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        PluginListPanel = new JPanel();
        PluginListPanel.setLayout(new GridLayoutManager(4, 1, new Insets(0, 0, 0, 0), -1, -1));
        PluginListPanel.setOpaque(false);
        gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.gridheight = 2;
        gbc.weighty = 1.0;
        gbc.fill = GridBagConstraints.BOTH;
        tabPlugin.add(PluginListPanel, gbc);
        paneListPlugins = new JScrollPane();
        PluginListPanel.add(paneListPlugins, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        listPlugins = new JList();
        listPlugins.setFocusable(false);
        listPlugins.setOpaque(false);
        listPlugins.setSelectionMode(1);
        paneListPlugins.setViewportView(listPlugins);
        buttonPluginMarket = new JButton();
        buttonPluginMarket.setText("Plugin Market");
        PluginListPanel.add(buttonPluginMarket, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderPlugin5 = new JLabel();
        placeholderPlugin5.setText("    ");
        PluginListPanel.add(placeholderPlugin5, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelHolder = new JLabel();
        labelHolder.setText("     ");
        PluginListPanel.add(labelHolder, new GridConstraints(2, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelInstalledPluginNum = new JLabel();
        labelInstalledPluginNum.setText("Installed plugins num:");
        gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        tabPlugin.add(labelInstalledPluginNum, gbc);
        labelPluginNum = new JLabel();
        labelPluginNum.setText("Label");
        gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 1;
        tabPlugin.add(labelPluginNum, gbc);
        labelUninstallPluginTip2 = new JLabel();
        labelUninstallPluginTip2.setText("Tip:");
        gbc = new GridBagConstraints();
        gbc.gridx = 1;
        gbc.gridy = 2;
        gbc.anchor = GridBagConstraints.WEST;
        tabPlugin.add(labelUninstallPluginTip2, gbc);
        tabHotKey = new JPanel();
        tabHotKey.setLayout(new GridLayoutManager(7, 3, new Insets(0, 0, 0, 0), -1, -1));
        tabHotKey.setBackground(new Color(-1));
        tabHotKey.setMaximumSize(new Dimension(1000, 600));
        tabHotKey.setMinimumSize(new Dimension(850, 500));
        tabHotKey.setOpaque(false);
        tabbedPane.addTab("Hotkey settings", tabHotKey);
        tabHotKey.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        textFieldHotkey = new JTextField();
        textFieldHotkey.setEditable(false);
        textFieldHotkey.setEnabled(true);
        textFieldHotkey.setOpaque(false);
        tabHotKey.add(textFieldHotkey, new GridConstraints(2, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        labelOpenSearchBarHotKey = new JLabel();
        labelOpenSearchBarHotKey.setText("Open search bar:");
        tabHotKey.add(labelOpenSearchBarHotKey, new GridConstraints(2, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelRunAsAdminHotKey = new JLabel();
        labelRunAsAdminHotKey.setText("Run as administrator:");
        tabHotKey.add(labelRunAsAdminHotKey, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldRunAsAdminHotKey = new JTextField();
        textFieldRunAsAdminHotKey.setEditable(false);
        textFieldRunAsAdminHotKey.setOpaque(false);
        tabHotKey.add(textFieldRunAsAdminHotKey, new GridConstraints(3, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        labelCopyPathHotKey = new JLabel();
        labelCopyPathHotKey.setText("Copy path:");
        tabHotKey.add(labelCopyPathHotKey, new GridConstraints(5, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldOpenLastFolder = new JTextField();
        textFieldOpenLastFolder.setEditable(false);
        textFieldOpenLastFolder.setOpaque(false);
        tabHotKey.add(textFieldOpenLastFolder, new GridConstraints(4, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        labelOpenFolderHotKey = new JLabel();
        labelOpenFolderHotKey.setText("Open the parent folder:");
        tabHotKey.add(labelOpenFolderHotKey, new GridConstraints(4, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        textFieldCopyPath = new JTextField();
        textFieldCopyPath.setEditable(false);
        textFieldCopyPath.setOpaque(false);
        tabHotKey.add(textFieldCopyPath, new GridConstraints(5, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, -1), null, 0, false));
        final Spacer spacer11 = new Spacer();
        tabHotKey.add(spacer11, new GridConstraints(2, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        checkBoxResponseCtrl = new JCheckBox();
        checkBoxResponseCtrl.setOpaque(false);
        checkBoxResponseCtrl.setText("Double-click \"Ctrl\" to open the search bar");
        tabHotKey.add(checkBoxResponseCtrl, new GridConstraints(0, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer12 = new Spacer();
        tabHotKey.add(spacer12, new GridConstraints(6, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        placeholderHotkey1 = new JLabel();
        placeholderHotkey1.setText("    ");
        tabHotKey.add(placeholderHotkey1, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        tabLanguage = new JPanel();
        tabLanguage.setLayout(new GridLayoutManager(5, 3, new Insets(0, 0, 0, 0), -1, -1));
        tabLanguage.setBackground(new Color(-1));
        tabLanguage.setEnabled(true);
        tabLanguage.setMaximumSize(new Dimension(1000, 600));
        tabLanguage.setMinimumSize(new Dimension(850, 500));
        tabLanguage.setOpaque(false);
        tabbedPane.addTab("Language settings", tabLanguage);
        tabLanguage.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        labelPlaceHolder4 = new JLabel();
        labelPlaceHolder4.setText("");
        tabLanguage.add(labelPlaceHolder4, new GridConstraints(1, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        paneListLanguage = new JScrollPane();
        tabLanguage.add(paneListLanguage, new GridConstraints(2, 0, 1, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        listLanguage = new JList();
        listLanguage.setOpaque(false);
        listLanguage.setSelectionMode(0);
        paneListLanguage.setViewportView(listLanguage);
        placeholderl = new JLabel();
        placeholderl.setText("     ");
        tabLanguage.add(placeholderl, new GridConstraints(1, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer13 = new Spacer();
        tabLanguage.add(spacer13, new GridConstraints(0, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        labelLanguageChooseTip = new JLabel();
        labelLanguageChooseTip.setText("Choose a language");
        tabLanguage.add(labelLanguageChooseTip, new GridConstraints(0, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelTranslationTip = new JLabel();
        labelTranslationTip.setText("The translation might not be 100% correct");
        tabLanguage.add(labelTranslationTip, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelLanguagePlaceholder = new JLabel();
        labelLanguagePlaceholder.setText(" ");
        tabLanguage.add(labelLanguagePlaceholder, new GridConstraints(4, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelHolder3 = new JLabel();
        labelHolder3.setText("  ");
        tabLanguage.add(labelHolder3, new GridConstraints(3, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        tabCommands = new JPanel();
        tabCommands.setLayout(new GridLayoutManager(7, 3, new Insets(0, 0, 0, 0), -1, -1));
        tabCommands.setBackground(new Color(-1));
        tabCommands.setMaximumSize(new Dimension(1000, 600));
        tabCommands.setMinimumSize(new Dimension(850, 500));
        tabCommands.setOpaque(false);
        tabbedPane.addTab("My commands", tabCommands);
        tabCommands.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        scrollPaneCmd = new JScrollPane();
        tabCommands.add(scrollPaneCmd, new GridConstraints(2, 0, 3, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        listCmds = new JList();
        listCmds.setOpaque(false);
        scrollPaneCmd.setViewportView(listCmds);
        buttonAddCMD = new JButton();
        buttonAddCMD.setText("Add");
        tabCommands.add(buttonAddCMD, new GridConstraints(3, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonDelCmd = new JButton();
        buttonDelCmd.setText("Delete");
        tabCommands.add(buttonDelCmd, new GridConstraints(4, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        labelSearchCommand = new JLabel();
        labelSearchCommand.setText("Search");
        tabCommands.add(labelSearchCommand, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(39, 38), null, 0, false));
        textFieldSearchCommands = new JTextField();
        textFieldSearchCommands.setOpaque(false);
        tabCommands.add(textFieldSearchCommands, new GridConstraints(1, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(150, 38), null, 0, false));
        labelCmdTip2 = new JLabel();
        labelCmdTip2.setText("You can add custom commands here. After adding, you can enter \": + your set identifier\" in the search box to quickly open");
        labelCmdTip2.setVerticalTextPosition(0);
        tabCommands.add(labelCmdTip2, new GridConstraints(0, 1, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, new Dimension(717, 35), null, 0, false));
        labelCommandsPlaceholder = new JLabel();
        labelCommandsPlaceholder.setText("  ");
        tabCommands.add(labelCommandsPlaceholder, new GridConstraints(6, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelHolder4 = new JLabel();
        labelHolder4.setText("   ");
        tabCommands.add(labelHolder4, new GridConstraints(5, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        tabAbout = new JPanel();
        tabAbout.setLayout(new GridLayoutManager(20, 12, new Insets(0, 0, 0, 0), -1, -1));
        tabAbout.setMaximumSize(new Dimension(1000, 600));
        tabAbout.setMinimumSize(new Dimension(850, 500));
        tabAbout.setOpaque(false);
        tabbedPane.addTab("About", tabAbout);
        tabAbout.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEmptyBorder(), null, TitledBorder.DEFAULT_JUSTIFICATION, TitledBorder.DEFAULT_POSITION, null, null));
        final Spacer spacer14 = new Spacer();
        tabAbout.add(spacer14, new GridConstraints(0, 11, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        final Spacer spacer15 = new Spacer();
        tabAbout.add(spacer15, new GridConstraints(19, 0, 1, 5, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        labelIcon = new JLabel();
        labelIcon.setText("");
        tabAbout.add(labelIcon, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer16 = new Spacer();
        tabAbout.add(spacer16, new GridConstraints(7, 10, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        final Spacer spacer17 = new Spacer();
        tabAbout.add(spacer17, new GridConstraints(7, 9, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        final Spacer spacer18 = new Spacer();
        tabAbout.add(spacer18, new GridConstraints(7, 8, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        final Spacer spacer19 = new Spacer();
        tabAbout.add(spacer19, new GridConstraints(7, 7, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        final Spacer spacer20 = new Spacer();
        tabAbout.add(spacer20, new GridConstraints(7, 6, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        labelDownloadProgress = new JLabel();
        labelDownloadProgress.setText("");
        tabAbout.add(labelDownloadProgress, new GridConstraints(7, 5, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        chooseUpdateAddress = new JComboBox();
        chooseUpdateAddress.setOpaque(false);
        tabAbout.add(chooseUpdateAddress, new GridConstraints(7, 1, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelAbout = new JLabel();
        labelAbout.setText("File-Engine");
        tabAbout.add(labelAbout, new GridConstraints(1, 1, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelGithubIssue = new JLabel();
        labelGithubIssue.setText("If you find a bug or have some suggestions, welcome to GitHub for feedback");
        tabAbout.add(labelGithubIssue, new GridConstraints(3, 1, 1, 2, GridConstraints.ANCHOR_NORTHWEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelGitHubTip = new JLabel();
        labelGitHubTip.setText("This is an open source software,GitHub:");
        tabAbout.add(labelGitHubTip, new GridConstraints(4, 1, 1, 2, GridConstraints.ANCHOR_NORTHWEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelAboutGithub = new JLabel();
        labelAboutGithub.setText("GitHubAddress");
        tabAbout.add(labelAboutGithub, new GridConstraints(5, 1, 1, 2, GridConstraints.ANCHOR_NORTHWEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelDescription = new JLabel();
        labelDescription.setText("Thanks for the following project");
        tabAbout.add(labelDescription, new GridConstraints(8, 1, 1, 10, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonCheckUpdate = new JButton();
        buttonCheckUpdate.setText("Check for update");
        tabAbout.add(buttonCheckUpdate, new GridConstraints(7, 4, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelVersion = new JLabel();
        labelVersion.setText("labelVersion");
        tabAbout.add(labelVersion, new GridConstraints(2, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        chooseUpdateAddressLabel = new JLabel();
        chooseUpdateAddressLabel.setText("Choose update address");
        tabAbout.add(chooseUpdateAddressLabel, new GridConstraints(6, 1, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelBuildVersion = new JLabel();
        labelBuildVersion.setText("LabelBuildVersion");
        tabAbout.add(labelBuildVersion, new GridConstraints(2, 2, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer21 = new Spacer();
        tabAbout.add(spacer21, new GridConstraints(2, 3, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        labelProject1 = new JLabel();
        labelProject1.setText("Label");
        tabAbout.add(labelProject1, new GridConstraints(9, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProject2 = new JLabel();
        labelProject2.setText("Label");
        tabAbout.add(labelProject2, new GridConstraints(10, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProject3 = new JLabel();
        labelProject3.setText("Label");
        tabAbout.add(labelProject3, new GridConstraints(11, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProject4 = new JLabel();
        labelProject4.setText("Label");
        tabAbout.add(labelProject4, new GridConstraints(12, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProject5 = new JLabel();
        labelProject5.setText("Label");
        tabAbout.add(labelProject5, new GridConstraints(13, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProject6 = new JLabel();
        labelProject6.setText("Label");
        tabAbout.add(labelProject6, new GridConstraints(14, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProject7 = new JLabel();
        labelProject7.setText("Label");
        tabAbout.add(labelProject7, new GridConstraints(15, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProject8 = new JLabel();
        labelProject8.setText("Label");
        tabAbout.add(labelProject8, new GridConstraints(16, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProject9 = new JLabel();
        labelProject9.setText("Label");
        tabAbout.add(labelProject9, new GridConstraints(17, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelProject10 = new JLabel();
        labelProject10.setText("Label");
        tabAbout.add(labelProject10, new GridConstraints(18, 1, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        tabIndex = new JPanel();
        tabIndex.setLayout(new GridLayoutManager(9, 3, new Insets(0, 0, 0, 0), -1, -1));
        tabbedPane.addTab("Index", tabIndex);
        labelIndexTip = new JLabel();
        labelIndexTip.setText("You can rebuild the disk index here");
        tabIndex.add(labelIndexTip, new GridConstraints(0, 0, 1, 3, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer22 = new Spacer();
        tabIndex.add(spacer22, new GridConstraints(8, 0, 1, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_VERTICAL, 1, GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        labelIndexChooseDisk = new JLabel();
        labelIndexChooseDisk.setText("The disks listed below will be indexed");
        tabIndex.add(labelIndexChooseDisk, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        final Spacer spacer23 = new Spacer();
        tabIndex.add(spacer23, new GridConstraints(1, 2, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_WANT_GROW, 1, null, null, null, 0, false));
        buttonRebuildIndex = new JButton();
        buttonRebuildIndex.setText("Rebuild");
        tabIndex.add(buttonRebuildIndex, new GridConstraints(7, 0, 1, 2, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        placeholderIndex1 = new JLabel();
        placeholderIndex1.setText("    ");
        tabIndex.add(placeholderIndex1, new GridConstraints(6, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        scrollPaneDisks = new JScrollPane();
        tabIndex.add(scrollPaneDisks, new GridConstraints(4, 0, 2, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_BOTH, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_WANT_GROW, null, null, null, 0, false));
        listDisks = new JList();
        scrollPaneDisks.setViewportView(listDisks);
        buttonAddNewDisk = new JButton();
        buttonAddNewDisk.setText("Add");
        tabIndex.add(buttonAddNewDisk, new GridConstraints(4, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        buttonDeleteDisk = new JButton();
        buttonDeleteDisk.setText("Delete");
        tabIndex.add(buttonDeleteDisk, new GridConstraints(5, 1, 1, 1, GridConstraints.ANCHOR_CENTER, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelTipNTFSTip = new JLabel();
        labelTipNTFSTip.setText("Only supports NTFS format disks");
        tabIndex.add(labelTipNTFSTip, new GridConstraints(2, 0, 1, 1, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        labelLocalDiskTip = new JLabel();
        labelLocalDiskTip.setText("Only the disks on this machine are listed below, click \"Add\" button to add a removable disk");
        tabIndex.add(labelLocalDiskTip, new GridConstraints(3, 0, 1, 2, GridConstraints.ANCHOR_WEST, GridConstraints.FILL_NONE, GridConstraints.SIZEPOLICY_FIXED, GridConstraints.SIZEPOLICY_FIXED, null, null, null, 0, false));
        paneSwingThemes = new JScrollPane();
        paneSwingThemes.setOpaque(false);
        panel.add(paneSwingThemes, new GridConstraints(1, 0, 1, 1, GridConstraints.ANCHOR_NORTH, GridConstraints.FILL_HORIZONTAL, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, GridConstraints.SIZEPOLICY_CAN_SHRINK | GridConstraints.SIZEPOLICY_CAN_GROW, null, new Dimension(0, 0), new Dimension(-1, 0), 0, false));
        listSwingThemes = new JList();
        listSwingThemes.setOpaque(false);
        paneSwingThemes.setViewportView(listSwingThemes);
        labelMaxCacheNum.setLabelFor(textFieldCacheNum);
        labelPlaceHolder8.setLabelFor(suffixScrollpane);
    }

    /**
     * @noinspection ALL
     */
    public JComponent $$$getRootComponent$$$() {
        return panel;
    }

    private static class PreviewStatus {
        private static boolean isPreview = false;
    }

    private void addButtonClosePreviewListener() {
        buttonClosePreview.addActionListener(e -> stopPreview());
    }

    private void stopPreview() {
        eventManagement.putEvent(new StopPreviewEvent());
        eventManagement.putEvent(new HideSearchBarEvent());
        PreviewStatus.isPreview = false;
    }

    private void addButtonPreviewListener() {
        buttonPreviewColor.addActionListener(e -> {
            PreviewStatus.isPreview = true;
            eventManagement.putEvent(new StartPreviewEvent());
            threadPoolUtil.executeTask(() -> {
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
                        if (
                                canParseToRGB(borderColor) &&
                                        canParseToRGB(searchBarColor) &&
                                        canParseToRGB(searchBarFontColor) &&
                                        canParseToRGB(labelColor) &&
                                        canParseToRGB(fontColorCoverage) &&
                                        canParseToRGB(fontColor) &&
                                        canParseToRGB(defaultBackgroundColor) &&
                                        canParseFloat(borderThickness, 0, 4)
                        ) {
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
            int ret = JOptionPane.showConfirmDialog(frame, translateService.getTranslation("The operation is irreversible. Are you sure you want to clear the cache?"));
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
            eventManagement.putEvent(new ShowTaskBarMessageEvent(translateService.getTranslation("Info"), translateService.getTranslation("Updating file index")));
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
            if (JOptionPane.showConfirmDialog(frame, pane, translateService.getTranslation("Select disk"), JOptionPane.YES_NO_OPTION) == JOptionPane.YES_OPTION) {
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
            int ret = JOptionPane.showConfirmDialog(frame, translateService.getTranslation("Are you sure to delete all the suffixes") + "?");
            if (ret == JOptionPane.YES_OPTION) {
                suffixMap.clear();
                isSuffixChanged = true;
                suffixMap.put("dirPriority", -1);
                suffixMap.put("defaultPriority", 0);
                eventManagement.putEvent(new ClearSuffixPriorityMapEvent());
                refreshPriorityTable();
            }
        });
    }

    private void showOnTabbedPane(String tabName) {
        String title = translateService.getTranslation(getTabTitle(tabName));
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
                if (translateService.getTranslation("General").equals(name)) {
                    showOnTabbedPane("tabGeneral");
                } else if (translateService.getTranslation("Interface").equals(name)) {
                    showOnTabbedPane("tabSearchBarSettings");
                } else if (translateService.getTranslation("Language").equals(name)) {
                    showOnTabbedPane("tabLanguage");
                } else if (translateService.getTranslation("Suffix priority").equals(name)) {
                    showOnTabbedPane("tabModifyPriority");
                } else if (translateService.getTranslation("Search settings").equals(name)) {
                    showOnTabbedPane("tabSearchSettings");
                } else if (translateService.getTranslation("Proxy settings").equals(name)) {
                    showOnTabbedPane("tabProxy");
                } else if (translateService.getTranslation("Hotkey settings").equals(name)) {
                    showOnTabbedPane("tabHotKey");
                } else if (translateService.getTranslation("Cache").equals(name)) {
                    showOnTabbedPane("tabCache");
                } else if (translateService.getTranslation("My commands").equals(name)) {
                    showOnTabbedPane("tabCommands");
                } else if (translateService.getTranslation("Plugins").equals(name)) {
                    showOnTabbedPane("tabPlugin");
                } else if (translateService.getTranslation("About").equals(name)) {
                    showOnTabbedPane("tabAbout");
                } else if (translateService.getTranslation("Index").equals(name)) {
                    showOnTabbedPane("tabIndex");
                }
            } else {
                showOnTabbedPane("tabGeneral", translateService.getTranslation("General"));
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
                errMsg.append(translateService.getTranslation("Duplicate suffix, please check")).append("\n");
            }
        }
        if (isPriorityChanged) {
            try {
                int _p = Integer.parseInt(priority);
                if (_p <= 0) {
                    errMsg.append(translateService.getTranslation("Priority num must be positive")).append("\n");
                }
                if (isPriorityRepeat(_p)) {
                    errMsg.append(translateService.getTranslation("Duplicate priority num, please check")).append("\n");
                }
            } catch (NumberFormatException e) {
                errMsg.append(translateService.getTranslation("What you entered is not a number, please try again")).append("\n");
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
                return row != 0 && row != 1;
            }
        };
        tableSuffix.setModel(model);
        tableSuffix.getTableHeader().setReorderingAllowed(false);

        tableSuffix.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
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
            class RestoreUtil {
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
            RestoreUtil util = new RestoreUtil();

            final int column = tableSuffix.getSelectedColumn();
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
                    isSuffixChanged = true;
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
                    isSuffixChanged = true;
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
            if (current.isEmpty() || "defaultPriority".equals(current) || "dirPriority".equals(current)) {
                return;
            }
            int ret = JOptionPane.showConfirmDialog(frame, translateService.getTranslation("Do you sure want to delete this suffix") + "  --  " + current + "?");
            if (ret == JOptionPane.YES_OPTION) {
                int rowNum = tableSuffix.getSelectedRow();
                if (rowNum != -1) {
                    String suffix = (String) tableSuffix.getValueAt(rowNum, 0);
                    suffixMap.remove(suffix);
                    eventManagement.putEvent(new DeleteFromSuffixPriorityMapEvent(suffix));
                    isSuffixChanged = true;
                    refreshPriorityTable();
                }
            }
        });
    }

    private void addButtonAddSuffixListener() {
        JPanel panel = new JPanel();
        JLabel labelSuffix = new JLabel(translateService.getTranslation("Suffix") + ":");
        JLabel labelNum = new JLabel(translateService.getTranslation("Priority num") + ":");
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
            int ret = JOptionPane.showConfirmDialog(frame, panel, translateService.getTranslation("Add"), JOptionPane.YES_NO_OPTION);
            if (ret == JOptionPane.YES_OPTION) {
                String suffix = suffixName.getText().toLowerCase();
                String priorityNumTmp = priorityNum.getText();
                StringBuilder err = new StringBuilder();
                if (checkSuffixAndPriority(suffix, priorityNumTmp, err, true, true)) {
                    int num = Integer.parseInt(priorityNumTmp);
                    suffixMap.put(suffix, num);
                    eventManagement.putEvent(new AddToSuffixPriorityMapEvent(suffix, num));
                    isSuffixChanged = true;
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
     * @param checkUpdateThread 检查线程
     *                          0x100L 表示检查成功
     *                          0xFFFL 表示检查失败
     * @return true如果检查成功
     */
    private boolean waitForCheckUpdateResult(Thread checkUpdateThread, AtomicBoolean isCheckDone) {
        final long timeout = 5000L;
        final long startCheck = System.currentTimeMillis();
        while (!isCheckDone.get()) {
            if (System.currentTimeMillis() - startCheck > timeout) {
                checkUpdateThread.interrupt();
                JOptionPane.showMessageDialog(frame, translateService.getTranslation("Check update failed"));
                return false;
            }
            if (!checkUpdateThread.isAlive() && !isCheckDone.get()) {
                JOptionPane.showMessageDialog(frame, translateService.getTranslation("Check update failed"));
                return false;
            }
            if (!eventManagement.notMainExit()) {
                return false;
            }
            try {
                TimeUnit.MILLISECONDS.sleep(200);
            } catch (InterruptedException e) {
                log.error("error: {}", e.getMessage(), e);
            }
        }
        return true;
    }

    private void addButtonOpenPluginSettingsListener() {
        buttonPluginSettings.addActionListener(e -> {
            String pluginName = (String) listPlugins.getSelectedValue();
            eventManagement.putEvent(new GetPluginByNameEvent(pluginName), event -> {
                GetPluginByNameEvent getPluginByNameEvent = (GetPluginByNameEvent) event;
                Optional<PluginService.PluginInfo> returnValue = getPluginByNameEvent.getReturnValue();
                returnValue.ifPresent(pluginInfo -> pluginInfo.plugin.openSettings());
            }, null);
        });
    }

    private void addButtonPluginUpdateCheckListener() {
        AtomicBoolean isVersionLatest = new AtomicBoolean(true);
        AtomicBoolean isSkipConfirm = new AtomicBoolean(false);

        HashMap<String, DownloadManager> pluginInfoMap = new HashMap<>();
        PluginService pluginService = PluginService.getInstance();

        buttonUpdatePlugin.addActionListener(e -> {
            String pluginName = (String) listPlugins.getSelectedValue();
            var downloadManagerContainer = new Object() {
                DownloadManager downloadManager = pluginInfoMap.get(pluginName);
            };
            if (downloadManagerContainer.downloadManager != null &&
                    downloadManagerContainer.downloadManager.getDownloadStatus() == Constants.Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                eventManagement.putEvent(new StopDownloadEvent(downloadManagerContainer.downloadManager));
            } else {
                var getPluginByNameEvent = new GetPluginByNameEvent(pluginName);
                eventManagement.putEvent(getPluginByNameEvent);
                eventManagement.waitForEvent(getPluginByNameEvent);
                Optional<PluginService.PluginInfo> pluginInfoOptional = getPluginByNameEvent.getReturnValue();
                pluginInfoOptional.ifPresent(pluginInfo -> {
                    AtomicBoolean isCheckDone = new AtomicBoolean();
                    Plugin plugin = pluginInfo.plugin;
                    String pluginFullName = pluginName + ".jar";

                    if (pluginService.hasPluginNotLatest(pluginName)) {
                        //已经检查过
                        isVersionLatest.set(false);
                        isSkipConfirm.set(true);
                        isCheckDone.set(true);
                    } else {
                        Thread checkUpdateThread = new Thread(() -> {
                            try {
                                isVersionLatest.set(plugin.isLatest());
                                if (!Thread.interrupted()) {
                                    isCheckDone.set(true); //表示检查成功
                                }
                            } catch (Exception ex) {
                                log.error(ex.getMessage(), ex);
                            }
                        });
                        checkUpdateThread.start();
                        //等待获取插件更新信息
                        if (!waitForCheckUpdateResult(checkUpdateThread, isCheckDone)) {
                            return;
                        }
                    }
                    if (!isCheckDone.get()) {
                        return;
                    }
                    if (isVersionLatest.get()) {
                        JOptionPane.showMessageDialog(frame, translateService.getTranslation("Latest version:") + plugin.getVersion() + "\n" + translateService.getTranslation("The current version is the latest"));
                        return;
                    }
                    if (!isSkipConfirm.get()) {
                        eventManagement.putEvent(new AddPluginsCanUpdateEvent(pluginName));
                        int ret = JOptionPane.showConfirmDialog(frame, translateService.getTranslation("New version available, do you want to update?"));
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
                    threadPoolUtil.executeTask(() ->
                            SetDownloadProgress.setProgress(labelProgress,
                                    buttonUpdatePlugin,
                                    finalDownloadManager,
                                    () -> finalDownloadManager.fileName.equals(listPlugins.getSelectedValue() + ".jar"),
                                    () -> {
                                        File updatePluginSign = new File("user/updatePlugin");
                                        if (!updatePluginSign.exists()) {
                                            try {
                                                if (updatePluginSign.createNewFile()) {
                                                    throw new RuntimeException("create user/updatePlugin file failed.");
                                                }
                                            } catch (IOException ex) {
                                                throw new RuntimeException(ex);
                                            }
                                        }
                                    }));
                });
            }
        });
    }

    private void setLabelGui() {
        String template = "<html>&nbsp;&nbsp;&nbsp;&nbsp;<a href='%s'><font size=\"4\">%s</font></a></html>";
        labelProject1.setText(String.format(template, "https://github.com/google/gson", "google/gson"));
        labelProject2.setText(String.format(template, "https://github.com/JFormDesigner/FlatLaf", "JFormDesigner/FlatLaf"));
        labelProject3.setText(String.format(template, "https://github.com/xerial/sqlite-jdbc", "xerial/sqlite-jdbc"));
        labelProject4.setText(String.format(template, "https://projectlombok.org", "lombok"));
        labelProject5.setText(String.format(template, "https://github.com/promeG/TinyPinyin", "promeG/TinyPinyin"));
        labelProject6.setText(String.format(template, "https://github.com/kuba--/zip", "kuba--/zip"));
        labelProject7.setText(String.format(template, "https://github.com/ProjectPhysX/OpenCL-Wrapper", "ProjectPhysX/OpenCLWrapper"));
        labelProject8.setText(String.format(template, "https://github.com/oshi/oshi", "oshi/oshi"));
        labelProject9.setText(String.format(template, "https://github.com/java-native-access/jna", "java-native-access/jna"));
        labelProject10.setText(String.format(template, "https://javalin.io/", "javalin/javalin"));
        labelAboutGithub.setText(String.format(template, "https://github.com/XUANXUQAQ/File-Engine", "File-Engine"));
        labelPluginNum.setText(String.valueOf(PluginService.getInstance().getInstalledPluginNum()));
        ImageIcon imageIcon = new ImageIcon(Objects.requireNonNull(SettingsFrame.class.getResource("/icons/frame.png")));
        labelIcon.setIcon(imageIcon);
        labelVersion.setText(translateService.getTranslation("Current Version:") + Constants.version);
        labelBuildVersion.setText(Constants.buildVersion);
        int cacheNum = cacheSet != null ? cacheSet.size() : 0;
        labelCurrentCacheNum.setText(translateService.getTranslation("Current Caches Num:") + cacheNum);
    }

    /**
     * 初始化textField的显示
     */
    private void setTextFieldAndTextAreaGui() {
        var configs = allConfigs.getConfigEntity();
        CoreConfigEntity coreConfigs = allConfigs.getCoreConfigs();
        textFieldSearchCache.setText("");
        textFieldBackgroundDefault.setText(toRGBHexString(configs.getDefaultBackgroundColor()));
        textFieldLabelColor.setText(toRGBHexString(configs.getLabelColor()));
        textFieldFontColorWithCoverage.setText(toRGBHexString(configs.getFontColorWithCoverage()));
        textFieldTransparency.setText(String.valueOf(configs.getTransparency()));
        textFieldBorderColor.setText(toRGBHexString(configs.getBorderColor()));
        textFieldFontColor.setText(toRGBHexString(configs.getFontColor()));
        textFieldSearchBarFontColor.setText(toRGBHexString(configs.getSearchBarFontColor()));
        textFieldCacheNum.setText(String.valueOf(coreConfigs.getCacheNumLimit()));
        textFieldHotkey.setText(configs.getHotkey());
        textFieldRoundRadius.setText(String.valueOf(configs.getRoundRadius()));
        textFieldPriorityFolder.setText(coreConfigs.getPriorityFolder());
        textFieldUpdateInterval.setText(String.valueOf(coreConfigs.getUpdateTimeLimit()));
        textFieldSearchBarColor.setText(toRGBHexString(configs.getSearchBarColor()));
        textFieldAddress.setText(configs.getProxyAddress());
        textFieldPort.setText(String.valueOf(configs.getProxyPort()));
        textFieldUserName.setText(configs.getProxyUserName());
        textFieldPassword.setText(configs.getProxyPassword());
        textFieldBorderThickness.setText(String.valueOf(configs.getBorderThickness()));
        textAreaIgnorePath.setText(RegexUtil.comma.matcher(coreConfigs.getIgnorePath()).replaceAll(",\n"));
        if (configs.getRunAsAdminKeyCode() == 17) {
            textFieldRunAsAdminHotKey.setText("Ctrl + Enter");
        } else if (configs.getRunAsAdminKeyCode() == 16) {
            textFieldRunAsAdminHotKey.setText("Shift + Enter");
        } else if (configs.getRunAsAdminKeyCode() == 18) {
            textFieldRunAsAdminHotKey.setText("Alt + Enter");
        }
        if (configs.getOpenLastFolderKeyCode() == 17) {
            textFieldOpenLastFolder.setText("Ctrl + Enter");
        } else if (configs.getOpenLastFolderKeyCode() == 16) {
            textFieldOpenLastFolder.setText("Shift + Enter");
        } else if (configs.getOpenLastFolderKeyCode() == 18) {
            textFieldOpenLastFolder.setText("Alt + Enter");
        }
        if (configs.getCopyPathKeyCode() == 17) {
            textFieldCopyPath.setText("Ctrl + Enter");
        } else if (configs.getCopyPathKeyCode() == 16) {
            textFieldCopyPath.setText("Shift + Enter");
        } else if (configs.getCopyPathKeyCode() == 18) {
            textFieldCopyPath.setText("Alt + Enter");
        }
    }

    /**
     * 初始化颜色选择器的显示
     */
    private void setColorChooserGui() {
        var configs = allConfigs.getConfigEntity();
        Color tmp_searchBarColor = new Color(configs.getSearchBarColor());
        searchBarColorChooser.setBackground(tmp_searchBarColor);
        searchBarColorChooser.setForeground(tmp_searchBarColor);

        Color tmp_defaultBackgroundColor = new Color(configs.getDefaultBackgroundColor());
        defaultBackgroundChooser.setBackground(tmp_defaultBackgroundColor);
        defaultBackgroundChooser.setForeground(tmp_defaultBackgroundColor);

        Color tmp_labelColor = new Color(configs.getLabelColor());
        labelColorChooser.setBackground(tmp_labelColor);
        labelColorChooser.setForeground(tmp_labelColor);

        Color tmp_fontColorWithCoverage = new Color(configs.getFontColorWithCoverage());
        FontColorWithCoverageChooser.setBackground(tmp_fontColorWithCoverage);
        FontColorWithCoverageChooser.setForeground(tmp_fontColorWithCoverage);

        Color tmp_fontColor = new Color(configs.getFontColor());
        FontColorChooser.setBackground(tmp_fontColor);
        FontColorChooser.setForeground(tmp_fontColor);

        Color tmp_searchBarFontColor = new Color(configs.getSearchBarFontColor());
        SearchBarFontColorChooser.setBackground(tmp_searchBarFontColor);
        SearchBarFontColorChooser.setForeground(tmp_searchBarFontColor);

        Color tmp_borderColor = new Color(configs.getBorderColor());
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
    private String getSuffixByValue(Map<String, Integer> suffixPriorityMap, int val) {
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
        tableModel.setColumnIdentifiers(new String[]{translateService.getTranslation("suffix"), translateService.getTranslation("priority")});
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

    private void setComboBoxGui() {
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        ArrayList<Integer> threads = new ArrayList<>();
        for (int i = 1; i <= availableProcessors * 2; i++) {
            threads.add(i);
        }
        CoreConfigEntity coreConfigs = allConfigs.getCoreConfigs();
        comboBoxSearchThread.setModel(new DefaultComboBoxModel<>(new Vector<>(threads)));
        comboBoxSearchThread.setSelectedItem(coreConfigs.getSearchThreadNumber());
        Map<String, String> gpuDevices = DatabaseNativeService.getGpuDevices();
        comboBoxCudaDevice.setEnabled(!gpuDevices.isEmpty());
        if (!gpuDevices.isEmpty()) {
            cudaDeviceMap = gpuDevices;
            comboBoxCudaDevice.setModel(new DefaultComboBoxModel<>(new Vector<>(cudaDeviceMap.keySet())));
            String gpuDevice = coreConfigs.getGpuDevice();
            if (gpuDevice == null || gpuDevice.isEmpty()) {
                comboBoxCudaDevice.setSelectedIndex(0);
            } else {
                HashMap<String, String> map = new HashMap<>();
                cudaDeviceMap.forEach((k, v) -> map.put(v, k));
                comboBoxCudaDevice.setSelectedItem(map.get(gpuDevice));
            }
        }
    }

    /**
     * 初始化所有选择栏的显示
     */
    private void setCheckBoxGui() {
        var configs = allConfigs.getConfigEntity();
        CoreConfigEntity coreConfig = allConfigs.getCoreConfigs();
        checkBoxLoseFocus.setSelected(configs.isLoseFocusClose());
        int startup = hasStartup();
        if (startup == 1) {
            eventManagement.putEvent(new ShowTaskBarMessageEvent(translateService.getTranslation("Warning"), translateService.getTranslation("The startup path is invalid")));
        }
        checkBoxAddToStartup.setSelected(startup == 0);
        checkBoxAdmin.setSelected(configs.isDefaultAdmin());
        checkBoxIsShowTipOnCreatingLnk.setSelected(configs.isShowTipCreatingLnk());
        checkBoxResponseCtrl.setSelected(configs.isDoubleClickCtrlOpen());
        checkBoxCheckUpdate.setSelected(configs.isCheckUpdateStartup());
        checkBoxIsAttachExplorer.setSelected(configs.isAttachExplorer());
        checkBoxEnableCuda.setSelected(coreConfig.isEnableGpuAccelerate());
        checkBoxEnableCuda.setEnabled(!DatabaseNativeService.getGpuDevices().isEmpty());
    }

    /**
     * 初始化所有列表的显示
     */
    private void setListGui() {
        listCmds.setListData(allConfigs.getCmdSet().toArray());
        listLanguage.setListData(translateService.getLanguageArray());
        listLanguage.setSelectedValue(translateService.getLanguage(), true);
        Object[] plugins = PluginService.getInstance().getPluginNameArray();
        listPlugins.setListData(plugins);
        listCache.setListData(cacheSet.toArray());
        ArrayList<String> list = new ArrayList<>();
        for (Constants.Enums.SwingThemes each : Constants.Enums.SwingThemes.values()) {
            list.add(each.toString());
        }
        listSwingThemes.setListData(list.toArray());
        listSwingThemes.setSelectedValue(allConfigs.getConfigEntity().getSwingTheme(), true);
        listDisks.setListData(diskSet.toArray());
    }

    /**
     * 初始化后缀名map
     */
    private void initSuffixMap() {
        suffixMap = DatabaseNativeService.getPriorityMap()
                .entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, entry -> (Integer) entry.getValue()));
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
            realTitle = translateService.getTranslation(each.title);
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
        Dimension tabbedPaneSize = new Dimension(Integer.parseInt(translateService.getFrameWidth()) - width, -1);
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
        setComboBoxGui();
        setTableGui();
        initTreeSettings();
        resizeGUI();

        tabbedPane.removeAll();
        tabbedPane.setBackground(new Color(0, 0, 0, 0));

        buttonUpdatePlugin.setVisible(false);
        buttonPluginSettings.setVisible(false);
        var configs = allConfigs.getConfigEntity();
        if (configs.getProxyType() == Constants.Enums.ProxyType.PROXY_DIRECT) {
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
        chooseUpdateAddress.setSelectedItem(configs.getUpdateAddress());
        comboBoxBorderType.setSelectedItem(allConfigs.getBorderType());
    }

    /**
     * 初始化cache
     */
    private void initCacheArray() {
        cacheSet = DatabaseNativeService.getCache();
    }

    /**
     * 初始化代理选择框
     */
    private void selectProxyType() {
        if (allConfigs.getConfigEntity().getProxyType() == Constants.Enums.ProxyType.PROXY_SOCKS) {
            radioButtonProxyTypeSocks5.setSelected(true);
        } else {
            radioButtonProxyTypeHttp.setSelected(true);
        }
    }

    private void initDiskSet() {
        String[] disks = RegexUtil.comma.split(allConfigs.getCoreConfigs().getDisks());
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
    }

    private void initTreeSettings() {
        //todo 添加新tab后在这里注册
        DefaultMutableTreeNode groupGeneral = new DefaultMutableTreeNode(translateService.getTranslation("General"));
        DefaultMutableTreeNode groupGuiSettings = new DefaultMutableTreeNode(translateService.getTranslation("GUI related settings"));
        groupGuiSettings.add(new DefaultMutableTreeNode(translateService.getTranslation("Interface")));
        groupGuiSettings.add(new DefaultMutableTreeNode(translateService.getTranslation("Language")));

        DefaultMutableTreeNode groupSearchSettings = new DefaultMutableTreeNode(translateService.getTranslation("Search related settings"));
        groupSearchSettings.add(new DefaultMutableTreeNode(translateService.getTranslation("Search settings")));
        groupSearchSettings.add(new DefaultMutableTreeNode(translateService.getTranslation("Cache")));
        groupSearchSettings.add(new DefaultMutableTreeNode(translateService.getTranslation("Suffix priority")));
        groupSearchSettings.add(new DefaultMutableTreeNode(translateService.getTranslation("My commands")));
        groupSearchSettings.add(new DefaultMutableTreeNode(translateService.getTranslation("Index")));

        DefaultMutableTreeNode groupProxy = new DefaultMutableTreeNode(translateService.getTranslation("Proxy settings"));
        DefaultMutableTreeNode groupHotkey = new DefaultMutableTreeNode(translateService.getTranslation("Hotkey settings"));
        DefaultMutableTreeNode groupPlugin = new DefaultMutableTreeNode(translateService.getTranslation("Plugins"));
        DefaultMutableTreeNode groupAbout = new DefaultMutableTreeNode(translateService.getTranslation("About"));

        DefaultMutableTreeNode root = new DefaultMutableTreeNode();
        root.add(groupGeneral);
        root.add(groupGuiSettings);
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
//        frame.setUndecorated(true);
        initCacheArray();
        initDiskSet();
        initSuffixMap();
    }

    private void prepareFrame() {
        frame.getRootPane().setWindowDecorationStyle(JRootPane.FRAME);
        frame.setIconImage(frameIcon.getImage());

        initTabNameMap();

        panel.remove(paneSwingThemes);
        excludeComponent.add(paneSwingThemes);

        ButtonGroup proxyButtonGroup = new ButtonGroup();
        proxyButtonGroup.add(radioButtonNoProxy);
        proxyButtonGroup.add(radioButtonUseProxy);

        ButtonGroup proxyTypeButtonGroup = new ButtonGroup();
        proxyTypeButtonGroup.add(radioButtonProxyTypeHttp);
        proxyTypeButtonGroup.add(radioButtonProxyTypeSocks5);

        addUpdateAddressToComboBox();

        addBorderTypeToComboBox();

        translate();

        addListeners();
    }

    private void addChooseUpdateComboBoxListener() {
        chooseUpdateAddress.addItemListener(item -> {
            if (item.getStateChange() == ItemEvent.SELECTED) {
                saveChanges();
            }
        });
    }

    /**
     * 添加所有监听器
     */
    private void addListeners() {
        addWindowCloseListener();
        addCheckBoxListener();
        addButtonRemoveDesktopListener();
        addFileChooserButtonListener();
        addTextFieldListener();
        addPriorityFileChooserListener();
        addPriorityTextFieldListener();
        addTextFieldRunAsAdminListener();
        addTextFieldProxyListener();
        addTextFieldOpenLastFolderListener();
        addButtonCMDListener();
        addButtonDelCMDListener();
        addGitHubLabelListener();
        addCheckForUpdateButtonListener();
        addTextFieldCopyPathListener();
        addResetColorButtonListener();
        addColorChooserLabelListener();
        addListPluginMouseListener();
        addButtonOpenPluginSettingsListener();
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
        addChooseUpdateComboBoxListener();
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
        for (var eachEntry : tabComponentNameMap.entrySet()) {
            if (eachEntry.getKey().tabName.equals(componentName)) {
                return eachEntry.getValue();
            }
        }
        return tabGeneral;
    }

    private void translateLabels() {
        labelMaxCacheNum.setText(translateService.getTranslation("Set the maximum number of caches:"));
        labelUpdateInterval.setText(translateService.getTranslation("File update detection interval:"));
        labelSecond.setText(translateService.getTranslation("Seconds"));
        labeltipPriorityFolder.setText(translateService.getTranslation("Priority search folder location (double-click to clear):"));
        labelConstIgnorePathTip.setText(translateService.getTranslation("Separate different paths with commas, and ignore C:\\Windows by default"));
        labelSetIgnorePathTip.setText(translateService.getTranslation("Set ignore folder:"));
        labelTransparency.setText(translateService.getTranslation("Search bar opacity:"));
        labelOpenSearchBarHotKey.setText(translateService.getTranslation("Open search bar:"));
        labelRunAsAdminHotKey.setText(translateService.getTranslation("Run as administrator:"));
        labelOpenFolderHotKey.setText(translateService.getTranslation("Open the parent folder:"));
        labelCopyPathHotKey.setText(translateService.getTranslation("Copy path:"));
        labelCmdTip2.setText(translateService.getTranslation("You can add custom commands here. After adding, " + "you can enter \": + your set identifier\" in the search box to quickly open"));
        labelColorTip.setText(translateService.getTranslation("Please enter the hexadecimal value of RGB color"));
        labelSearchBarColor.setText(translateService.getTranslation("Search bar Color:"));
        labelLabelColor.setText(translateService.getTranslation("Chosen label color:"));
        labelFontColor.setText(translateService.getTranslation("Chosen label font Color:"));
        labelDefaultColor.setText(translateService.getTranslation("Default background Color:"));
        labelNotChosenFontColor.setText(translateService.getTranslation("Unchosen label font Color:"));
        labelGitHubTip.setText(translateService.getTranslation("This is an open source software,GitHub:"));
        labelGithubIssue.setText(translateService.getTranslation("If you find a bug or have some suggestions, welcome to GitHub for feedback"));
        labelDescription.setText(translateService.getTranslation("Thanks for the following project"));
        labelTranslationTip.setText(translateService.getTranslation("The translation might not be 100% accurate"));
        labelLanguageChooseTip.setText(translateService.getTranslation("Choose a language"));
        labelVersion.setText(translateService.getTranslation("Current Version:") + Constants.version);
        labelBuildVersion.setText(Constants.buildVersion);
        labelInstalledPluginNum.setText(translateService.getTranslation("Installed plugins num:"));
        labelVacuumTip.setText(translateService.getTranslation("Click to organize the database and reduce the size of the database"));
        labelVacuumTip2.setText(translateService.getTranslation("but it will consume a lot of time."));
        labelAddress.setText(translateService.getTranslation("Address"));
        labelPort.setText(translateService.getTranslation("Port"));
        labelUserName.setText(translateService.getTranslation("User name"));
        labelPassword.setText(translateService.getTranslation("Password"));
        labelProxyTip.setText(translateService.getTranslation("If you need a proxy to access the Internet, You can add a proxy here."));
        labelCacheSettings.setText(translateService.getTranslation("Cache Settings"));
        labelCacheTip.setText(translateService.getTranslation("You can edit the saved caches here"));
        labelCacheTip2.setText(translateService.getTranslation("The cache is automatically generated " + "by the software and will be displayed first when searching."));
        labelSearchBarFontColor.setText(translateService.getTranslation("SearchBar Font Color:"));
        labelBorderColor.setText(translateService.getTranslation("Border Color:"));
        int cacheNum = cacheSet != null ? cacheSet.size() : 0;
        labelCurrentCacheNum.setText(translateService.getTranslation("Current Caches Num:") + cacheNum);
        labelUninstallPluginTip.setText(translateService.getTranslation("If you need to delete a plug-in, just delete it under the \"plugins\" folder in the software directory."));
        labelUninstallPluginTip2.setText(translateService.getTranslation("Tip:"));
        chooseUpdateAddressLabel.setText(translateService.getTranslation("Choose update address"));
        labelSearchCommand.setText(translateService.getTranslation("Search"));
        labelSuffixTip.setText(translateService.getTranslation("Modifying the suffix priority requires rebuilding the index (input \":update\") to take effect"));
        labelIndexTip.setText(translateService.getTranslation("You can rebuild the disk index here"));
        labelIndexChooseDisk.setText(translateService.getTranslation("The disks listed below will be indexed"));
        labelTipNTFSTip.setText(translateService.getTranslation("Only supports NTFS format disks"));
        labelLocalDiskTip.setText(translateService.getTranslation("Only the disks on this machine are listed below, click \"Add\" button to add a removable disk"));
        labelBorderThickness.setText(translateService.getTranslation("Border Thickness"));
        labelBorderType.setText(translateService.getTranslation("Border Type"));
        labelRoundRadius.setText(translateService.getTranslation("Round rectangle radius"));
        labelSearchThread.setText(translateService.getTranslation("Number of search threads"));
    }

    private void translateCheckbox() {
        checkBoxAddToStartup.setText(translateService.getTranslation("Add to startup"));
        checkBoxLoseFocus.setText(translateService.getTranslation("Close search bar when focus lost"));
        checkBoxAdmin.setText(translateService.getTranslation("Open other programs as an administrator " + "(provided that the software has privileges)"));
        checkBoxIsShowTipOnCreatingLnk.setText(translateService.getTranslation("Show tip on creating shortcut"));
        checkBoxResponseCtrl.setText(translateService.getTranslation("Double-click \"Ctrl\" to open the search bar"));
        checkBoxCheckUpdate.setText(translateService.getTranslation("Check for update at startup"));
        checkBoxIsAttachExplorer.setText(translateService.getTranslation("Attach to explorer"));
        checkBoxEnableCuda.setText(translateService.getTranslation("Enable GPU acceleration"));
    }

    private void translateButtons() {
        buttonSaveAndRemoveDesktop.setText(translateService.getTranslation("Clear desktop"));
        buttonSaveAndRemoveDesktop.setToolTipText(translateService.getTranslation("Backup and remove all desktop files"));
        ButtonPriorityFolder.setText(translateService.getTranslation("Choose"));
        buttonChooseFile.setText(translateService.getTranslation("Choose"));
        buttonAddCMD.setText(translateService.getTranslation("Add"));
        buttonDelCmd.setText(translateService.getTranslation("Delete"));
        buttonResetColor.setText(translateService.getTranslation("Reset to default"));
        buttonCheckUpdate.setText(translateService.getTranslation("Check for update"));
        buttonUpdatePlugin.setText(translateService.getTranslation("Check for update"));
        buttonPluginMarket.setText(translateService.getTranslation("Plugin Market"));
        buttonDeleteCache.setText(translateService.getTranslation("Delete cache"));
        buttonDeleteAllCache.setText(translateService.getTranslation("Delete all"));
        buttonChangeTheme.setText(translateService.getTranslation("change theme"));
        buttonVacuum.setText(translateService.getTranslation("Optimize database"));
        buttonPreviewColor.setText(translateService.getTranslation("Preview"));
        buttonClosePreview.setText(translateService.getTranslation("Close preview window"));
        buttonAddSuffix.setText(translateService.getTranslation("Add"));
        buttonDeleteSuffix.setText(translateService.getTranslation("Delete"));
        buttonDeleteAllSuffix.setText(translateService.getTranslation("Delete all"));
        buttonRebuildIndex.setText(translateService.getTranslation("Rebuild"));
        buttonAddNewDisk.setText(translateService.getTranslation("Add"));
        buttonDeleteDisk.setText(translateService.getTranslation("Delete"));
        buttonPluginSettings.setText(translateService.getTranslation("Settings"));
    }

    private void translateRadioButtons() {
        radioButtonNoProxy.setText(translateService.getTranslation("No proxy"));
        radioButtonUseProxy.setText(translateService.getTranslation("Configure proxy"));
    }

    private void translate() {
        initTreeSettings();
        translateLabels();
        translateCheckbox();
        translateButtons();
        translateRadioButtons();
        frame.setTitle(translateService.getTranslation("Settings"));
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
            stopPreview();
        });
    }

    @EventRegister(registerClass = ShowSettingsFrameEvent.class)
    private static void showSettingsFrameEvent(Event event) {
        ShowSettingsFrameEvent showSettingsFrameEvent = (ShowSettingsFrameEvent) event;
        SettingsFrame settingsFrame = getInstance();
        settingsFrame.isSuffixChanged = false;
        if (showSettingsFrameEvent.showTabName == null) {
            settingsFrame.showWindow();
        } else {
            settingsFrame.showWindow(showSettingsFrameEvent.showTabName);
        }
        settingsFrame.checkForUpdate(settingsFrame.downloadManager, false);
    }

    @EventListener(listenClass = AddToCacheEvent.class)
    private static void addCacheEvent(Event event) {
        AddToCacheEvent addToCacheEvent = (AddToCacheEvent) event;
        getInstance().addCache(addToCacheEvent.path);
    }

    @EventRegister(registerClass = GetExcludeComponentEvent.class)
    private static void getExcludeComponentEvent(Event event) {
        if (instance == null) {
            return;
        }
        SettingsFrame settingsFrame = getInstance();
        settingsFrame.excludeComponent.addAll(settingsFrame.tabComponentNameMap.values());
        event.setReturnValue(getInstance().excludeComponent);
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void restartEvent(Event event) {
        if (instance == null) {
            return;
        }
        SettingsFrame settingsFrame = getInstance();
        settingsFrame.hideFrame();
    }

    private void showWindow() {
        showWindow("tabGeneral");
    }

    private void showWindow(String tabName) {
        if (frame.isVisible()) {
            return;
        }
        if (!isFramePrepared) {
            isFramePrepared = true;
            prepareFrame();
        }
        var configs = allConfigs.getConfigEntity();
        tmp_openLastFolderKeyCode = configs.getOpenLastFolderKeyCode();
        tmp_runAsAdminKeyCode = configs.getRunAsAdminKeyCode();
        tmp_copyPathKeyCode = configs.getCopyPathKeyCode();
        frame.setResizable(true);
        double dpi = DpiUtil.getDpi();
        int width, height;
        try {
            width = Integer.parseInt(translateService.getFrameWidth());
            height = Integer.parseInt(translateService.getFrameHeight());
        } catch (NumberFormatException e) {
            log.error("error: {}", e.getMessage(), e);
            width = (int) (1000 / dpi);
            height = (int) (700 / dpi);
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
            stringBuilder.append(translateService.getTranslation("Round rectangle radius is wrong, please change"));
        }
    }

    private void checkBorderThickness(StringBuilder stringBuilder) {
        if (!canParseFloat(textFieldBorderThickness.getText(), 0, 4)) {
            stringBuilder.append(translateService.getTranslation("Border thickness is too large, please change"));
        }
    }

    private void checkUpdateTimeLimit(StringBuilder strBuilder) {
        if (!canParseInteger(textFieldUpdateInterval.getText(), 1, 3600)) {
            strBuilder.append(translateService.getTranslation("The file index update setting is wrong, please change")).append("\n");
        }
    }

    private void checkCacheNumLimit(StringBuilder strBuilder) {
        if (!canParseInteger(textFieldCacheNum.getText(), 1, 3600)) {
            strBuilder.append(translateService.getTranslation("The cache capacity is set incorrectly, please change")).append("\n");
        }
    }

    private void checkHotKey(StringBuilder strBuilder) {
        String tmp_hotkey = textFieldHotkey.getText();
        if (tmp_hotkey.length() < 5) {
            strBuilder.append(translateService.getTranslation("Hotkey setting is wrong, please change")).append("\n");
        } else {
            CheckHotKeyAvailableEvent checkHotKeyAvailableEvent = new CheckHotKeyAvailableEvent(tmp_hotkey);
            eventManagement.putEvent(checkHotKeyAvailableEvent);
            eventManagement.waitForEvent(checkHotKeyAvailableEvent);
            Optional<Boolean> isAvailable = checkHotKeyAvailableEvent.getReturnValue();
            isAvailable.ifPresent(ret -> {
                if (!ret) {
                    strBuilder.append(translateService.getTranslation("Hotkey setting is wrong, please change")).append("\n");
                }
            });
        }
        if (tmp_openLastFolderKeyCode == tmp_runAsAdminKeyCode || tmp_openLastFolderKeyCode == tmp_copyPathKeyCode || tmp_runAsAdminKeyCode == tmp_copyPathKeyCode) {
            strBuilder.append(translateService.getTranslation("HotKey conflict")).append("\n");
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
            strBuilder.append(translateService.getTranslation("Transparency setting error")).append("\n");
        }
    }

    private void checkLabelColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldLabelColor.getText())) {
            strBuilder.append(translateService.getTranslation("Chosen label color is set incorrectly")).append("\n");
        }
    }

    private void checkLabelFontColorWithCoverage(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldFontColorWithCoverage.getText())) {
            strBuilder.append(translateService.getTranslation("Chosen label font color is set incorrectly")).append("\n");
        }
    }

    private void checkDefaultBackgroundColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldBackgroundDefault.getText())) {
            strBuilder.append(translateService.getTranslation("Incorrect default background color setting")).append("\n");
        }
    }

    private void checkBorderColor(StringBuilder stringBuilder) {
        if (!canParseToRGB(textFieldBorderColor.getText())) {
            stringBuilder.append(translateService.getTranslation("Border color is set incorrectly")).append("\n");
        }
    }

    private void checkLabelFontColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldFontColor.getText())) {
            strBuilder.append(translateService.getTranslation("Unchosen label font color is set incorrectly")).append("\n");
        }
    }

    private void checkSearchBarColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldSearchBarColor.getText())) {
            strBuilder.append(translateService.getTranslation("The color of the search bar is set incorrectly")).append("\n");
        }
    }

    private void checkSearchBarFontColor(StringBuilder strBuilder) {
        if (!canParseToRGB(textFieldSearchBarFontColor.getText())) {
            strBuilder.append(translateService.getTranslation("The font color of the search bar is set incorrectly")).append("\n");
        }
    }

    private void checkProxy(StringBuilder strBuilder) {
        if (!canParseInteger(textFieldPort.getText(), 0, 65535)) {
            strBuilder.append(translateService.getTranslation("Proxy port is set incorrectly.")).append("\n");
        }
    }

    /**
     * 生成configuration
     *
     * @return ConfigEntity
     */
    private ConfigEntity getConfigEntity() {
        ConfigEntity configEntity = new ConfigEntity();
        CoreConfigEntity coreConfigEntity = new CoreConfigEntity();
        String ignorePathTemp = RegexUtil.getPattern("\n", 0).matcher(textAreaIgnorePath.getText()).replaceAll("");
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
        configEntity.setBorderThickness(Float.parseFloat(textFieldBorderThickness.getText()));
        if (borderType == null) {
            borderType = Constants.Enums.BorderType.AROUND;
        }
        configEntity.setBorderType(borderType.toString());
        coreConfigEntity.setPriorityFolder(textFieldPriorityFolder.getText());
        configEntity.setHotkey(textFieldHotkey.getText());
        coreConfigEntity.setCacheNumLimit(Integer.parseInt(textFieldCacheNum.getText()));
        coreConfigEntity.setUpdateTimeLimit(Integer.parseInt(textFieldUpdateInterval.getText()));
        coreConfigEntity.setIgnorePath(ignorePathTemp);
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
        configEntity.setLanguage(translateService.getLanguage());
        configEntity.setDoubleClickCtrlOpen(checkBoxResponseCtrl.isSelected());
        configEntity.setCheckUpdateStartup(checkBoxCheckUpdate.isSelected());
        coreConfigEntity.setDisks(parseDisk());
        configEntity.setAttachExplorer(checkBoxIsAttachExplorer.isSelected());
        coreConfigEntity.setEnableGpuAccelerate(checkBoxEnableCuda.isSelected());
        String selectedCudaDevice = (String) comboBoxCudaDevice.getSelectedItem();
        coreConfigEntity.setGpuDevice(cudaDeviceMap.getOrDefault(selectedCudaDevice, ""));
        var threadNum = (Integer) comboBoxSearchThread.getSelectedItem();
        coreConfigEntity.setSearchThreadNumber(threadNum == null ? Runtime.getRuntime().availableProcessors() * 2 : threadNum);
        configEntity.setCoreConfigEntity(coreConfigEntity);
        return configEntity;
    }

    /**
     * 保存所有设置
     *
     * @return 出现的错误信息，若成功则为空
     */
    private String saveChanges() {
        StringBuilder errorsStrBuilder = new StringBuilder();

        checkProxy(errorsStrBuilder);
        checkSearchBarColor(errorsStrBuilder);
        checkSearchBarFontColor(errorsStrBuilder);
        checkLabelColor(errorsStrBuilder);
        checkLabelFontColor(errorsStrBuilder);
        checkLabelFontColorWithCoverage(errorsStrBuilder);
        checkBorderColor(errorsStrBuilder);
        checkDefaultBackgroundColor(errorsStrBuilder);
        checkTransparency(errorsStrBuilder);
        checkHotKey(errorsStrBuilder);
        checkCacheNumLimit(errorsStrBuilder);
        checkUpdateTimeLimit(errorsStrBuilder);
        checkBorderThickness(errorsStrBuilder);
        checkRoundRadius(errorsStrBuilder);

        String errors = errorsStrBuilder.toString();
        if (!errors.isEmpty()) {
            return errors;
        }

        //重新显示翻译GUI，采用等号比对内存地址，不是用equals
        if (!(listLanguage.getSelectedValue() == translateService.getLanguage())) {
            translateService.setLanguage((String) listLanguage.getSelectedValue());
            translate();
        }

        //所有配置均正确
        //使所有配置生效
        ConfigEntity configEntity = getConfigEntity();

        eventManagement.putEvent(new SetConfigsEvent(configEntity));

        var configs = allConfigs.getConfigEntity();
        Color tmp_color = new Color(configs.getLabelColor());
        labelColorChooser.setBackground(tmp_color);
        labelColorChooser.setForeground(tmp_color);
        tmp_color = new Color(configs.getDefaultBackgroundColor());
        defaultBackgroundChooser.setBackground(tmp_color);
        defaultBackgroundChooser.setForeground(tmp_color);
        tmp_color = new Color(configs.getFontColorWithCoverage());
        FontColorWithCoverageChooser.setBackground(tmp_color);
        FontColorWithCoverageChooser.setForeground(tmp_color);
        tmp_color = new Color(configs.getFontColor());
        FontColorChooser.setBackground(tmp_color);
        FontColorChooser.setForeground(tmp_color);
        tmp_color = new Color(configs.getSearchBarFontColor());
        SearchBarFontColorChooser.setBackground(tmp_color);
        SearchBarFontColorChooser.setForeground(tmp_color);
        tmp_color = new Color(configs.getBorderColor());
        borderColorChooser.setBackground(tmp_color);
        borderColorChooser.setForeground(tmp_color);
        return "";
    }

    private void setStartup(boolean b) {
        if (b) {
            try {
                StartupUtil.deleteStartup();
                Process p = StartupUtil.addStartupByXml();
                BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                String line;
                StringBuilder result = new StringBuilder();
                while ((line = outPut.readLine()) != null) {
                    result.append(line);
                }
                outPut.close();
                if (!result.toString().isEmpty()) {
                    checkBoxAddToStartup.setSelected(false);
                    JOptionPane.showMessageDialog(frame, translateService.getTranslation("Add to startup failed, please try to run as administrator") + "\n" + result);
                }
            } catch (Exception e) {
                log.error(e.getMessage(), e);
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
                        JOptionPane.showMessageDialog(frame, translateService.getTranslation("Delete startup failed, please try to run as administrator"));
                    }
                } catch (Exception e) {
                    log.error(e.getMessage(), e);
                }
            }
        }
    }

    private record TabNameAndTitle(String tabName, String title) {
    }
}
