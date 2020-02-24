package frame;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import download.DownloadUpdate;
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
import java.util.HashSet;
import java.util.Objects;


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
    public static File tmp = new File("tmp");
    public static File settings = new File("./user/settings.json");
    public static HashSet<String> cmdSet = new HashSet<>();
    private static int _openLastFolderKeyCode;
    private static int _runAsAdminKeyCode;
    private static CheckHotKey HotKeyListener;
    private JTextField textFieldUpdateTime;
    private JTextField textFieldCacheNum;
    private JTextArea textAreaIgnorePath;
    private JTextField textFieldSearchDepth;
    private JCheckBox checkBox1;
    private JLabel label3;
    private JLabel label1;
    private JLabel label2;
    private Search searchObj = Search.getInstance();
    private JFrame frame = new JFrame("设置");
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
    private JButton buttonHelp;
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
    private JLabel labelCmdTip;
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
    private boolean isStartup;
    private Unzip unzipInstance = Unzip.getInstance();
    private Thread updateThread = null;

    public SettingsFrame() {
        labelAboutGithub.setText("<html><a href='https://github.com/XUANXUQAQ/File-Engine'>File-Engine</a></html>");
        ImageIcon imageIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        labelIcon.setIcon(imageIcon);
        checkBox1.addActionListener(e -> {
            setStartup(checkBox1.isSelected());

        });
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
                    if (!txt.equals("")) {
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
            String MaxUpdateTime = "";
            MaxUpdateTime = MaxUpdateTime + updateTimeLimit;
            textFieldUpdateTime.setText(MaxUpdateTime);
            ignorePath = settings.getString("ignorePath");
            textAreaIgnorePath.setText(ignorePath.replaceAll(",", ",\n"));
            cacheNumLimit = settings.getInteger("cacheNumLimit");
            String MaxCacheNum = "";
            MaxCacheNum = MaxCacheNum + cacheNumLimit;
            textFieldCacheNum.setText(MaxCacheNum);
            String searchDepth = "";
            int searchDepthInSettings = settings.getInteger("searchDepth");
            searchDepth = searchDepth + searchDepthInSettings;
            textFieldSearchDepth.setText(searchDepth);
            textFieldHotkey.setText(hotkey);
            textFieldDataPath.setText(dataPath);
            textFieldPriorityFolder.setText(priorityFolder);
            isDefaultAdmin = settings.getBoolean("isDefaultAdmin");
            checkBoxAdmin.setSelected(isDefaultAdmin);
            isLoseFocusClose = settings.getBoolean("isLoseFocusClose");
            checkBoxLoseFocus.setSelected(isLoseFocusClose);
            runAsAdminKeyCode = settings.getInteger("runAsAdminKeyCode");
            openLastFolderKeyCode = settings.getInteger("openLastFolderKeyCode");
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
            listCmds.setListData(cmdSet.toArray());
        } catch (IOException ignored) {

        }
        buttonHelp.addActionListener(e -> JOptionPane.showMessageDialog(null, "帮助：\n" +
                "1.默认Ctrl + Alt + J打开搜索框\n" +
                "2.Enter键运行程序\n" +
                "3.Ctrl + Enter键打开并选中文件所在文件夹\n" +
                "4.Shift + Enter键以管理员权限运行程序（前提是该程序拥有管理员权限）\n" +
                "5.在搜索框中输入  : update  强制重建本地索引\n" +
                "6.在搜索框中输入  : version  查看当前版本\n" +
                "7.在搜索框中输入  : clearbin  清空回收站\n" +
                "8.在搜索框中输入  : help  查看帮助\n" +
                "9.在设置中可以自定义命令，在搜索框中输入  : 自定义标识  运行自己的命令\n" +
                "10.在输入的文件名后输入  : full  可全字匹配\n" +
                "11.在输入的文件名后输入  : file  可只匹配文件\n" +
                "12.在输入的文件名后输入  : folder  可只匹配文件夹\n" +
                "13.在输入的文件名后输入  : filefull  可只匹配文件并全字匹配\n" +
                "14.在输入的文件名后输入  : folderfull  可只匹配文件夹并全字匹配"));
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
            if (name == null || name.equals("")) {
                //未输入
                return;
            }
            if (name.equals("update") || name.equals("clearbin") || name.equals("help") || name.equals("version")) {
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
    }

    public static JSONObject getInfo() throws IOException {
        StringBuilder jsonUpdate = new StringBuilder();
        URL updateServer = new URL("https://rawcdn.githack.com/XUANXUQAQ/File-Engine/eb36e75f54d4261b59462f4f6fe24e84e0fa0271/version.json");
        URLConnection uc = updateServer.openConnection();
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
        try (BufferedReader buffR = new BufferedReader(new FileReader(settings))) {
            String line;
            StringBuilder result = new StringBuilder();
            while (null != (line = buffR.readLine())) {
                result.append(line);
            }
            JSONObject settings = JSON.parseObject(result.toString());
            cacheNumLimit = settings.getInteger("cacheNumLimit");
            hotkey = settings.getString("hotkey");
            HotKeyListener = CheckHotKey.getInstance();
            dataPath = settings.getString("dataPath");
            priorityFolder = settings.getString("priorityFolder");
            searchDepth = settings.getInteger("searchDepth");
            ignorePath = settings.getString("ignorePath");
            updateTimeLimit = settings.getInteger("updateTimeLimit");
            isDefaultAdmin = settings.getBoolean("isDefaultAdmin");
            isLoseFocusClose = settings.getBoolean("isLoseFocusClose");
            openLastFolderKeyCode = settings.getInteger("openLastFolderKeyCode");
            _openLastFolderKeyCode = openLastFolderKeyCode;
            runAsAdminKeyCode = settings.getInteger("runAsAdminKeyCode");
            _runAsAdminKeyCode = runAsAdminKeyCode;
            File data = new File(dataPath);
            if (!data.exists()) {
                MainClass.showMessage("提示", "检测到缓存不存在，正在重新搜索");
                data = new File("data");
                dataPath = data.getAbsolutePath();
                new SettingsFrame().saveChangesWithoutPrompt();
                Search.getInstance().setManualUpdate(true);
            }
        } catch (Exception ignored) {

        }
        //获取所有自定义命令
        try (BufferedReader br = new BufferedReader(new FileReader(new File("./user/cmds.txt")))) {
            String each;
            while ((each = br.readLine()) != null) {
                cmdSet.add(each);
            }
        } catch (IOException ignored) {

        }
    }

    private void update() {
        JSONObject updateInfo;
        String updateVersion;
        try {
            updateInfo = getInfo();
            updateVersion = updateInfo.getString("version");
        } catch (IOException e1) {
            JOptionPane.showMessageDialog(null, "检查更新失败");
            return;
        }

        if (!updateVersion.equals(MainClass.version)) {
            int result = JOptionPane.showConfirmDialog(null, "有新版本" + updateVersion + "，是否更新");
            if (result == 0) {
                //开始更新,下载更新文件到tmp
                String urlChoose;
                String fileName;
                if (MainClass.name.contains("x64")) {
                    urlChoose = "url64";
                    fileName = "File-Engine-x64.exe";
                } else {
                    urlChoose = "url32";
                    fileName = "File-Engine-x32.exe";
                }
                try {
                    MainClass.showMessage("提示", "已开始下载");
                    DownloadUpdate download = DownloadUpdate.getInstance();
                    download.downLoadFromUrl(updateInfo.getString(urlChoose), fileName, tmp.getAbsolutePath());
                } catch (Exception e) {
                    if (!e.equals(new Exception("用户中断下载"))) {
                        JOptionPane.showMessageDialog(null, "下载失败");
                    }
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
            JOptionPane.showMessageDialog(null, "当前版本已是最新");
        }
    }

    public void hideFrame() {
        frame.setVisible(false);
    }

    private void saveChangesWithoutPrompt() {
        JSONObject allSettings = new JSONObject();
        String MaxUpdateTime = textFieldUpdateTime.getText();
        try {
            updateTimeLimit = Integer.parseInt(MaxUpdateTime);
        } catch (Exception e1) {
            updateTimeLimit = -1; // 输入不正确
        }
        if (updateTimeLimit > 3600 || updateTimeLimit <= 0) {
            JOptionPane.showMessageDialog(null, "文件索引更新设置错误，请更改");
            return;
        }
        isStartup = checkBox1.isSelected();
        String MaxCacheNum = textFieldCacheNum.getText();
        try {
            cacheNumLimit = Integer.parseInt(MaxCacheNum);
        } catch (Exception e1) {
            cacheNumLimit = -1;
        }
        if (cacheNumLimit > 10000 || cacheNumLimit <= 0) {
            JOptionPane.showMessageDialog(null, "缓存容量设置错误，请更改");
            return;
        }
        ignorePath = textAreaIgnorePath.getText();
        ignorePath = ignorePath.replaceAll("\n", "");
        if (!ignorePath.toLowerCase().contains("c:\\windows")) {
            ignorePath = ignorePath + "C:\\Windows,";
        }
        try {
            searchDepth = Integer.parseInt(textFieldSearchDepth.getText());
        } catch (Exception e1) {
            searchDepth = -1;
        }

        if (searchDepth > 10 || searchDepth <= 0) {
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
        if (_openLastFolderKeyCode == _runAsAdminKeyCode) {
            JOptionPane.showMessageDialog(null, "打开上级文件夹快捷键与以管理员方式运行快捷键冲突");
            return;
        } else {
            openLastFolderKeyCode = _openLastFolderKeyCode;
            runAsAdminKeyCode = _runAsAdminKeyCode;
        }
        priorityFolder = textFieldPriorityFolder.getText();
        dataPath = textFieldDataPath.getText();
        hotkey = textFieldHotkey.getText();
        HotKeyListener.registerHotkey(hotkey);
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
        try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
            buffW.write(allSettings.toJSONString());
        } catch (IOException ignored) {

        }
        //保存自定义命令
        StringBuilder strb = new StringBuilder();
        for (String each : cmdSet) {
            strb.append(each);
            strb.append("\n");
        }
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File("./user/cmds.txt")))) {
            bw.write(strb.toString());
        } catch (IOException ignored) {

        }
    }

    public boolean isSettingsVisible() {
        return frame.isVisible();
    }

    public void showWindow() {
        frame.setContentPane(new SettingsFrame().panel);
        URL frameIcon = SettingsFrame.class.getResource("/icons/frame.png");
        frame.setIconImage(new ImageIcon(frameIcon).getImage());
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取当前分辨率
        int width = screenSize.width;
        int height = screenSize.height;
        frame.setSize(width / 2, height / 2);
        frame.setLocation(width / 2 - width / 4, height / 2 - height / 4);
        panel.setOpaque(true);
        frame.setVisible(true);
    }

    private void saveChanges() {
        {
            JSONObject allSettings = new JSONObject();
            String MaxUpdateTime = textFieldUpdateTime.getText();
            try {
                updateTimeLimit = Integer.parseInt(MaxUpdateTime);
            } catch (Exception e1) {
                updateTimeLimit = -1; // 输入不正确
            }
            if (updateTimeLimit > 3600 || updateTimeLimit <= 0) {
                JOptionPane.showMessageDialog(null, "文件索引更新设置错误，请更改");
                return;
            }
            isStartup = checkBox1.isSelected();
            String MaxCacheNum = textFieldCacheNum.getText();
            try {
                cacheNumLimit = Integer.parseInt(MaxCacheNum);
            } catch (Exception e1) {
                cacheNumLimit = -1;
            }
            if (cacheNumLimit > 10000 || cacheNumLimit <= 0) {
                JOptionPane.showMessageDialog(null, "缓存容量设置错误，请更改");
                return;
            }
            ignorePath = textAreaIgnorePath.getText();
            ignorePath = ignorePath.replaceAll("\n", "");
            if (!ignorePath.toLowerCase().contains("c:\\windows")) {
                ignorePath = ignorePath + "C:\\Windows,";
            }
            try {
                searchDepth = Integer.parseInt(textFieldSearchDepth.getText());
            } catch (Exception e1) {
                searchDepth = -1;
            }

            if (searchDepth > 10 || searchDepth <= 0) {
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
            if (_openLastFolderKeyCode == _runAsAdminKeyCode) {
                JOptionPane.showMessageDialog(null, "打开上级文件夹快捷键与以管理员方式运行快捷键冲突");
                return;
            } else {
                openLastFolderKeyCode = _openLastFolderKeyCode;
                runAsAdminKeyCode = _runAsAdminKeyCode;
            }
            priorityFolder = textFieldPriorityFolder.getText();
            dataPath = textFieldDataPath.getText();
            hotkey = textFieldHotkey.getText();
            HotKeyListener.registerHotkey(hotkey);
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
            try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
                buffW.write(allSettings.toJSONString());
                JOptionPane.showMessageDialog(null, "保存成功"); //将设置保存
            } catch (IOException ignored) {

            }
            //保存自定义命令
            StringBuilder strb = new StringBuilder();
            for (String each : cmdSet) {
                strb.append(each);
                strb.append("\n");
            }
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File("./user/cmds.txt")))) {
                bw.write(strb.toString());
            } catch (IOException ignored) {

            }
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
                if (!result.toString().equals("")) {
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
                if (!result.toString().equals("")) {
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

    class moveDesktopFiles implements Runnable {

        @Override
        public void run() {
            File fileDesktop = FileSystemView.getFileSystemView().getHomeDirectory();
            File fileBackUp = new File("Files");
            if (!fileBackUp.exists()) {
                fileBackUp.mkdir();
            }
            moveFiles moveFiles = new moveFiles();
            moveFiles.moveFolder(fileDesktop.toString(), fileBackUp.getAbsolutePath());
            moveFiles.moveFolder("C:\\Users\\Public\\Desktop", fileBackUp.getAbsolutePath());
            searchObj.setUsable(false);
        }
    }
}