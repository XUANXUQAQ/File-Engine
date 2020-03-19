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
    public static int backgroundColor;
    public static int backgroundColorLight;
    public static int fontColorWithCoverage;
    public static int fontColor;
    private JTextField textFieldUpdateTime;
    private JTextField textFieldCacheNum;
    private JTextArea textAreaIgnorePath;
    private JTextField textFieldSearchDepth;
    private JCheckBox checkBox1;
    private JFrame frame = new JFrame("����");
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
    private JLabel labelCopyPath;
    private JTextField textFieldCopyPath;
    private JLabel labelVersion;
    private JPanel tab1;
    private JPanel tab2;
    private JPanel tab3;
    private JPanel tab4;
    private JPanel tab5;
    private JPanel tab6;
    private JTextField textFieldTransparency;
    private JLabel labelTransparency;
    private JLabel labelPlaceHolder5;
    private JPanel tab7;
    private JLabel labelColorTip;
    private JTextField textFieldBackgroundColorLight;
    private JTextField textFieldBackground;
    private JTextField textFieldFontColorWithCoverage;
    private JTextField textFieldLabelColor;
    private JLabel labelLabelColor;
    private JLabel labelFontColor;
    private JLabel labelBackgroundColor;
    private JLabel labelLightColor;
    private JLabel labelSharp1;
    private JLabel labelSharp2;
    private JLabel labelSharp3;
    private JLabel labelSharp4;
    private JButton buttonResetColor;
    private JTextField textFieldFontColor;
    private JLabel labelSharp5;
    private JLabel labelFontColors;
    private static boolean isStartup;
    private Unzip unzipInstance = Unzip.getInstance();
    private Thread updateThread = null;


    public SettingsFrame() {
        Color transColor = new Color(0, 0, 0, 0);
        UIManager.put("TabbedPane.focus", transColor);
        labelAboutGithub.setText("<html><a href='https://github.com/XUANXUQAQ/File-Engine'><font size=\"4\">File-Engine</font></a></html>");
        ImageIcon imageIcon = new ImageIcon(SettingsFrame.class.getResource("/icons/frame.png"));
        labelIcon.setIcon(imageIcon);
        checkBox1.addActionListener(e -> setStartup(checkBox1.isSelected()));
        buttonSaveAndRemoveDesktop.addActionListener(e -> {
            String currentFolder = new File("").getAbsolutePath();
            if (currentFolder.equals(FileSystemView.getFileSystemView().getHomeDirectory().getAbsolutePath()) || currentFolder.equals("C:\\Users\\Public\\Desktop")) {
                JOptionPane.showMessageDialog(null, "��⵽�ó��������棬�޷��ƶ�");
                return;
            }
            int isConfirmed = JOptionPane.showConfirmDialog(null, "�Ƿ��Ƴ������������ϵ������ļ�\n���ǻ��ڸó����Files�ļ�����\n�������Ҫ������ʱ��");
            if (isConfirmed == 0) {
                Thread fileMover = new Thread(new moveDesktopFiles());
                fileMover.start();
            }
        });
        Button3.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), "ѡ��");
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
                JOptionPane.showMessageDialog(null, "�뽫�������ڵ����һ����Ϊ��ǰ�ļ��У�\n�޸ĳɹ������������ÿ���������\n�´�����ʱ��ʹ�ô�(elevated)�Ŀ�ݷ�ʽ��");
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
                JOptionPane.showMessageDialog(null, "���Թ���Ա�������");
            }
        });
        ButtonRecoverAdmin.addActionListener(e -> {
            if (MainClass.isAdmin()) {
                JOptionPane.showMessageDialog(null, "������һ��Remove shortcut����ѡ��ɾ����ݷ�ʽ��\n�޸ĳɹ������������ÿ���������");
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
                JOptionPane.showMessageDialog(null, "���Թ���Ա�������");
            }
        });
        ButtonDataPath.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), "ѡ��");
            File file = fileChooser.getSelectedFile();
            if (file != null && returnValue == JFileChooser.APPROVE_OPTION) {
                textFieldDataPath.setText(file.getAbsolutePath());
            }

        });
        ButtonPriorityFolder.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            int returnValue = fileChooser.showDialog(new JLabel(), "ѡ��");
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
        labelVersion.setText("��ǰ�汾��" + MainClass.version);
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
            textFieldBackground.setText(Integer.toHexString(backgroundColor));
            textFieldBackgroundColorLight.setText(Integer.toHexString(backgroundColorLight));
            textFieldLabelColor.setText(Integer.toHexString(labelColor));
            textFieldFontColorWithCoverage.setText(Integer.toHexString(fontColorWithCoverage));
            checkBoxLoseFocus.setSelected(isLoseFocusClose);
            runAsAdminKeyCode = settings.getInteger("runAsAdminKeyCode");
            openLastFolderKeyCode = settings.getInteger("openLastFolderKeyCode");
            transparency = settings.getFloat("transparency");
            textFieldTransparency.setText(String.valueOf(transparency));
            textFieldFontColor.setText(Integer.toHexString(fontColor));
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
        buttonHelp.addActionListener(e -> JOptionPane.showMessageDialog(null, "������\n" +
                "1.Ĭ��Ctrl + Alt + J��������\n" +
                "2.Enter�����г���\n" +
                "3.Ctrl + Enter���򿪲�ѡ���ļ������ļ���\n" +
                "4.Shift + Enter���Թ���ԱȨ�����г���ǰ���Ǹó���ӵ�й���ԱȨ�ޣ�\n" +
                "5.��������������  : update  ǿ���ؽ���������\n" +
                "6.��������������  : version  �鿴��ǰ�汾\n" +
                "7.��������������  : clearbin  ��ջ���վ\n" +
                "8.��������������  : help  �鿴����\n" +
                "9.�������п����Զ��������������������  : �Զ����ʶ  �����Լ�������\n" +
                "10.��������ļ���������  : full  ��ȫ��ƥ��\n" +
                "11.��������ļ���������  : F  ��ֻƥ���ļ�\n" +
                "12.��������ļ���������  : D  ��ֻƥ���ļ���\n" +
                "13.��������ļ���������  : Ffull  ��ֻƥ���ļ���ȫ��ƥ��\n" +
                "14.��������ļ���������  : Dfull  ��ֻƥ���ļ��в�ȫ��ƥ��"));
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
            String name = JOptionPane.showInputDialog("������Ը�����ı�ʶ,֮��������������������� \": ��ʶ��\" ֱ��ִ�и�����");
            if (name == null || name.isEmpty()) {
                //δ����
                return;
            }
            if (name.equals("update") || name.equals("clearbin") || name.equals("help") || name.equals("version") || isRepeatCommand(name)) {
                JOptionPane.showMessageDialog(null, "�����е������ͻ");
                return;
            }
            String cmd;
            JOptionPane.showMessageDialog(null, "��ѡ���ִ���ļ�λ��(�ļ���Ҳ����)");
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
            int returnValue = fileChooser.showDialog(new Label(), "ѡ��");
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
            textFieldLabelColor.setText(Integer.toHexString(0xFF9868));
            textFieldBackgroundColorLight.setText(Integer.toHexString(0x4B4B4B));
            textFieldBackground.setText(Integer.toHexString(0x6C6C6C));
            textFieldFontColor.setText(Integer.toHexString(0xC5C5C5));
        });
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
        //��ֹ���γ���ץȡ������403����
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
        //��ȡ����������Ϣ
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
                    MainClass.showMessage("��ʾ", "��⵽���治���ڣ�������������");
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
            if (settings.containsKey("backgroundColor")) {
                backgroundColor = settings.getInteger("backgroundColor");
            } else {
                backgroundColor = 0x6C6C6C;
            }
            if (settings.containsKey("backgroundColorLight")) {
                backgroundColorLight = settings.getInteger("backgroundColorLight");
            } else {
                backgroundColorLight = 0x4B4B4B;
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
        allSettings.put("backgroundColor", backgroundColor);
        allSettings.put("backgroundColorLight", backgroundColorLight);
        allSettings.put("fontColorWithCoverage", fontColorWithCoverage);
        allSettings.put("fontColor", fontColor);
        try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
            buffW.write(allSettings.toJSONString());
        } catch (IOException ignored) {

        }

        //��ȡ�����Զ�������
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
            JOptionPane.showMessageDialog(null, "������ʧ��");
            return;
        }
        File test = new File("TEST");
        if (Double.parseDouble(latestVersion) > Double.parseDouble(MainClass.version) || test.exists()) {
            int result = JOptionPane.showConfirmDialog(null, "���°汾" + latestVersion + "���Ƿ����");
            if (result == 0) {
                //��ʼ����,���ظ����ļ���tmp
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
                    if (!e.getMessage().equals("�û��ж�����")) {
                        JOptionPane.showMessageDialog(null, "����ʧ��");
                    }
                    download.hideFrame();
                    return;
                }
                MainClass.showMessage("��ʾ", "������ɣ����½����´�����ʱ��ʼ");
                try {
                    File updateSignal = new File("user/update");
                    updateSignal.createNewFile();
                } catch (Exception ignored) {

                }
            }
        } else {
            JOptionPane.showMessageDialog(null, "���°汾��" + latestVersion + "\n��ǰ�汾��������");
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
        frame.setSize(800, 600);
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
            updateTimeLimitTemp = -1; // ���벻��ȷ
        }
        if (updateTimeLimitTemp > 3600 || updateTimeLimitTemp <= 0) {
            JOptionPane.showMessageDialog(null, "�ļ������������ô��������");
            return;
        }
        isStartup = checkBox1.isSelected();
        try {
            cacheNumLimitTemp = Integer.parseInt(textFieldCacheNum.getText());
        } catch (Exception e1) {
            cacheNumLimitTemp = -1;
        }
        if (cacheNumLimitTemp > 10000 || cacheNumLimitTemp <= 0) {
            JOptionPane.showMessageDialog(null, "�����������ô��������");
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
            JOptionPane.showMessageDialog(null, "����������ô��������");
            return;
        }

        String _hotkey = textFieldHotkey.getText();
        if (_hotkey.length() == 1) {
            JOptionPane.showMessageDialog(null, "��ݼ����ô���");
            return;
        } else {
            if (!(64 < _hotkey.charAt(_hotkey.length() - 1) && _hotkey.charAt(_hotkey.length() - 1) < 91)) {
                JOptionPane.showMessageDialog(null, "��ݼ����ô���");
                return;
            }
        }
        try {
            transparencyTemp = Float.parseFloat(textFieldTransparency.getText());
        } catch (Exception e) {
            transparencyTemp = -1f;
        }
        if (transparencyTemp > 1 || transparencyTemp <= 0) {
            JOptionPane.showMessageDialog(null, "͸�������ô���");
            return;
        }

        if (_openLastFolderKeyCode == _runAsAdminKeyCode || _openLastFolderKeyCode == _copyPathKeyCode || _runAsAdminKeyCode == _copyPathKeyCode) {
            JOptionPane.showMessageDialog(null, "��ݼ���ͻ");
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
            JOptionPane.showMessageDialog(null, "ѡ�п���ɫ���ô���");
            return;
        }
        int _fontColorWithCoverage;
        try {
            _fontColorWithCoverage = Integer.parseInt(textFieldFontColorWithCoverage.getText(), 16);
        } catch (Exception e) {
            _fontColorWithCoverage = -1;
        }
        if (_fontColorWithCoverage < 0) {
            JOptionPane.showMessageDialog(null, "ѡ�п�������ɫ���ô���");
            return;
        }
        int _backgroundColor;
        try {
            _backgroundColor = Integer.parseInt(textFieldBackground.getText(), 16);
        } catch (Exception e) {
            _backgroundColor = -1;
        }
        if (_backgroundColor < 0) {
            JOptionPane.showMessageDialog(null, "2��4��Ĭ����ɫ���ô���");
            return;
        }
        int _backgroundColorLight;
        try {
            _backgroundColorLight = Integer.parseInt(textFieldBackgroundColorLight.getText(), 16);
        } catch (Exception e) {
            _backgroundColorLight = -1;
        }
        if (_backgroundColorLight < 0) {
            JOptionPane.showMessageDialog(null, "1��3��Ĭ����ɫ���ô���");
            return;
        }
        int _fontColor;
        try {
            _fontColor = Integer.parseInt(textFieldFontColor.getText(), 16);
        } catch (Exception e) {
            _fontColor = -1;
        }
        if (_fontColor < 0) {
            JOptionPane.showMessageDialog(null, "δѡ�п���������ɫ���ô���");
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
        backgroundColorLight = _backgroundColorLight;
        backgroundColor = _backgroundColor;
        fontColorWithCoverage = _fontColorWithCoverage;
        fontColor = _fontColor;
        SearchBar instance = SearchBar.getInstance();
        instance.setBackgroundColor(backgroundColor);
        instance.setBackgroundColorLight(backgroundColorLight);
        instance.setLabelColor(labelColor);
        instance.setFontColorWithCoverage(fontColorWithCoverage);
        instance.setFontColor(fontColor);

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
        allSettings.put("backgroundColor", backgroundColor);
        allSettings.put("backgroundColorLight", backgroundColorLight);
        allSettings.put("fontColorWithCoverage", fontColorWithCoverage);
        allSettings.put("fontColor", fontColor);
        try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
            buffW.write(allSettings.toJSONString());
            JOptionPane.showMessageDialog(null, "����ɹ�");
        } catch (IOException ignored) {

        }
        //�����Զ�������
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
                    JOptionPane.showMessageDialog(null, "��ӵ���������ʧ�ܣ��볢���Թ���Ա�������");
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
                    JOptionPane.showMessageDialog(null, "ɾ����������ʧ�ܣ��볢���Թ���Ա�������");
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
            // ����������
            byte[] buffer = new byte[8192];
            int count = bis.read(buffer);
            while (count != -1) {
                //ʹ�û�����д����
                bos.write(buffer, 0, count);
                //ˢ��
                bos.flush();
                count = bis.read(buffer);
            }
        } catch (IOException ignored) {

        }
    }

    static class moveDesktopFiles implements Runnable {

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
        }
    }
}