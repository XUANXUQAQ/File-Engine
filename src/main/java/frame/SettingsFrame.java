package frame;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import main.MainClass;
import moveFiles.moveFiles;
import search.Search;
import unzipFile.Unzip;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.URL;
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
    public static File settings = new File("settings.json");
    public static String exds;
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
    private JFrame frame = new JFrame("����");
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
    private JLabel labelSelfDefinedTip;
    private JButton buttonAddCMD;
    private JButton buttonDelCmd;
    private JScrollPane scrollPaneCmd;
    private JList<Object> listCmds;
    private JLabel labelExdTip;
    private JTextArea textAreaExds;
    private JLabel labelPlaceHoder4;
    private JButton buttonSave;
    private boolean isStartup;
    private Unzip unzipInstance = Unzip.getInstance();

    public SettingsFrame() {
        checkBox1.addActionListener(e -> {
            setStartup(checkBox1.isSelected());

        });
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
                File searchexe = new File(MainClass.name);
                try {
                    Runtime.getRuntime().exec("\"" + mainUAC.getAbsolutePath() + "\"" + " \"" + searchexe.getAbsolutePath() + "\"");
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
            textAreaExds.setText(exds);
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
                "11.��������ļ���������  : file  ��ֻƥ���ļ�\n" +
                "12.��������ļ���������  : folder  ��ֻƥ���ļ���\n" +
                "13.��������ļ���������  : filefull  ��ֻƥ���ļ���ȫ��ƥ��\n" +
                "14.��������ļ���������  : folderfull  ��ֻƥ���ļ��в�ȫ��ƥ��"));
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
            if (name == null || name.equals("")) {
                //δ����
                return;
            }
            if (name.equals("update") || name.equals("clearbin") || name.equals("help") || name.equals("version")) {
                JOptionPane.showMessageDialog(null, "�����е������ͻ");
                return;
            }
            String cmd;
            JOptionPane.showMessageDialog(null, "��ѡ���ִ���������ļ�λ��");
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
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
        buttonSave.addActionListener(e -> {
            saveChanges();
        });
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
            exds = settings.getString("exds");
        } catch (IOException ignored) {

        }
        //��ȡ�����Զ�������
        try (BufferedReader br = new BufferedReader(new FileReader(new File("cmds.txt")))) {
            String each;
            while ((each = br.readLine()) != null) {
                cmdSet.add(each);
            }
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
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // ��ȡ��ǰ�ֱ���
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
                updateTimeLimit = -1; // ���벻��ȷ
            }
            if (updateTimeLimit > 3600 || updateTimeLimit <= 0) {
                JOptionPane.showMessageDialog(null, "�ļ������������ô��������");
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
                JOptionPane.showMessageDialog(null, "�����������ô��������");
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
            if (_openLastFolderKeyCode == _runAsAdminKeyCode) {
                JOptionPane.showMessageDialog(null, "���ϼ��ļ��п�ݼ����Թ���Ա��ʽ���п�ݼ���ͻ");
                return;
            } else {
                openLastFolderKeyCode = _openLastFolderKeyCode;
                runAsAdminKeyCode = _runAsAdminKeyCode;
            }
            priorityFolder = textFieldPriorityFolder.getText();
            dataPath = textFieldDataPath.getText();
            hotkey = textFieldHotkey.getText();
            HotKeyListener.registHotkey(hotkey);
            exds = textAreaExds.getText();
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
            allSettings.put("exds", exds);
            try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
                buffW.write(allSettings.toJSONString());
                JOptionPane.showMessageDialog(null, "����ɹ�"); //�����ñ���
            } catch (IOException ignored) {

            }
            //�����Զ�������
            StringBuilder strb = new StringBuilder();
            for (String each : cmdSet) {
                strb.append(each);
                strb.append("\n");
            }
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File("cmds.txt")))) {
                bw.write(strb.toString());
            } catch (IOException ignored) {

            }
        }
    }

    private void setStartup(boolean b) {
        Process p;
        File superSearch = new File(MainClass.name);
        if (b) {
            try {
                String currentPath = System.getProperty("user.dir");
                File file = new File(currentPath);
                for (File each : Objects.requireNonNull(file.listFiles())) {
                    if (each.getName().contains("elevated")) {
                        superSearch = each;
                        break;
                    }
                }
                p = Runtime.getRuntime().exec("reg add \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\" /v superSearch /t REG_SZ /d " + "\"" + superSearch.getAbsolutePath() + "\"" + " /f");
                p.waitFor();
                BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                String line;
                StringBuilder result = new StringBuilder();
                while ((line = outPut.readLine()) != null) {
                    result.append(line);
                }
                if (!result.toString().equals("")) {
                    checkBox1.setSelected(false);
                    JOptionPane.showMessageDialog(null, "��ӵ���������ʧ�ܣ��볢���Թ���Ա�������");
                }
            } catch (IOException | InterruptedException ignored) {

            }
        } else {
            try {
                p = Runtime.getRuntime().exec("reg delete \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\" /v superSearch /f");
                p.waitFor();
                BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                String line;
                StringBuilder result = new StringBuilder();
                while ((line = outPut.readLine()) != null) {
                    result.append(line);
                }
                if (!result.toString().equals("")) {
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