package frames;


import getAscII.GetAscII;
import getIcon.GetIcon;
import main.MainClass;
import search.Search;

import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.Transferable;
import java.awt.event.*;
import java.io.*;
import java.net.URL;
import java.util.*;
import java.util.concurrent.*;
import java.util.regex.Pattern;

import static main.MainClass.mainExit;


public class SearchBar {
    private volatile static SearchBar searchBarInstance = new SearchBar();
    private JFrame searchBar = new JFrame();
    private CopyOnWriteArrayList<String> listResult = new CopyOnWriteArrayList<>();
    private ConcurrentHashMap<String, ReaderInfo> readerMap = new ConcurrentHashMap<>();
    private JLabel label1;
    private JLabel label2;
    private JLabel label3;
    private JLabel label4;
    private boolean isOpenLastFolderPressed = false;
    private int labelCount = 0;
    private JTextField textField;
    private Search search = Search.getInstance();
    private Color labelColor;
    private Color backgroundColor1;
    private Color backgroundColor2;
    private Color backgroundColor3;
    private Color backgroundColor4;
    private Color fontColorWithCoverage;
    private Color fontColor;
    private long startTime = 0;
    private boolean timer = false;
    private Thread searchWaiter = null;
    private boolean isUsing = false;
    private boolean isRunAsAdminPressed = false;
    private Pattern semicolon = Pattern.compile(";");
    private Pattern resultSplit = Pattern.compile(":");
    private boolean isUserPressed = false;
    private boolean isCommandMode = false;
    private boolean isLockMouseMotion = false;
    private JPanel panel = new JPanel();
    private long mouseWheelTime = 0;
    private boolean isCopyPathPressed = false;
    private int iconSideLength;
    private long visibleStartTime = 0;


    private SearchBar() {
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // ��ȡ��Ļ��С
        int width = screenSize.width;
        int height = screenSize.height;
        int searchBarWidth = (int) (width * 0.4);
        int searchBarHeight = (int) (height * 0.5);
        int positionX = width / 2 - searchBarWidth / 2;
        int positionY = height / 2 - searchBarHeight / 2;

        Border border = BorderFactory.createLineBorder(new Color(73, 162, 255, 255));

        labelColor = new Color(SettingsFrame.labelColor);
        fontColorWithCoverage = new Color(SettingsFrame.fontColorWithCoverage);
        backgroundColor2 = new Color(SettingsFrame.backgroundColor2);
        backgroundColor1 = new Color(SettingsFrame.backgroundColor1);
        backgroundColor3 = new Color(SettingsFrame.backgroundColor3);
        backgroundColor4 = new Color(SettingsFrame.backgroundColor4);
        fontColor = new Color(SettingsFrame.fontColor);

        //frame
        searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
        searchBar.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        searchBar.setUndecorated(true);
        searchBar.getRootPane().setWindowDecorationStyle(JRootPane.NONE);
        searchBar.setBackground(null);
        searchBar.setOpacity(SettingsFrame.transparency);
        searchBar.setContentPane(panel);
        searchBar.setType(JFrame.Type.UTILITY);


        //labels
        Font font = new Font("Microsoft JhengHei", Font.BOLD, (int) ((height * 0.1) / 96 * 72) / 4);
        label1 = new JLabel();
        label1.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label1.setLocation(0, (int) (searchBarHeight * 0.2));
        label1.setFont(font);
        label1.setForeground(fontColor);
        label1.setOpaque(true);
        label1.setBackground(null);

        iconSideLength = label1.getHeight() / 3; //����ͼ��߳�

        label2 = new JLabel();
        label2.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label2.setLocation(0, (int) (searchBarHeight * 0.4));
        label2.setFont(font);
        label2.setForeground(fontColor);
        label2.setOpaque(true);
        label2.setBackground(null);

        label3 = new JLabel();
        label3.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label3.setLocation(0, (int) (searchBarHeight * 0.6));
        label3.setFont(font);
        label3.setForeground(fontColor);
        label3.setOpaque(true);
        label3.setBackground(null);

        label4 = new JLabel();
        label4.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label4.setLocation(0, (int) (searchBarHeight * 0.8));
        label4.setFont(font);
        label4.setForeground(fontColor);
        label4.setOpaque(true);
        label4.setBackground(null);


        URL icon = TaskBar.class.getResource("/icons/taskbar_32x32.png");
        Image image = new ImageIcon(icon).getImage();
        searchBar.setIconImage(image);
        Color transparentColor = new Color(0, 0, 0, 0);
        searchBar.setBackground(transparentColor);


        //TextField
        textField = new JTextField(300);
        textField.setSize(searchBarWidth - 6, (int) (searchBarHeight * 0.2) - 5);
        Font textFieldFont = new Font("Microsoft JhengHei", Font.PLAIN, (int) (((height * 0.1) / 96 * 72 / 1.2)));
        textField.setFont(textFieldFont);
        textField.setBorder(border);
        textField.setForeground(Color.BLACK);
        textField.setHorizontalAlignment(JTextField.LEFT);
        textField.setBackground(Color.WHITE);
        textField.setLocation(3, 0);
        textField.setOpaque(true);
        textField.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent e) {

            }

            @Override
            public void focusLost(FocusEvent e) {
                if (System.currentTimeMillis() - visibleStartTime > 500) {
                    if (SettingsFrame.isLoseFocusClose) {
                        closedTodo();
                    }
                }
            }
        });

        //panel
        panel.setLayout(null);
        panel.setBackground(transparentColor);
        panel.add(textField);
        panel.add(label1);
        panel.add(label2);
        panel.add(label3);
        panel.add(label4);


        ExecutorService fixedThreadPool = Executors.newFixedThreadPool(7);

        fixedThreadPool.execute(() -> {
            //���ñ߿�
            while (!mainExit) {
                int size = listResult.size();
                try {
                    if (size == 1) {
                        label1.setBorder(border);
                        label2.setBorder(null);
                        label3.setBorder(null);
                        label4.setBorder(null);
                    } else if (size == 2) {
                        label1.setBorder(border);
                        label2.setBorder(border);
                        label3.setBorder(null);
                        label4.setBorder(null);
                    } else if (size == 3) {
                        label1.setBorder(border);
                        label2.setBorder(border);
                        label3.setBorder(border);
                        label4.setBorder(null);
                    } else if (size > 3) {
                        label1.setBorder(border);
                        label2.setBorder(border);
                        label3.setBorder(border);
                        label4.setBorder(border);
                    }
                    Thread.sleep(5);
                } catch (NullPointerException | InterruptedException ignored) {

                }
            }
        });

        fixedThreadPool.execute(() -> {
            //��סMouseMotion��⣬��ֹͬʱ������������
            try {
                while (!mainExit) {
                    if (System.currentTimeMillis() - mouseWheelTime > 500) {
                        isLockMouseMotion = false;
                    }
                    Thread.sleep(5);
                }
            } catch (Exception ignored) {

            }
        });

        fixedThreadPool.execute(() -> {
            //���ӹ����߳�
            try {
                while (!mainExit) {
                    if ((!isUsing) && Search.getInstance().isUsable()) {
                        for (String eachKey : readerMap.keySet()) {
                            ReaderInfo info = readerMap.get(eachKey);
                            if (System.currentTimeMillis() - info.time > SettingsFrame.connectionTimeLimit) {
                                info.reader.close();
                                readerMap.remove(eachKey, info);
                            }
                        }
                        if (readerMap.size() < SettingsFrame.minConnectionNum) {
                            Random random = new Random();
                            String path;
                            int randomInt = random.nextInt(26);
                            if (randomInt < 25) {
                                path = SettingsFrame.dataPath + "\\" + "\\list" + randomInt * 100 + "-" + (randomInt + 1) * 100 + ".txt";
                            } else {
                                path = SettingsFrame.dataPath + "\\" + "\\list2500-.txt";
                            }
                            if (isExist(path)) {
                                if (readerMap.get(path) == null) {
                                    ReaderInfo _tmp = new ReaderInfo(System.currentTimeMillis(), new BufferedReader(new FileReader(path)));
                                    readerMap.put(path, _tmp);
                                }
                            }
                        }
                    }
                    Thread.sleep(500);
                }
                closeAllConnection();
            } catch (InterruptedException | IOException e) {
                e.printStackTrace();
            }
        });

        fixedThreadPool.execute(() -> {
            try {
                while (!mainExit) {
                    //����Ⱦɫ�߳�
                    //�ж���ǰѡ��λ��
                    int position;
                    if (label1.getBackground() == labelColor) {
                        position = 0;
                    } else if (label2.getBackground() == labelColor) {
                        position = 1;
                    } else if (label3.getBackground() == labelColor) {
                        position = 2;
                    } else {
                        position = 3;
                    }
                    if (position == 0) {
                        label1.setForeground(fontColorWithCoverage);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColor);
                    } else if (position == 1) {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColorWithCoverage);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColor);
                    } else if (position == 2) {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColorWithCoverage);
                        label4.setForeground(fontColor);
                    } else {
                        label1.setForeground(fontColor);
                        label2.setForeground(fontColor);
                        label3.setForeground(fontColor);
                        label4.setForeground(fontColorWithCoverage);
                    }
                    Thread.sleep(5);
                }
            } catch (Exception ignored) {

            }
        });

        fixedThreadPool.execute(() -> {
            //��ʾ����߳�
            while (!mainExit) {
                if (labelCount < listResult.size()) {//�н��������ʾ
                    try {
                        if (label2.getText().isEmpty() || label3.getText().isEmpty() || label4.getText().isEmpty() || label1.getText().isEmpty()) {
                            showResults();
                        }
                    } catch (NullPointerException e) {
                        showResults();
                    }
                }
                String text = textField.getText();
                if (text.isEmpty()) {
                    clearLabel();
                    listResult.clear();
                }
                try {
                    if (!isUserPressed && !label1.getText().isEmpty()) {
                        if (labelCount == 0) {
                            label1.setBackground(labelColor);
                        } else {
                            label1.setBackground(backgroundColor1);
                        }
                    }
                    if (!isUserPressed && !label2.getText().isEmpty()) {
                        if (labelCount == 1) {
                            label2.setBackground(labelColor);
                        } else {
                            label2.setBackground(backgroundColor2);
                        }
                    }
                    if (!isUserPressed && !label3.getText().isEmpty()) {
                        if (labelCount == 2) {
                            label3.setBackground(labelColor);
                        } else {
                            label3.setBackground(backgroundColor3);
                        }
                    }
                    if (!isUserPressed && !label4.getText().isEmpty()) {
                        if (labelCount == 3) {
                            label4.setBackground(labelColor);
                        } else {
                            label4.setBackground(backgroundColor4);
                        }
                    }
                } catch (NullPointerException ignored) {

                }
                try {
                    Thread.sleep(10);
                } catch (InterruptedException ignored) {

                }
            }
        });

        fixedThreadPool.execute(() -> {
            //��⻺���С ����ʱ��������
            try {
                while (!mainExit) {
                    if (!search.isManualUpdate() && !isUsing) {
                        if (search.getRecycleBinSize() > 1000) {
                            closeAllConnection();
                            System.out.println("�Ѽ�⵽����վ�����Զ�����");
                            search.setUsable(false);
                            search.mergeAndClearRecycleBin();
                            search.setUsable(true);
                        }
                    }
                    Thread.sleep(50);
                }
            } catch (InterruptedException ignored) {

            }
        });

        fixedThreadPool.execute(() -> {
            //����insertUpdate����Ϣ����������
            //ͣ��ʱ��0.5s��ÿһ����������һ��startTime�����̼߳�¼endTime
            while (!mainExit) {
                long endTime = System.currentTimeMillis();
                if ((endTime - startTime > 500) && (timer)) {
                    timer = false; //��ʼ���� ��ʱֹͣ
                    labelCount = 0;
                    clearLabel();
                    if (!textField.getText().isEmpty()) {
                        label1.setBackground(labelColor);
                    } else {
                        clearLabel();
                    }
                    listResult.clear();
                    String text = textField.getText();
                    if (search.isUsable()) {
                        if (isCommandMode) {
                            if (text.equals(":update")) {
                                closeAllConnection();
                                clearLabel();
                                MainClass.showMessage("��ʾ", "���ڸ����ļ�����");
                                clearTextFieldText();
                                closedTodo();
                                search.setManualUpdate(true);
                                timer = false;
                                continue;
                            }
                            if (text.equals(":version")) {
                                clearLabel();
                                clearTextFieldText();
                                closedTodo();
                                JOptionPane.showMessageDialog(null, "��ǰ�汾��" + MainClass.version);
                            }
                            if (text.equals(":help")) {
                                clearLabel();
                                clearTextFieldText();
                                closedTodo();
                                JOptionPane.showMessageDialog(null, "������\n" +
                                        "1.Ĭ��Ctrl + Alt + J��������\n" +
                                        "2.Enter�����г���\n" +
                                        "3.Ctrl + Enter���򿪲�ѡ���ļ������ļ���\n" +
                                        "4.Shift + Enter���Թ���ԱȨ�����г���ǰ���Ǹó���ӵ�й���ԱȨ�ޣ�\n" +
                                        "5.��������������  : update  ǿ���ؽ���������\n" +
                                        "6.��������������  : version  �鿴��ǰ�汾\n" +
                                        "7.��������������  : clearbin  ��ջ���վ\n" +
                                        "8.��������������  : help  �鿴����\"\n" +
                                        "9.�������п����Զ��������������������  : �Զ����ʶ  �����Լ�������\n" +
                                        "10.��������ļ���������  : full  ��ȫ��ƥ��\n" +
                                        "11.��������ļ���������  : F  ��ֻƥ���ļ�\n" +
                                        "12.��������ļ���������  : D  ��ֻƥ���ļ���\n" +
                                        "13.��������ļ���������  : Ffull  ��ֻƥ���ļ���ȫ��ƥ��\n" +
                                        "14.��������ļ���������  : Dfull  ��ֻƥ���ļ��в�ȫ��ƥ��");
                            }
                            if (text.equals(":clearbin")) {
                                clearLabel();
                                clearTextFieldText();
                                closedTodo();
                                int r = JOptionPane.showConfirmDialog(null, "��ȷ��Ҫ��ջ���վ��");
                                if (r == 0) {
                                    try {
                                        File[] roots = File.listRoots();
                                        for (File root : roots) {
                                            Process p = Runtime.getRuntime().exec("cmd /c rd /s /q " + root.getAbsolutePath() + "$Recycle.Bin");
                                            p.getErrorStream().close();
                                            p.getOutputStream().close();
                                            p.getInputStream().close();
                                        }
                                        JOptionPane.showMessageDialog(null, "��ջ���վ�ɹ�");
                                    } catch (IOException e) {
                                        JOptionPane.showMessageDialog(null, "��ջ���վʧ��");
                                    }
                                }
                            }
                            for (String i : SettingsFrame.cmdSet) {
                                if (i.startsWith(text)) {
                                    listResult.add("��������" + i);
                                }
                                String[] cmdInfo = semicolon.split(i);
                                if (cmdInfo[0].equals(text)) {
                                    clearLabel();
                                    clearTextFieldText();
                                    closedTodo();
                                    openWithAdmin(cmdInfo[1]);
                                }
                            }
                            showResults();
                        } else {
                            String[] strings;
                            String searchText;
                            int length;
                            String searchCase;
                            strings = resultSplit.split(text);
                            length = strings.length;
                            if (length == 2) {
                                searchCase = strings[1].toLowerCase();
                                searchText = strings[0];
                            } else {
                                searchText = strings[0];
                                searchCase = "";
                            }
                            searchPriorityFolder(searchText, searchCase);
                            searchCache(searchText, searchCase);
                            showResults();

                            String listPath;
                            int ascII = getAscIISum(searchText);
                            ConcurrentLinkedQueue<String> paths = new ConcurrentLinkedQueue<>();

                            if (0 < ascII && ascII <= 100) {
                                for (int i = 0; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);
                            } else if (100 < ascII && ascII <= 200) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 100; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (200 < ascII && ascII <= 300) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 200; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (300 < ascII && ascII <= 400) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 300; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (400 < ascII && ascII <= 500) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 400; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (500 < ascII && ascII <= 600) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 500; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (600 < ascII && ascII <= 700) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 600; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (700 < ascII && ascII <= 800) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 700; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (800 < ascII && ascII <= 900) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 800; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (900 < ascII && ascII <= 1000) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 900; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1000 < ascII && ascII <= 1100) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1000; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1100 < ascII && ascII <= 1200) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1100; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1200 < ascII && ascII <= 1300) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1200; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1300 < ascII && ascII <= 1400) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1300; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1400 < ascII && ascII <= 1500) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1400; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1500 < ascII && ascII <= 1600) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1500; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1600 < ascII && ascII <= 1700) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1600; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1700 < ascII && ascII <= 1800) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1700; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1800 < ascII && ascII <= 1900) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1800; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (1900 < ascII && ascII <= 2000) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 1900; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (2000 < ascII && ascII <= 2100) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 2000; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (2100 < ascII && ascII <= 2200) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 2100; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (2200 < ascII && ascII <= 2300) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 2200; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (2300 < ascII && ascII <= 2400) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 2300; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else if (2400 < ascII && ascII <= 2500) {
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list0-100.txt");
                                for (int i = 2400; i < 2500; i += 100) {
                                    int name = i + 100;
                                    listPath = SettingsFrame.dataPath + "\\" + "\\list" + i + "-" + name + ".txt";
                                    paths.add(listPath);
                                }
                                paths.add(SettingsFrame.dataPath + "\\" + "\\list2500-.txt");
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            } else {
                                for (int j = 0; j < SettingsFrame.diskCount; j++) {
                                    paths.add(SettingsFrame.dataPath + "\\" + j + "\\list2500-.txt");
                                }
                                addResult(paths, searchText, System.currentTimeMillis(), searchCase);

                            }
                        }
                    } else {
                        if (search.isManualUpdate()) {
                            if (searchWaiter == null || !searchWaiter.isAlive()) {
                                searchWaiter = new Thread(() -> {
                                    while (!mainExit) {
                                        if (Thread.currentThread().isInterrupted()) {
                                            break;
                                        }
                                        if (search.isUsable()) {
                                            startTime = System.currentTimeMillis() - 500;
                                            timer = true;
                                            break;
                                        }
                                        try {
                                            Thread.sleep(20);
                                        } catch (InterruptedException ignored) {

                                        }
                                    }
                                });
                                searchWaiter.start();
                            }
                        }
                        clearLabel();
                        if (!search.isUsable()) {
                            label1.setBackground(labelColor);
                            label1.setText("���ڽ�������...");
                        }
                    }
                }
                try {
                    Thread.sleep(20);
                } catch (InterruptedException ignored) {

                }
            }
        });

        textField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                panel.repaint();
                clearLabel();
                listResult.clear();
                labelCount = 0;
                startTime = System.currentTimeMillis();
                timer = true;
                isCommandMode = textField.getText().charAt(0) == ':';
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                panel.repaint();
                clearLabel();
                listResult.clear();
                labelCount = 0;
                String t = textField.getText();

                if (t.isEmpty()) {
                    clearLabel();
                    listResult.clear();
                    labelCount = 0;
                    startTime = System.currentTimeMillis();
                    timer = false;
                } else {
                    startTime = System.currentTimeMillis();
                    timer = true;
                }
                try {
                    isCommandMode = textField.getText().charAt(0) == ':';
                } catch (StringIndexOutOfBoundsException ignored) {

                }
            }

            @Override
            public void changedUpdate(DocumentEvent e) {

            }
        });

        searchBar.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                panel.repaint();
            }

            @Override
            public void mousePressed(MouseEvent e) {
                panel.repaint();
                int count = e.getClickCount();
                if (count == 2) {
                    closedTodo();
                    if (listResult.size() != 0) {
                        if (!isCommandMode) {
                            if (isOpenLastFolderPressed) {
                                //���ϼ��ļ���
                                File open = new File(listResult.get(labelCount));
                                try {
                                    Process p = Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                    p.getInputStream().close();
                                    p.getOutputStream().close();
                                    p.getErrorStream().close();
                                } catch (IOException e1) {
                                    e1.printStackTrace();
                                }
                            } else if (SettingsFrame.isDefaultAdmin || isRunAsAdminPressed) {
                                openWithAdmin(listResult.get(labelCount));
                            } else {
                                String openFile = listResult.get(labelCount);
                                if (openFile.endsWith(".bat") || openFile.endsWith(".cmd")) {
                                    openWithAdmin(openFile);
                                } else {
                                    openWithoutAdmin(openFile);
                                }
                            }
                            saveCache(listResult.get(labelCount) + ';');
                        } else {
                            //ֱ�Ӵ�
                            String command = listResult.get(labelCount);
                            if (Desktop.isDesktopSupported()) {
                                Desktop desktop = Desktop.getDesktop();
                                try {
                                    desktop.open(new File(semicolon.split(command)[1]));
                                } catch (Exception e1) {
                                    JOptionPane.showMessageDialog(null, "ִ��ʧ��");
                                }
                            }
                        }
                    }
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                panel.repaint();
            }

            @Override
            public void mouseEntered(MouseEvent e) {
                panel.repaint();
            }

            @Override
            public void mouseExited(MouseEvent e) {
                panel.repaint();
            }
        });

        searchBar.addMouseWheelListener(e -> {
            panel.repaint();
            mouseWheelTime = System.currentTimeMillis();
            isLockMouseMotion = true;
            if (e.getPreciseWheelRotation() > 0) {
                //���¹���
                try {
                    if (!label1.getText().isEmpty() && !label2.getText().isEmpty() && !label3.getText().isEmpty() && !label4.getText().isEmpty()) {
                        isUserPressed = true;
                    }
                } catch (NullPointerException ignored) {

                }
                boolean isNextExist = false;
                if (labelCount == 0) {
                    try {
                        if (!label2.getText().isEmpty()) {
                            isNextExist = true;
                        }
                    } catch (NullPointerException ignored) {

                    }
                } else if (labelCount == 1) {
                    try {
                        if (!label3.getText().isEmpty()) {
                            isNextExist = true;
                        }
                    } catch (NullPointerException ignored) {

                    }
                } else if (labelCount == 2) {
                    try {
                        if (!label4.getText().isEmpty()) {
                            isNextExist = true;
                        }
                    } catch (NullPointerException ignored) {

                    }
                } else {
                    isNextExist = true;
                }
                if (isNextExist) {
                    if (!textField.getText().isEmpty()) {
                        labelCount++;
                        if (labelCount < 0) {
                            labelCount = 0;
                        }

                        //System.out.println(labelCount);
                        if (labelCount >= listResult.size()) {
                            labelCount = listResult.size() - 1;
                        }
                        //�ж���ǰѡ��λ��
                        int position;
                        try {
                            if (label1.getBackground() == labelColor) {
                                position = 0;
                            } else if (label2.getBackground() == labelColor) {
                                position = 1;
                            } else if (label3.getBackground() == labelColor) {
                                position = 2;
                            } else {
                                position = 3;
                            }
                            if (!isCommandMode) {
                                switch (position) {
                                    case 0:
                                        int size = listResult.size();
                                        if (size == 2) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(labelColor);
                                        } else if (size == 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(labelColor);
                                            label3.setBackground(backgroundColor3);
                                        } else if (size > 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(labelColor);
                                            label3.setBackground(backgroundColor3);
                                            label4.setBackground(backgroundColor4);
                                        }
                                        break;
                                    case 1:
                                        size = listResult.size();
                                        if (size == 2) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                        } else if (size == 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                            label3.setBackground(labelColor);
                                        } else if (size > 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                            label3.setBackground(labelColor);
                                            label4.setBackground(backgroundColor4);
                                        }
                                        break;
                                    case 2:
                                        size = listResult.size();
                                        if (size == 2) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                        } else if (size == 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                            label3.setBackground(backgroundColor3);
                                        } else if (size > 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                            label3.setBackground(backgroundColor3);
                                            label4.setBackground(labelColor);
                                        }
                                        break;
                                    case 3:
                                        //�������¶ˣ�ˢ����ʾ
                                        try {
                                            String path = listResult.get(labelCount - 3);
                                            String name = getFileName(listResult.get(labelCount - 3));
                                            ImageIcon icon1;
                                            if (isDirectory(path) || isFile(path)) {
                                                icon1 = (ImageIcon) GetIcon.getBigIcon(path);
                                                icon1 = changeIcon(icon1, iconSideLength, iconSideLength);
                                                label1.setIcon(icon1);
                                                label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</font></body></html>");
                                            } else {
                                                label1.setIcon(null);
                                                label1.setText("��Ч�ļ�");
                                                search.addToRecycleBin(path);
                                            }
                                            path = listResult.get(labelCount - 2);
                                            name = getFileName(listResult.get(labelCount - 2));

                                            if (isDirectory(path) || isFile(path)) {
                                                icon1 = (ImageIcon) GetIcon.getBigIcon(path);
                                                icon1 = changeIcon(icon1, iconSideLength, iconSideLength);
                                                label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                label2.setIcon(icon1);
                                            } else {
                                                label2.setIcon(null);
                                                label2.setText("��Ч�ļ�");
                                                search.addToRecycleBin(path);
                                            }
                                            path = listResult.get(labelCount - 1);
                                            name = getFileName(listResult.get(labelCount - 1));


                                            if (isDirectory(path) || isFile(path)) {
                                                icon1 = (ImageIcon) GetIcon.getBigIcon(path);
                                                icon1 = changeIcon(icon1, iconSideLength, iconSideLength);
                                                label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                label3.setIcon(icon1);
                                            } else {
                                                label3.setIcon(null);
                                                label3.setText("��Ч�ļ�");
                                                search.addToRecycleBin(path);
                                            }
                                            path = listResult.get(labelCount);
                                            name = getFileName(listResult.get(labelCount));


                                            if (isDirectory(path) || isFile(path)) {
                                                icon1 = (ImageIcon) GetIcon.getBigIcon(path);
                                                icon1 = changeIcon(icon1, iconSideLength, iconSideLength);
                                                label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                label4.setIcon(icon1);
                                            } else {
                                                label4.setIcon(null);
                                                label4.setText("��Ч�ļ�");
                                                search.addToRecycleBin(path);
                                            }
                                        } catch (ArrayIndexOutOfBoundsException ignored) {

                                        }
                                        break;
                                }
                            } else {
                                switch (position) {
                                    case 0:
                                        int size = listResult.size();
                                        if (size == 2) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(labelColor);
                                        } else if (size == 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(labelColor);
                                            label3.setBackground(backgroundColor3);
                                        } else if (size > 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(labelColor);
                                            label3.setBackground(backgroundColor3);
                                            label4.setBackground(backgroundColor4);
                                        }
                                        break;
                                    case 1:
                                        size = listResult.size();
                                        if (size == 2) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                        } else if (size == 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                            label3.setBackground(labelColor);
                                        } else if (size > 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                            label3.setBackground(labelColor);
                                            label4.setBackground(backgroundColor4);
                                        }
                                        break;
                                    case 2:
                                        size = listResult.size();
                                        if (size == 2) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                        } else if (size == 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                            label3.setBackground(backgroundColor3);
                                        } else if (size > 3) {
                                            label1.setBackground(backgroundColor1);
                                            label2.setBackground(backgroundColor2);
                                            label3.setBackground(backgroundColor3);
                                            label4.setBackground(labelColor);
                                        }
                                        break;
                                    case 3:
                                        //���������¶ˣ�ˢ����ʾ
                                        try {
                                            String command = listResult.get(labelCount - 3);
                                            String[] info = semicolon.split(command);
                                            String path = info[1];
                                            String name = info[0];
                                            ImageIcon imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                            imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                            label1.setIcon(imageIcon);
                                            label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                            command = listResult.get(labelCount - 2);
                                            info = semicolon.split(command);
                                            path = info[1];
                                            name = info[0];
                                            imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                            imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                            label2.setIcon(imageIcon);
                                            label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                            command = listResult.get(labelCount - 1);
                                            info = semicolon.split(command);
                                            path = info[1];
                                            name = info[0];
                                            imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                            imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                            label3.setIcon(imageIcon);
                                            label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                            command = listResult.get(labelCount);
                                            info = semicolon.split(command);
                                            path = info[1];
                                            name = info[0];
                                            imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                            imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                            label4.setIcon(imageIcon);
                                            label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");
                                        } catch (ArrayIndexOutOfBoundsException ignored) {

                                        }
                                        break;
                                }
                            }
                        } catch (NullPointerException ignored) {

                        }
                    }
                }
            } else if (e.getPreciseWheelRotation() < 0) {
                //���Ϲ���
                try {
                    if (!label1.getText().isEmpty() && !label2.getText().isEmpty() && !label3.getText().isEmpty() && !label4.getText().isEmpty()) {
                        isUserPressed = true;
                    }
                } catch (NullPointerException ignored) {

                }
                if (!textField.getText().isEmpty()) {
                    labelCount--;
                    if (labelCount < 0) {
                        labelCount = 0;
                    }

                    //System.out.println(labelCount);
                    if (labelCount >= listResult.size()) {
                        labelCount = listResult.size() - 1;
                    }

                    //�ж���ǰѡ��λ��
                    int position;
                    try {
                        if (label1.getBackground() == labelColor) {
                            position = 0;
                        } else if (label2.getBackground() == labelColor) {
                            position = 1;
                        } else if (label3.getBackground() == labelColor) {
                            position = 2;
                        } else {
                            position = 3;
                        }
                        if (!isCommandMode) {
                            switch (position) {
                                case 0:
                                    //���������϶ˣ�ˢ����ʾ
                                    try {
                                        String path = listResult.get(labelCount);
                                        String name = getFileName(listResult.get(labelCount));
                                        ImageIcon icon1;
                                        if (isDirectory(path) || isFile(path)) {
                                            icon1 = (ImageIcon) GetIcon.getBigIcon(path);
                                            icon1 = changeIcon(icon1, iconSideLength, iconSideLength);
                                            label1.setIcon(icon1);
                                            label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                        } else {
                                            label1.setIcon(null);
                                            label1.setText("��Ч�ļ�");
                                            search.addToRecycleBin(path);
                                        }
                                        path = listResult.get(labelCount + 1);
                                        name = getFileName(listResult.get(labelCount + 1));

                                        if (isDirectory(path) || isFile(path)) {
                                            icon1 = (ImageIcon) GetIcon.getBigIcon(path);
                                            icon1 = changeIcon(icon1, iconSideLength, iconSideLength);
                                            label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                            label2.setIcon(icon1);
                                        } else {
                                            label2.setIcon(null);
                                            label2.setText("��Ч�ļ�");
                                            search.addToRecycleBin(path);
                                        }
                                        path = listResult.get(labelCount + 2);
                                        name = getFileName(listResult.get(labelCount + 2));


                                        if (isDirectory(path) || isFile(path)) {
                                            icon1 = (ImageIcon) GetIcon.getBigIcon(path);
                                            icon1 = changeIcon(icon1, iconSideLength, iconSideLength);
                                            label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                            label3.setIcon(icon1);
                                        } else {
                                            label3.setIcon(null);
                                            label3.setText("��Ч�ļ�");
                                            search.addToRecycleBin(path);
                                        }
                                        path = listResult.get(labelCount + 3);
                                        name = getFileName(listResult.get(labelCount + 3));


                                        if (isDirectory(path) || isFile(path)) {
                                            icon1 = (ImageIcon) GetIcon.getBigIcon(path);
                                            icon1 = changeIcon(icon1, iconSideLength, iconSideLength);
                                            label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                            label4.setIcon(icon1);
                                        } else {
                                            label4.setIcon(null);
                                            label4.setText("��Ч�ļ�");
                                            search.addToRecycleBin(path);
                                        }
                                    } catch (ArrayIndexOutOfBoundsException ignored) {

                                    }
                                    break;
                                case 1:
                                    try {
                                        if (!label1.getText().isEmpty()) {
                                            label1.setBackground(labelColor);
                                        }
                                        if (!label2.getText().isEmpty()) {
                                            label2.setBackground(backgroundColor2);
                                        }
                                        if (!label3.getText().isEmpty()) {
                                            label3.setBackground(backgroundColor3);
                                        }
                                        if (!label4.getText().isEmpty()) {
                                            label4.setBackground(backgroundColor4);
                                        }
                                    } catch (NullPointerException ignored) {

                                    }
                                    break;
                                case 2:
                                    try {
                                        if (!label1.getText().isEmpty()) {
                                            label1.setBackground(backgroundColor1);
                                        }
                                        if (!label2.getText().isEmpty()) {
                                            label2.setBackground(labelColor);
                                        }
                                        if (!label3.getText().isEmpty()) {
                                            label3.setBackground(backgroundColor3);
                                        }
                                        if (!label4.getText().isEmpty()) {
                                            label4.setBackground(backgroundColor4);
                                        }
                                    } catch (NullPointerException ignored) {

                                    }
                                    break;
                                case 3:
                                    try {
                                        if (!label1.getText().isEmpty()) {
                                            label1.setBackground(backgroundColor1);
                                        }
                                        if (!label2.getText().isEmpty()) {
                                            label2.setBackground(backgroundColor2);
                                        }
                                        if (!label3.getText().isEmpty()) {
                                            label3.setBackground(labelColor);
                                        }
                                        if (!label4.getText().isEmpty()) {
                                            label4.setBackground(backgroundColor4);
                                        }
                                    } catch (NullPointerException ignored) {

                                    }
                                    break;
                            }
                        } else {
                            switch (position) {
                                case 0:
                                    //���������϶ˣ�ˢ����ʾ
                                    try {
                                        String command = listResult.get(labelCount);
                                        String[] info = semicolon.split(command);
                                        String path = info[1];
                                        String name = info[0];
                                        ImageIcon imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                        imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                        label1.setIcon(imageIcon);
                                        label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                        command = listResult.get(labelCount + 1);
                                        info = semicolon.split(command);
                                        path = info[1];
                                        name = info[0];
                                        imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                        imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                        label2.setIcon(imageIcon);
                                        label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                        command = listResult.get(labelCount + 2);
                                        info = semicolon.split(command);
                                        path = info[1];
                                        name = info[0];
                                        imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                        imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                        label3.setIcon(imageIcon);
                                        label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                        command = listResult.get(labelCount + 3);
                                        info = semicolon.split(command);
                                        path = info[1];
                                        name = info[0];
                                        imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                        imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                        label4.setIcon(imageIcon);
                                        label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");
                                    } catch (ArrayIndexOutOfBoundsException ignored) {

                                    }
                                    break;
                                case 1:
                                    try {
                                        if (!label1.getText().isEmpty()) {
                                            label1.setBackground(labelColor);
                                        }
                                        if (!label2.getText().isEmpty()) {
                                            label2.setBackground(backgroundColor2);
                                        }
                                        if (!label3.getText().isEmpty()) {
                                            label3.setBackground(backgroundColor3);
                                        }
                                        if (!label4.getText().isEmpty()) {
                                            label4.setBackground(backgroundColor4);
                                        }
                                    } catch (NullPointerException ignored) {

                                    }
                                    break;
                                case 2:
                                    try {
                                        if (!label1.getText().isEmpty()) {
                                            label1.setBackground(backgroundColor1);
                                        }
                                        if (!label2.getText().isEmpty()) {
                                            label2.setBackground(labelColor);
                                        }
                                        if (!label3.getText().isEmpty()) {
                                            label3.setBackground(backgroundColor3);
                                        }
                                        if (!label4.getText().isEmpty()) {
                                            label4.setBackground(backgroundColor4);
                                        }
                                    } catch (NullPointerException ignored) {

                                    }
                                    break;
                                case 3:
                                    try {
                                        if (!label1.getText().isEmpty()) {
                                            label1.setBackground(backgroundColor1);
                                        }
                                        if (!label2.getText().isEmpty()) {
                                            label2.setBackground(backgroundColor2);
                                        }
                                        if (!label3.getText().isEmpty()) {
                                            label3.setBackground(labelColor);
                                        }
                                        if (!label4.getText().isEmpty()) {
                                            label4.setBackground(backgroundColor4);
                                        }
                                    } catch (NullPointerException ignored) {

                                    }
                                    break;
                                case 4:
                                    try {
                                        if (!label1.getText().isEmpty()) {
                                            label1.setBackground(backgroundColor1);
                                        }
                                        if (!label2.getText().isEmpty()) {
                                            label2.setBackground(backgroundColor2);
                                        }
                                        if (!label3.getText().isEmpty()) {
                                            label3.setBackground(backgroundColor3);
                                        }
                                        if (!label4.getText().isEmpty()) {
                                            label4.setBackground(labelColor);
                                        }
                                    } catch (NullPointerException ignored) {

                                    }
                                    break;
                            }
                        }
                    } catch (NullPointerException ignored) {

                    }

                    if (labelCount < 0) {
                        labelCount = 0;
                    }
                }

            }
        });

        //�ж����λ��
        int labelPosition = label1.getY();
        int labelPosition2 = labelPosition * 2;
        int labelPosition3 = labelPosition * 3;
        int labelPosition4 = labelPosition * 4;
        int end = labelPosition * 5;
        searchBar.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent e) {
                //�ж���ǰλ��
                if (!isLockMouseMotion) {
                    int position;
                    if (label1.getBackground() == labelColor) {
                        position = 0;
                    } else if (label2.getBackground() == labelColor) {
                        position = 1;
                    } else if (label3.getBackground() == labelColor) {
                        position = 2;
                    } else {
                        position = 3;
                    }
                    int mousePosition = 0;
                    if (labelPosition2 <= e.getY() && e.getY() < labelPosition3) {
                        mousePosition = 1;
                    } else if (labelPosition3 <= e.getY() && e.getY() < labelPosition4) {
                        mousePosition = 2;
                    } else if (labelPosition4 <= e.getY() && e.getY() < end) {
                        mousePosition = 3;
                    }
                    if (mousePosition < listResult.size()) {
                        if (position < mousePosition) {
                            labelCount = labelCount + (mousePosition - position);
                        } else {
                            labelCount = labelCount - (position - mousePosition);
                        }
                        switch (mousePosition) {
                            case 0:
                                try {
                                    if (!label1.getText().isEmpty()) {
                                        label1.setBackground(labelColor);
                                    }
                                    if (!label2.getText().isEmpty()) {
                                        label2.setBackground(backgroundColor2);
                                    }
                                    if (!label3.getText().isEmpty()) {
                                        label3.setBackground(backgroundColor3);
                                    }
                                    if (!label4.getText().isEmpty()) {
                                        label4.setBackground(backgroundColor4);
                                    }
                                } catch (NullPointerException ignored) {

                                }
                                break;
                            case 1:
                                try {
                                    if (!label1.getText().isEmpty()) {
                                        label1.setBackground(backgroundColor1);
                                    }
                                    if (!label2.getText().isEmpty()) {
                                        label2.setBackground(labelColor);
                                    }
                                    if (!label3.getText().isEmpty()) {
                                        label3.setBackground(backgroundColor3);
                                    }
                                    if (!label4.getText().isEmpty()) {
                                        label4.setBackground(backgroundColor4);
                                    }
                                } catch (NullPointerException ignored) {

                                }
                                break;
                            case 2:
                                try {
                                    if (!label1.getText().isEmpty()) {
                                        label1.setBackground(backgroundColor1);
                                    }
                                    if (!label2.getText().isEmpty()) {
                                        label2.setBackground(backgroundColor2);
                                    }
                                    if (!label3.getText().isEmpty()) {
                                        label3.setBackground(labelColor);
                                    }
                                    if (!label4.getText().isEmpty()) {
                                        label4.setBackground(backgroundColor4);
                                    }
                                } catch (NullPointerException ignored) {

                                }
                                break;
                            case 3:
                                try {
                                    if (!label1.getText().isEmpty()) {
                                        label1.setBackground(backgroundColor1);
                                    }
                                    if (!label2.getText().isEmpty()) {
                                        label2.setBackground(backgroundColor2);
                                    }
                                    if (!label3.getText().isEmpty()) {
                                        label3.setBackground(backgroundColor3);
                                    }
                                    if (!label4.getText().isEmpty()) {
                                        label4.setBackground(labelColor);
                                    }
                                } catch (NullPointerException ignored) {

                                }
                                break;
                        }
                    }
                }
            }
        });

        textField.addKeyListener(new KeyListener() {
            int timeLimit = 50;
            long pressTime;
            boolean isFirstPress = true;

            @Override
            public void keyPressed(KeyEvent arg0) {
                panel.repaint();
                int key = arg0.getKeyCode();
                if (key == 8 && textField.getText().isEmpty()) {
                    arg0.consume();
                }
                if (!listResult.isEmpty()) {
                    if (38 == key) {
                        //�ϼ������
                        if (isFirstPress || System.currentTimeMillis() - pressTime > timeLimit) {
                            pressTime = System.currentTimeMillis();
                            isFirstPress = false;
                            try {
                                if (!label1.getText().isEmpty() && !label2.getText().isEmpty() && !label3.getText().isEmpty() && !label4.getText().isEmpty()) {
                                    isUserPressed = true;
                                }
                            } catch (NullPointerException ignored) {

                            }
                            if (!textField.getText().isEmpty()) {
                                labelCount--;
                                if (labelCount < 0) {
                                    labelCount = 0;
                                }

                                //System.out.println(labelCount);
                                if (labelCount >= listResult.size()) {
                                    labelCount = listResult.size() - 1;
                                }

                                //�ж���ǰѡ��λ��
                                int position;
                                try {
                                    if (label1.getBackground() == labelColor) {
                                        position = 0;
                                    } else if (label2.getBackground() == labelColor) {
                                        position = 1;
                                    } else if (label3.getBackground() == labelColor) {
                                        position = 2;
                                    } else {
                                        position = 3;
                                    }
                                    if (!isCommandMode) {
                                        switch (position) {
                                            case 0:
                                                //���������϶ˣ�ˢ����ʾ
                                                try {
                                                    String path = listResult.get(labelCount);
                                                    String name = getFileName(listResult.get(labelCount));
                                                    ImageIcon icon;
                                                    if (isDirectory(path) || isFile(path)) {
                                                        icon = (ImageIcon) GetIcon.getBigIcon(path);
                                                        icon = changeIcon(icon, iconSideLength, iconSideLength);
                                                        label1.setIcon(icon);
                                                        label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                    } else {
                                                        label1.setIcon(null);
                                                        label1.setText("��Ч�ļ�");
                                                        search.addToRecycleBin(path);
                                                    }
                                                    path = listResult.get(labelCount + 1);
                                                    name = getFileName(listResult.get(labelCount + 1));

                                                    if (isDirectory(path) || isFile(path)) {
                                                        icon = (ImageIcon) GetIcon.getBigIcon(path);
                                                        icon = changeIcon(icon, iconSideLength, iconSideLength);
                                                        label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                        label2.setIcon(icon);
                                                    } else {
                                                        label2.setIcon(null);
                                                        label2.setText("��Ч�ļ�");
                                                        search.addToRecycleBin(path);
                                                    }
                                                    path = listResult.get(labelCount + 2);
                                                    name = getFileName(listResult.get(labelCount + 2));


                                                    if (isDirectory(path) || isFile(path)) {
                                                        icon = (ImageIcon) GetIcon.getBigIcon(path);
                                                        icon = changeIcon(icon, iconSideLength, iconSideLength);
                                                        label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                        label3.setIcon(icon);
                                                    } else {
                                                        label3.setIcon(null);
                                                        label3.setText("��Ч�ļ�");
                                                        search.addToRecycleBin(path);
                                                    }
                                                    path = listResult.get(labelCount + 3);
                                                    name = getFileName(listResult.get(labelCount + 3));


                                                    if (isDirectory(path) || isFile(path)) {
                                                        icon = (ImageIcon) GetIcon.getBigIcon(path);
                                                        icon = changeIcon(icon, iconSideLength, iconSideLength);
                                                        label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                        label4.setIcon(icon);
                                                    } else {
                                                        label4.setIcon(null);
                                                        label4.setText("��Ч�ļ�");
                                                        search.addToRecycleBin(path);
                                                    }
                                                } catch (ArrayIndexOutOfBoundsException ignored) {

                                                }
                                                break;
                                            case 1:
                                                try {
                                                    if (!label1.getText().isEmpty()) {
                                                        label1.setBackground(labelColor);
                                                    }
                                                    if (!label2.getText().isEmpty()) {
                                                        label2.setBackground(backgroundColor2);
                                                    }
                                                    if (!label3.getText().isEmpty()) {
                                                        label3.setBackground(backgroundColor3);
                                                    }
                                                    if (!label4.getText().isEmpty()) {
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                } catch (NullPointerException ignored) {

                                                }
                                                break;
                                            case 2:
                                                try {
                                                    if (!label1.getText().isEmpty()) {
                                                        label1.setBackground(backgroundColor1);
                                                    }
                                                    if (!label2.getText().isEmpty()) {
                                                        label2.setBackground(labelColor);
                                                    }
                                                    if (!label3.getText().isEmpty()) {
                                                        label3.setBackground(backgroundColor3);
                                                    }
                                                    if (!label4.getText().isEmpty()) {
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                } catch (NullPointerException ignored) {

                                                }
                                                break;
                                            case 3:
                                                try {
                                                    if (!label1.getText().isEmpty()) {
                                                        label1.setBackground(backgroundColor1);
                                                    }
                                                    if (!label2.getText().isEmpty()) {
                                                        label2.setBackground(backgroundColor2);
                                                    }
                                                    if (!label3.getText().isEmpty()) {
                                                        label3.setBackground(labelColor);
                                                    }
                                                    if (!label4.getText().isEmpty()) {
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                } catch (NullPointerException ignored) {

                                                }
                                                break;
                                        }
                                    } else {
                                        switch (position) {
                                            case 0:
                                                //���������϶ˣ�ˢ����ʾ
                                                try {
                                                    String command = listResult.get(labelCount);
                                                    String[] info = semicolon.split(command);
                                                    String path = info[1];
                                                    String name = info[0];
                                                    ImageIcon imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                                    imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                                    label1.setIcon(imageIcon);
                                                    label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                                    command = listResult.get(labelCount + 1);
                                                    info = semicolon.split(command);
                                                    path = info[1];
                                                    name = info[0];
                                                    imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                                    imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                                    label2.setIcon(imageIcon);
                                                    label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                                    command = listResult.get(labelCount + 2);
                                                    info = semicolon.split(command);
                                                    path = info[1];
                                                    name = info[0];
                                                    imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                                    imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                                    label3.setIcon(imageIcon);
                                                    label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                                    command = listResult.get(labelCount + 3);
                                                    info = semicolon.split(command);
                                                    path = info[1];
                                                    name = info[0];
                                                    imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                                    imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                                    label4.setIcon(imageIcon);
                                                    label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");
                                                } catch (ArrayIndexOutOfBoundsException ignored) {

                                                }
                                                break;
                                            case 1:
                                                try {
                                                    if (!label1.getText().isEmpty()) {
                                                        label1.setBackground(labelColor);
                                                    }
                                                    if (!label2.getText().isEmpty()) {
                                                        label2.setBackground(backgroundColor2);
                                                    }
                                                    if (!label3.getText().isEmpty()) {
                                                        label3.setBackground(backgroundColor3);
                                                    }
                                                    if (!label4.getText().isEmpty()) {
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                } catch (NullPointerException ignored) {

                                                }
                                                break;
                                            case 2:
                                                try {
                                                    if (!label1.getText().isEmpty()) {
                                                        label1.setBackground(backgroundColor1);
                                                    }
                                                    if (!label2.getText().isEmpty()) {
                                                        label2.setBackground(labelColor);
                                                    }
                                                    if (!label3.getText().isEmpty()) {
                                                        label3.setBackground(backgroundColor3);
                                                    }
                                                    if (!label4.getText().isEmpty()) {
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                } catch (NullPointerException ignored) {

                                                }
                                                break;
                                            case 3:
                                                try {
                                                    if (!label1.getText().isEmpty()) {
                                                        label1.setBackground(backgroundColor1);
                                                    }
                                                    if (!label2.getText().isEmpty()) {
                                                        label2.setBackground(backgroundColor2);
                                                    }
                                                    if (!label3.getText().isEmpty()) {
                                                        label3.setBackground(labelColor);
                                                    }
                                                    if (!label4.getText().isEmpty()) {
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                } catch (NullPointerException ignored) {

                                                }
                                                break;
                                            case 4:
                                                try {
                                                    if (!label1.getText().isEmpty()) {
                                                        label1.setBackground(backgroundColor1);
                                                    }
                                                    if (!label2.getText().isEmpty()) {
                                                        label2.setBackground(backgroundColor2);
                                                    }
                                                    if (!label3.getText().isEmpty()) {
                                                        label3.setBackground(backgroundColor3);
                                                    }
                                                    if (!label4.getText().isEmpty()) {
                                                        label4.setBackground(labelColor);
                                                    }
                                                } catch (NullPointerException ignored) {

                                                }
                                                break;
                                        }
                                    }
                                } catch (NullPointerException ignored) {

                                }

                            }
                            if (labelCount < 0) {
                                labelCount = 0;
                            }
                        }
                    } else if (40 == key) {
                        //�¼������
                        if (isFirstPress || System.currentTimeMillis() - pressTime > timeLimit) {
                            pressTime = System.currentTimeMillis();
                            isFirstPress = false;
                            try {
                                if (!label1.getText().isEmpty() && !label2.getText().isEmpty() && !label3.getText().isEmpty() && !label4.getText().isEmpty()) {
                                    isUserPressed = true;
                                }
                            } catch (NullPointerException ignored) {

                            }
                            boolean isNextExist = false;
                            if (labelCount == 0) {
                                try {
                                    if (!label2.getText().isEmpty()) {
                                        isNextExist = true;
                                    }
                                } catch (NullPointerException ignored) {

                                }
                            } else if (labelCount == 1) {
                                try {
                                    if (!label3.getText().isEmpty()) {
                                        isNextExist = true;
                                    }
                                } catch (NullPointerException ignored) {

                                }
                            } else if (labelCount == 2) {
                                try {
                                    if (!label4.getText().isEmpty()) {
                                        isNextExist = true;
                                    }
                                } catch (NullPointerException ignored) {

                                }
                            } else {
                                isNextExist = true;
                            }
                            if (isNextExist) {
                                if (!textField.getText().isEmpty()) {
                                    labelCount++;
                                    if (labelCount < 0) {
                                        labelCount = 0;
                                    }

                                    //System.out.println(labelCount);
                                    if (labelCount >= listResult.size()) {
                                        labelCount = listResult.size() - 1;
                                    }
                                    //�ж���ǰѡ��λ��
                                    int position;
                                    try {
                                        if (label1.getBackground() == labelColor) {
                                            position = 0;
                                        } else if (label2.getBackground() == labelColor) {
                                            position = 1;
                                        } else if (label3.getBackground() == labelColor) {
                                            position = 2;
                                        } else {
                                            position = 3;
                                        }
                                        if (!isCommandMode) {
                                            switch (position) {
                                                case 0:
                                                    int size = listResult.size();
                                                    if (size == 2) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(labelColor);
                                                    } else if (size == 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(labelColor);
                                                        label3.setBackground(backgroundColor3);
                                                    } else if (size > 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(labelColor);
                                                        label3.setBackground(backgroundColor3);
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                    break;
                                                case 1:
                                                    size = listResult.size();
                                                    if (size == 2) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                    } else if (size == 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                        label3.setBackground(labelColor);
                                                    } else if (size > 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                        label3.setBackground(labelColor);
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                    break;
                                                case 2:
                                                    size = listResult.size();
                                                    if (size == 2) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                    } else if (size == 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                        label3.setBackground(backgroundColor3);
                                                    } else if (size > 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                        label3.setBackground(backgroundColor3);
                                                        label4.setBackground(labelColor);
                                                    }
                                                    break;
                                                case 3:
                                                    //�������¶ˣ�ˢ����ʾ
                                                    try {
                                                        String path = listResult.get(labelCount - 3);
                                                        String name = getFileName(listResult.get(labelCount - 3));
                                                        ImageIcon icon;
                                                        if (isDirectory(path) || isFile(path)) {
                                                            icon = (ImageIcon) GetIcon.getBigIcon(path);
                                                            icon = changeIcon(icon, iconSideLength, iconSideLength);
                                                            label1.setIcon(icon);
                                                            label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                        } else {
                                                            label1.setIcon(null);
                                                            label1.setText("��Ч�ļ�");
                                                            search.addToRecycleBin(path);
                                                        }
                                                        path = listResult.get(labelCount - 2);
                                                        name = getFileName(listResult.get(labelCount - 2));

                                                        if (isDirectory(path) || isFile(path)) {
                                                            icon = (ImageIcon) GetIcon.getBigIcon(path);
                                                            icon = changeIcon(icon, iconSideLength, iconSideLength);
                                                            label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                            label2.setIcon(icon);
                                                        } else {
                                                            label2.setIcon(null);
                                                            label2.setText("��Ч�ļ�");
                                                            search.addToRecycleBin(path);
                                                        }
                                                        path = listResult.get(labelCount - 1);
                                                        name = getFileName(listResult.get(labelCount - 1));


                                                        if (isDirectory(path) || isFile(path)) {
                                                            icon = (ImageIcon) GetIcon.getBigIcon(path);
                                                            icon = changeIcon(icon, iconSideLength, iconSideLength);
                                                            label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                            label3.setIcon(icon);
                                                        } else {
                                                            label3.setIcon(null);
                                                            label3.setText("��Ч�ļ�");
                                                            search.addToRecycleBin(path);
                                                        }
                                                        path = listResult.get(labelCount);
                                                        name = getFileName(listResult.get(labelCount));


                                                        if (isDirectory(path) || isFile(path)) {
                                                            icon = (ImageIcon) GetIcon.getBigIcon(path);
                                                            icon = changeIcon(icon, iconSideLength, iconSideLength);
                                                            label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                                                            label4.setIcon(icon);
                                                        } else {
                                                            label4.setIcon(null);
                                                            label4.setText("��Ч�ļ�");
                                                            search.addToRecycleBin(path);
                                                        }
                                                    } catch (ArrayIndexOutOfBoundsException ignored) {

                                                    }
                                                    break;
                                            }
                                        } else {
                                            switch (position) {
                                                case 0:
                                                    int size = listResult.size();
                                                    if (size == 2) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(labelColor);
                                                    } else if (size == 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(labelColor);
                                                        label3.setBackground(backgroundColor3);
                                                    } else if (size > 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(labelColor);
                                                        label3.setBackground(backgroundColor3);
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                    break;
                                                case 1:
                                                    size = listResult.size();
                                                    if (size == 2) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                    } else if (size == 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                        label3.setBackground(labelColor);
                                                    } else if (size > 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                        label3.setBackground(labelColor);
                                                        label4.setBackground(backgroundColor4);
                                                    }
                                                    break;
                                                case 2:
                                                    size = listResult.size();
                                                    if (size == 2) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                    } else if (size == 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                        label3.setBackground(backgroundColor3);
                                                    } else if (size > 3) {
                                                        label1.setBackground(backgroundColor1);
                                                        label2.setBackground(backgroundColor2);
                                                        label3.setBackground(backgroundColor3);
                                                        label4.setBackground(labelColor);
                                                    }
                                                    break;
                                                case 3:
                                                    //���������¶ˣ�ˢ����ʾ
                                                    try {
                                                        String command = listResult.get(labelCount - 3);
                                                        String[] info = semicolon.split(command);
                                                        String path = info[1];
                                                        String name = info[0];
                                                        ImageIcon imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                                        imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                                        label1.setIcon(imageIcon);
                                                        label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                                        command = listResult.get(labelCount - 2);
                                                        info = semicolon.split(command);
                                                        path = info[1];
                                                        name = info[0];
                                                        imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                                        imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                                        label2.setIcon(imageIcon);
                                                        label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                                        command = listResult.get(labelCount - 1);
                                                        info = semicolon.split(command);
                                                        path = info[1];
                                                        name = info[0];
                                                        imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                                        imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                                        label3.setIcon(imageIcon);
                                                        label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");

                                                        command = listResult.get(labelCount);
                                                        info = semicolon.split(command);
                                                        path = info[1];
                                                        name = info[0];
                                                        imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                                                        imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                                                        label4.setIcon(imageIcon);
                                                        label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");
                                                    } catch (ArrayIndexOutOfBoundsException ignored) {

                                                    }
                                                    break;
                                            }
                                        }
                                    } catch (NullPointerException ignored) {

                                    }
                                }
                            }
                        }
                    } else if (10 == key) {
                        //enter�����
                        closedTodo();
                        if (!isCommandMode) {
                            if (isOpenLastFolderPressed) {
                                //���ϼ��ļ���
                                File open = new File(listResult.get(labelCount));
                                try {
                                    Process p = Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                                    p.getOutputStream().close();
                                    p.getErrorStream().close();
                                    p.getInputStream().close();
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            } else if (SettingsFrame.isDefaultAdmin || isRunAsAdminPressed) {
                                openWithAdmin(listResult.get(labelCount));
                            } else if (isCopyPathPressed) {
                                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                                Transferable trans = new StringSelection(listResult.get(labelCount));
                                clipboard.setContents(trans, null);
                            } else {
                                String openFile = listResult.get(labelCount);
                                if (openFile.endsWith(".bat") || openFile.endsWith(".cmd")) {
                                    openWithAdmin(openFile);
                                } else {
                                    openWithoutAdmin(openFile);
                                }
                            }
                            saveCache(listResult.get(labelCount) + ';');
                        } else {
                            //ֱ�Ӵ�
                            String command = listResult.get(labelCount);
                            if (Desktop.isDesktopSupported()) {
                                Desktop desktop = Desktop.getDesktop();
                                try {
                                    desktop.open(new File(semicolon.split(command)[1]));
                                } catch (Exception e) {
                                    JOptionPane.showMessageDialog(null, "ִ��ʧ��");
                                }
                            }
                        }
                    } else if (SettingsFrame.openLastFolderKeyCode == key) {
                        //���ϼ��ļ����ȼ������
                        isOpenLastFolderPressed = true;
                    } else if (SettingsFrame.runAsAdminKeyCode == key) {
                        //�Թ���Ա��ʽ�����ȼ������
                        isRunAsAdminPressed = true;
                    } else if (SettingsFrame.copyPathKeyCode == key) {
                        isCopyPathPressed = true;
                    }
                }
            }

            @Override
            public void keyReleased(KeyEvent arg0) {
                panel.repaint();
                int key = arg0.getKeyCode();
                if (SettingsFrame.openLastFolderKeyCode == key) {
                    //��λ����״̬
                    isOpenLastFolderPressed = false;
                } else if (SettingsFrame.runAsAdminKeyCode == key) {
                    isRunAsAdminPressed = false;
                } else if (SettingsFrame.copyPathKeyCode == key) {
                    isCopyPathPressed = false;
                }
            }

            @Override
            public void keyTyped(KeyEvent arg0) {
                panel.repaint();
            }
        });
    }

    public static SearchBar getInstance() {
        return searchBarInstance;
    }

    public boolean isUsing() {
        //�����Ƿ�������ʾ
        return this.isUsing;
    }

    private boolean isExist(String path) {
        File f = new File(path);
        return f.exists();
    }

    private void addResult(ConcurrentLinkedQueue<String> paths, String searchText, long time, String searchCase) {
        //Ϊlabel��ӽ��
        ExecutorService threadPool = Executors.newFixedThreadPool(4);
        for (int i = 0; i < 4; i++) {
            threadPool.execute(() -> {
                String each;
                String path;
                try {
                    while ((path = paths.poll()) != null) {
                        ReaderInfo readerInfo = readerMap.get(path);
                        if (readerInfo != null) {
                            BufferedReader reader = readerInfo.reader;
                            labelOut:
                            while ((each = reader.readLine()) != null) {
                                if (startTime > time) { //�û�������������Ϣ
                                    break;
                                }
                                if (search.isUsable()) {
                                    if (isMatched(getFileName(each), searchText)) {
                                        switch (searchCase) {
                                            case "f":
                                                if (isFile(each)) {
                                                    if (!listResult.contains(each)) {
                                                        if (isExist(each)) {
                                                            listResult.add(each);
                                                            if (listResult.size() > 50) {
                                                                break labelOut;
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            case "d":
                                                if (isDirectory(each)) {
                                                    if (!listResult.contains(each)) {
                                                        if (isExist(each)) {
                                                            listResult.add(each);
                                                            if (listResult.size() > 50) {
                                                                break labelOut;
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            case "full":
                                                if (getFileName(each).toLowerCase().equals(searchText.toLowerCase())) {
                                                    if (!listResult.contains(each)) {
                                                        if (isExist(each)) {
                                                            listResult.add(each);
                                                            if (listResult.size() > 50) {
                                                                break labelOut;
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            case "dfull":
                                                if (getFileName(each).toLowerCase().equals(searchText.toLowerCase())) {
                                                    if (isDirectory(each)) {
                                                        if (!listResult.contains(each)) {
                                                            if (isExist(each)) {
                                                                listResult.add(each);
                                                                if (listResult.size() > 50) {
                                                                    break labelOut;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            case "ffull":
                                                if (getFileName(each).toLowerCase().equals(searchText.toLowerCase())) {
                                                    if (isFile(each)) {
                                                        if (!listResult.contains(each)) {
                                                            if (isExist(each)) {
                                                                listResult.add(each);
                                                                if (listResult.size() > 50) {
                                                                    break labelOut;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            default:
                                                if (!listResult.contains(each)) {
                                                    if (isExist(each)) {
                                                        listResult.add(each);
                                                        if (listResult.size() > 50) {
                                                            break labelOut;
                                                        }
                                                    }
                                                }
                                                break;
                                        }
                                    }
                                }
                            }
                            reader.close();
                            reader = new BufferedReader(new FileReader(path));
                            readerInfo = new ReaderInfo(System.currentTimeMillis(), reader);
                            readerMap.put(path, readerInfo);
                        } else {
                            BufferedReader reader = new BufferedReader(new FileReader(path));
                            labelOut:
                            while ((each = reader.readLine()) != null) {
                                if (startTime > time) { //�û�������������Ϣ
                                    break;
                                }
                                if (search.isUsable()) {
                                    if (isMatched(getFileName(each), searchText)) {
                                        switch (searchCase) {
                                            case "f":
                                                if (isFile(each)) {
                                                    if (!listResult.contains(each)) {
                                                        if (isExist(each)) {
                                                            listResult.add(each);
                                                            if (listResult.size() > 50) {
                                                                break labelOut;
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            case "d":
                                                if (isDirectory(each)) {
                                                    if (!listResult.contains(each)) {
                                                        if (isExist(each)) {
                                                            listResult.add(each);
                                                            if (listResult.size() > 50) {
                                                                break labelOut;
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            case "full":
                                                if (getFileName(each).toLowerCase().equals(searchText.toLowerCase())) {
                                                    if (!listResult.contains(each)) {
                                                        if (isExist(each)) {
                                                            listResult.add(each);
                                                            if (listResult.size() > 50) {
                                                                break labelOut;
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            case "dfull":
                                                if (getFileName(each).toLowerCase().equals(searchText.toLowerCase())) {
                                                    if (isDirectory(each)) {
                                                        if (!listResult.contains(each)) {
                                                            if (isExist(each)) {
                                                                listResult.add(each);
                                                                if (listResult.size() > 50) {
                                                                    break labelOut;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            case "ffull":
                                                if (getFileName(each).toLowerCase().equals(searchText.toLowerCase())) {
                                                    if (isFile(each)) {
                                                        if (!listResult.contains(each)) {
                                                            if (isExist(each)) {
                                                                listResult.add(each);
                                                                if (listResult.size() > 50) {
                                                                    break labelOut;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                break;
                                            default:
                                                if (!listResult.contains(each)) {
                                                    if (isExist(each)) {
                                                        listResult.add(each);
                                                        if (listResult.size() > 50) {
                                                            break labelOut;
                                                        }
                                                    }
                                                }
                                                break;
                                        }
                                    }
                                }
                            }
                            if (readerMap.size() < SettingsFrame.maxConnectionNum) {
                                reader = new BufferedReader(new FileReader(path));
                                ReaderInfo temp = new ReaderInfo(System.currentTimeMillis(), reader);
                                readerMap.put(path, temp);
                            } else {
                                reader.close();
                            }
                        }
                    }
                } catch (IOException ignored) {

                }
            });
        }
        threadPool.shutdown();
        try {
            threadPool.awaitTermination(Integer.MAX_VALUE, TimeUnit.SECONDS);
        } catch (InterruptedException ignored) {

        }
        if (!textField.getText().isEmpty()) {
            delRepeated();
            if (listResult.size() == 0) {
                label1.setText("�޽��");
                label1.setIcon(null);
            }
        }
    }

    public void showSearchbar() {
        if (!searchBar.isVisible()) {
            searchBar.setVisible(true);
            searchBar.requestFocusInWindow();
            searchBar.setAlwaysOnTop(true);
            textField.setCaretPosition(0);
            textField.requestFocusInWindow();
            isUsing = true;
            visibleStartTime = System.currentTimeMillis();
        }
    }

    private void showResults() {
        panel.repaint();
        if (!isCommandMode) {
            try {
                String path = listResult.get(0);
                String name = getFileName(listResult.get(0));
                ImageIcon icon;
                if (isDirectory(path) || isFile(path)) {
                    icon = (ImageIcon) GetIcon.getBigIcon(path);
                    icon = changeIcon(icon, iconSideLength, iconSideLength);
                    label1.setIcon(icon);
                    label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                    if (labelCount == 0) {
                        label1.setBackground(labelColor);
                    } else {
                        label1.setBackground(backgroundColor1);
                    }
                } else {
                    label1.setIcon(null);
                    label1.setText("��Ч�ļ�");
                    if (labelCount == 0) {
                        label1.setBackground(labelColor);
                    } else {
                        label1.setBackground(backgroundColor1);
                    }
                    search.addToRecycleBin(path);
                }
                path = listResult.get(1);
                name = getFileName(listResult.get(1));


                if (isDirectory(path) || isFile(path)) {
                    icon = (ImageIcon) GetIcon.getBigIcon(path);
                    icon = changeIcon(icon, iconSideLength, iconSideLength);
                    label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                    label2.setIcon(icon);
                    if (labelCount == 1) {
                        label2.setBackground(labelColor);
                    } else {
                        label2.setBackground(backgroundColor2);
                    }
                } else {
                    label2.setIcon(null);
                    label2.setText("��Ч�ļ�");
                    if (labelCount == 1) {
                        label2.setBackground(labelColor);
                    } else {
                        label2.setBackground(backgroundColor2);
                    }
                    search.addToRecycleBin(path);
                }
                path = listResult.get(2);
                name = getFileName(listResult.get(2));


                if (isDirectory(path) || isFile(path)) {
                    icon = (ImageIcon) GetIcon.getBigIcon(path);
                    icon = changeIcon(icon, iconSideLength, iconSideLength);
                    label3.setIcon(icon);
                    label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                    if (labelCount == 2) {
                        label3.setBackground(labelColor);
                    } else {
                        label3.setBackground(backgroundColor3);
                    }
                } else {
                    label3.setIcon(null);
                    label3.setText("��Ч�ļ�");
                    if (labelCount == 2) {
                        label3.setBackground(labelColor);
                    } else {
                        label3.setBackground(backgroundColor3);
                    }
                    search.addToRecycleBin(path);
                }
                path = listResult.get(3);
                name = getFileName(listResult.get(3));


                if (isDirectory(path) || isFile(path)) {
                    icon = (ImageIcon) GetIcon.getBigIcon(path);
                    icon = changeIcon(icon, iconSideLength, iconSideLength);
                    label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + getParentPath(path) + "</body></html>");
                    label4.setIcon(icon);
                    if (labelCount >= 3) {
                        label4.setBackground(labelColor);
                    } else {
                        label4.setBackground(backgroundColor4);
                    }
                } else {
                    label4.setIcon(null);
                    label4.setText("��Ч�ļ�");
                    if (labelCount >= 3) {
                        label4.setBackground(labelColor);
                    } else {
                        label4.setBackground(backgroundColor4);
                    }
                    search.addToRecycleBin(path);
                }
            } catch (java.lang.IndexOutOfBoundsException ignored) {

            }
        } else {
            try {
                String command = listResult.get(0);
                String[] info = semicolon.split(command);
                String path = info[1];
                String name = info[0];
                ImageIcon imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                label1.setIcon(imageIcon);
                label1.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");
                if (labelCount == 0) {
                    label1.setBackground(labelColor);
                } else {
                    label1.setBackground(backgroundColor1);
                }

                command = listResult.get(1);
                info = semicolon.split(command);
                path = info[1];
                name = info[0];
                imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                label2.setIcon(imageIcon);
                label2.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");
                if (labelCount == 1) {
                    label2.setBackground(labelColor);
                } else {
                    label2.setBackground(backgroundColor2);
                }

                command = listResult.get(2);
                info = semicolon.split(command);
                path = info[1];
                name = info[0];
                imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                label3.setIcon(imageIcon);
                label3.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");
                if (labelCount == 2) {
                    label3.setBackground(labelColor);
                } else {
                    label3.setBackground(backgroundColor3);
                }

                command = listResult.get(3);
                info = semicolon.split(command);
                path = info[1];
                name = info[0];
                imageIcon = (ImageIcon) GetIcon.getBigIcon(path);
                imageIcon = changeIcon(imageIcon, iconSideLength, iconSideLength);
                label4.setIcon(imageIcon);
                label4.setText("<html><body>" + name + "<br><font size=\"-1\">" + ">>>" + path + "</font></body></html>");
                if (labelCount >= 3) {
                    label4.setBackground(labelColor);
                } else {
                    label4.setBackground(backgroundColor4);
                }
            } catch (IndexOutOfBoundsException ignored) {

            }
        }
    }


    private void clearLabel() {
        label4.setBorder(null);
        label3.setBorder(null);
        label2.setBorder(null);
        label1.setBorder(null);
        label4.setIcon(null);
        label3.setIcon(null);
        label2.setIcon(null);
        label1.setIcon(null);
        label4.setText(null);
        label3.setText(null);
        label2.setText(null);
        label1.setText(null);
        label4.setBackground(null);
        label3.setBackground(null);
        label2.setBackground(null);
        label1.setBackground(null);
    }

    private void openWithAdmin(String path) {
        searchBar.setVisible(false);
        File name = new File(path);
        if (name.exists()) {
            try {
                String command = name.getAbsolutePath();
                String start = "cmd /c start " + command.substring(0, 2);
                String end = "\"" + command.substring(2) + "\"";
                Process p = Runtime.getRuntime().exec(start + end, null, name.getParentFile());
                p.getInputStream().close();
                p.getOutputStream().close();
                p.getErrorStream().close();
            } catch (IOException e) {
                //���ϼ��ļ���
                try {
                    Runtime.getRuntime().exec("explorer.exe /select, \"" + name.getAbsolutePath() + "\"");
                } catch (IOException ignored) {

                }
            }
        }
    }

    private void openWithoutAdmin(String path) {
        searchBar.setVisible(false);
        if (isExist(path)) {
            try {
                if (path.toLowerCase().endsWith(".lnk")) {
                    String command = "cmd /c explorer.exe " + "\"" + path + "\"";
                    Process p = Runtime.getRuntime().exec(command);
                    p.getOutputStream().close();
                    p.getErrorStream().close();
                    p.getInputStream().close();
                } else if (path.toLowerCase().endsWith(".url")) {
                    Desktop desktop;
                    if (Desktop.isDesktopSupported()) {
                        desktop = Desktop.getDesktop();
                        desktop.open(new File(path));
                    }
                } else {
                    //������ݷ�ʽ����ʱ�ļ��У��򿪺�ɾ��
                    createShortCut(path, SettingsFrame.tmp.getAbsolutePath() + "\\open");
                    Process p = Runtime.getRuntime().exec("cmd /c explorer.exe " + "\"" + SettingsFrame.tmp.getAbsolutePath() + "\\open.lnk" + "\"");
                    p.getInputStream().close();
                    p.getOutputStream().close();
                    p.getErrorStream().close();
                }
            } catch (Exception e) {
                //���ϼ��ļ���
                try {
                    Process p = Runtime.getRuntime().exec("explorer.exe /select, \"" + path + "\"");
                    p.getOutputStream().close();
                    p.getErrorStream().close();
                    p.getInputStream().close();
                } catch (IOException ignored) {

                }
            }
        }
    }

    private void createShortCut(String fileOrFolderPath, String writeShortCutPath) throws Exception {
        File shortcutGen = new File("user/shortcutGenerator.vbs");
        String shortcutGenPath = shortcutGen.getAbsolutePath();
        String start = "cmd /c start " + shortcutGenPath.substring(0, 2);
        String end = "\"" + shortcutGenPath.substring(2) + "\"";
        String commandToGenLnk = start + end + " /target:" + "\"" + fileOrFolderPath + "\"" + " " + "/shortcut:" + "\"" + writeShortCutPath + "\"" + " /workingdir:" + "\"" + fileOrFolderPath.substring(0, fileOrFolderPath.lastIndexOf("\\")) + "\"";
        Process p = Runtime.getRuntime().exec("cmd /c " + commandToGenLnk);
        p.getInputStream().close();
        p.getOutputStream().close();
        p.getErrorStream().close();
        while (p.isAlive()) {
            Thread.sleep(1);
        }
    }

    public String getFileName(String path) {
        File file = new File(path);
        return file.getName();
    }

    private ImageIcon changeIcon(ImageIcon icon, int width, int height) {
        try {
            Image image = icon.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT);
            return new ImageIcon(image);
        } catch (NullPointerException e) {
            return null;
        }
    }

    public int getAscIISum(String path) {
        path = path.toUpperCase();
        if (path.contains(";")) {
            path = path.replace(";", "");
        }
        return GetAscII.INSTANCE.getAscII(path);
    }

    private void saveCache(String content) {
        int cacheNum = 0;
        File cache = new File("user/cache.dat");
        StringBuilder oldCaches = new StringBuilder();
        boolean isRepeated;
        if (cache.exists()) {
            try (BufferedReader reader = new BufferedReader(new FileReader(cache))) {
                String eachLine;
                while ((eachLine = reader.readLine()) != null) {
                    oldCaches.append(eachLine);
                    cacheNum++;
                }
            } catch (IOException ignored) {

            }
        }
        if (cacheNum < SettingsFrame.cacheNumLimit) {
            isRepeated = isMatched(oldCaches.toString() + ";", (content));
            if (!isRepeated) {
                try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File("user/cache.dat"), true)))) {
                    out.write(content + "\r\n");
                } catch (Exception ignored) {

                }
            }
        }
    }

    private void delCacheRepeated() {
        File cacheFile = new File("user/cache.dat");
        HashSet<String> set = new HashSet<>();
        StringBuilder allCaches = new StringBuilder();
        String eachLine;
        if (cacheFile.exists()) {
            try (BufferedReader br = new BufferedReader(new FileReader(cacheFile))) {
                while ((eachLine = br.readLine()) != null) {
                    String[] each = semicolon.split(eachLine);
                    Collections.addAll(set, each);
                }
            } catch (IOException ignored) {

            }
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(cacheFile))) {
                for (String cache : set) {
                    allCaches.append(cache).append(";\n");
                }
                bw.write(allCaches.toString());
            } catch (IOException ignored) {

            }
        }
    }

    private void delCache(ArrayList<String> cache) {
        File cacheFile = new File("user/cache.dat");
        StringBuilder allCaches = new StringBuilder();
        String eachLine;
        if (cacheFile.exists()) {
            try (BufferedReader br = new BufferedReader(new FileReader(cacheFile))) {
                while ((eachLine = br.readLine()) != null) {
                    String[] each = semicolon.split(eachLine);
                    for (String eachCache : each) {
                        if (!(cache.contains(eachCache))) {
                            allCaches.append(eachCache).append(";\n");
                        }
                    }
                }
            } catch (IOException ignored) {

            }
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(cacheFile))) {
                bw.write(allCaches.toString());
            } catch (IOException ignored) {

            }
        }
    }

    private void searchCache(String text, String searchCase) {
        String cacheResult;
        boolean isCacheRepeated = false;
        ArrayList<String> cachesToDel = new ArrayList<>();
        File cache = new File("user/cache.dat");
        if (cache.exists()) {
            try (BufferedReader reader = new BufferedReader(new FileReader(cache))) {
                while ((cacheResult = reader.readLine()) != null) {
                    String[] caches = semicolon.split(cacheResult);
                    for (String eachCache : caches) {
                        if (!(new File(eachCache).exists())) {
                            cachesToDel.add(eachCache);
                        } else {
                            String eachCacheName = getFileName(eachCache);
                            if (isMatched(eachCacheName, text)) {
                                if (!listResult.contains(eachCache)) {
                                    boolean fullMatched = eachCacheName.toLowerCase().equals(text.toLowerCase());
                                    switch (searchCase) {
                                        case "f":
                                            if (isFile(eachCache)) {
                                                listResult.add(eachCache);
                                            }
                                            break;
                                        case "d":
                                            if (isDirectory(eachCache)) {
                                                listResult.add(eachCache);
                                            }
                                            break;
                                        case "full":
                                            if (fullMatched) {
                                                listResult.add(eachCache);
                                            }
                                            break;
                                        case "dfull":
                                            if (fullMatched && isDirectory(eachCache)) {
                                                listResult.add(eachCache);
                                            }
                                            break;
                                        case "ffull":
                                            if (fullMatched && isFile(eachCache)) {
                                                listResult.add(eachCache);
                                            }
                                            break;
                                        default:
                                            listResult.add(eachCache);
                                    }
                                    if (listResult.size() > 50) {
                                        break;
                                    }
                                } else {
                                    isCacheRepeated = true;
                                }
                            }
                        }
                    }
                }
            } catch (IOException ignored) {

            }
        }
        delCache(cachesToDel);
        if (isCacheRepeated) {
            delCacheRepeated();
        }
    }

    private boolean isMatched(String srcText, String txt) {
        if (!txt.isEmpty()) {
            srcText = srcText.toLowerCase();
            txt = txt.toLowerCase();
            String[] keyWords = semicolon.split(txt);
            for (String each : keyWords) {
                if (!srcText.contains(each)) {
                    return false;
                }
            }
        }
        return true;
    }

    private void delRepeated() {
        LinkedHashSet<String> set = new LinkedHashSet<>(listResult);
        listResult.clear();
        listResult.addAll(set);
    }

    private void searchPriorityFolder(String text, String searchCase) {
        File path = new File(SettingsFrame.priorityFolder);
        boolean exist = path.exists();
        LinkedList<File> listRemain = new LinkedList<>();
        if (exist) {
            File[] files = path.listFiles();
            if (!(null == files || files.length == 0)) {
                for (File each : files) {
                    if (isMatched(getFileName(each.getAbsolutePath()), text)) {
                        switch (searchCase) {
                            case "f":
                                if (each.isFile()) {
                                    listResult.add(0, each.getAbsolutePath());
                                }
                                break;
                            case "d":
                                if (each.isDirectory()) {
                                    listResult.add(0, each.getAbsolutePath());
                                }
                                break;
                            case "full":
                                if (each.getName().equals(text)) {
                                    listResult.add(0, each.getAbsolutePath());
                                }
                                break;
                            case "dfull":
                                if (each.getName().equals(text) && each.isDirectory()) {
                                    listResult.add(0, each.getAbsolutePath());
                                }
                                break;
                            case "ffull":
                                if (each.getName().equals(text) && each.isFile()) {
                                    listResult.add(0, each.getAbsolutePath());
                                }
                                break;
                            default:
                                listResult.add(0, each.getAbsolutePath());
                        }
                    }
                    if (each.isDirectory()) {
                        listRemain.add(each);
                    }
                }
                while (!listRemain.isEmpty()) {
                    File remain = listRemain.pop();
                    File[] allFiles = remain.listFiles();
                    assert allFiles != null;
                    for (File each : allFiles) {
                        if (isMatched(getFileName(each.getAbsolutePath()), text)) {
                            switch (searchCase) {
                                case "F":
                                    if (each.isFile()) {
                                        listResult.add(0, each.getAbsolutePath());
                                    }
                                    break;
                                case "D":
                                    if (each.isDirectory()) {
                                        listResult.add(0, each.getAbsolutePath());
                                    }
                                    break;
                                case "FULL":
                                    if (each.getName().toLowerCase().equals(text.toLowerCase())) {
                                        listResult.add(0, each.getAbsolutePath());
                                    }
                                    break;
                                case "DFULL":
                                    if (each.getName().toLowerCase().equals(text.toLowerCase()) && each.isDirectory()) {
                                        listResult.add(0, each.getAbsolutePath());
                                    }
                                    break;
                                case "FFULL":
                                    if (each.getName().toLowerCase().equals(text.toLowerCase()) && each.isFile()) {
                                        listResult.add(0, each.getAbsolutePath());
                                    }
                                    break;
                            }
                        }
                        if (each.isDirectory()) {
                            listRemain.add(each);
                        }
                    }
                }
            }
        }
    }

    public void setTransparency(float trans) {
        searchBar.setOpacity(trans);
    }

    private void clearTextFieldText() {
        Runnable clear = () -> textField.setText(null);
        SwingUtilities.invokeLater(clear);
    }

    public void closedTodo() {
        Runnable todo = () -> {
            if (searchBar.isVisible()) {
                searchBar.setVisible(false);
            }
            clearLabel();
            startTime = System.currentTimeMillis();//��������
            isUsing = false;
            labelCount = 0;
            listResult.clear();
            textField.setText(null);
            isOpenLastFolderPressed = false;
            isRunAsAdminPressed = false;
            isCopyPathPressed = false;
            try {
                searchWaiter.interrupt();
            } catch (NullPointerException ignored) {

            }
        };
        SwingUtilities.invokeLater(todo);
    }

    public boolean isVisible() {
        return searchBar.isVisible();
    }

    private String getParentPath(String path) {
        File f = new File(path);
        return f.getParent();
    }

    private boolean isFile(String text) {
        File file = new File(text);
        return file.isFile();
    }

    private boolean isDirectory(String text) {
        File file = new File(text);
        return file.isDirectory();
    }

    public void setFontColorWithCoverage(int colorNum) {
        fontColorWithCoverage = new Color(colorNum);
    }

    public void setBackgroundColor1(int colorNum) {
        backgroundColor1 = new Color(colorNum);
    }

    public void setBackgroundColor2(int colorNum) {
        backgroundColor2 = new Color(colorNum);
    }

    public void setBackgroundColor3(int colorNum) {
        backgroundColor3 = new Color(colorNum);
    }

    public void setBackgroundColor4(int colorNum) {
        backgroundColor4 = new Color(colorNum);
    }

    public void setLabelColor(int colorNum) {
        labelColor = new Color(colorNum);
    }

    public void setFontColor(int colorNum) {
        fontColor = new Color(colorNum);
    }

    public void setSearchBarColor(int colorNum) {
        textField.setBackground(new Color(colorNum));
    }

    public int currentConnectionNum() {
        return readerMap.size();
    }

    public void closeAllConnection() {
        try {
            for (String eachKey : readerMap.keySet()) {
                ReaderInfo tmp = readerMap.get(eachKey);
                tmp.reader.close();
                readerMap.remove(eachKey, tmp);
            }
        } catch (IOException ignored) {

        }
    }
}

class ReaderInfo {
    public long time;
    public BufferedReader reader;

    public ReaderInfo(long _time, BufferedReader _reader) {
        this.time = _time;
        this.reader = _reader;
    }
}

