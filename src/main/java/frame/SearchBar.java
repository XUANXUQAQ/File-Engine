package frame;

import main.MainClass;
import search.Search;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.*;
import java.net.URL;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static main.MainClass.mainExit;


public class SearchBar {
    private static SearchBar searchBarInstance = new SearchBar();
    private JFrame searchBar = new JFrame();
    private Container panel;
    private LinkedList<String> listResult = new LinkedList<>();
    private JLabel label1 = new JLabel();
    private JLabel label2 = new JLabel();
    private JLabel label3 = new JLabel();
    private JLabel label4 = new JLabel();
    private boolean isCtrlPressed = false;
    private int labelCount = 0;
    private JTextField textField;
    private Search search = new Search();
    private Color labelColor = new Color(255, 75, 12, 255);
    private Color backgroundColor = new Color(108, 108, 108, 255);
    private Color backgroundColorLight = new Color(75, 75, 75, 255);
    private LinkedHashSet<String> list = new LinkedHashSet<>();
    private Thread thread;
    private boolean isFirstRun = true;
    private long startTime = 0;
    private boolean timer = false;
    private Thread searchWaiter = null;
    private boolean isUsing = false;

    private SearchBar() {
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
        int width = screenSize.width;
        int height = screenSize.height;
        int searchBarWidth = (int) (width * 0.5);
        int searchBarHeight = (int) (height * 0.5);
        int positionX = width / 2 - searchBarWidth / 2;
        int positionY = height / 2 - searchBarHeight / 2;


        //frame
        searchBar.setBounds(positionX, positionY, searchBarWidth, searchBarHeight);
        searchBar.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        searchBar.setUndecorated(true);
        searchBar.setBackground(null);
        searchBar.setOpacity(0.8f);
        panel = searchBar.getContentPane();


        //TextField
        textField = new JTextField(300);
        textField.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        Font textFieldFont = new Font("Microsoft JhengHei", Font.BOLD, (int) ((height * 0.1) / 96 * 72));
        textField.setFont(textFieldFont);
        textField.setForeground(Color.WHITE);
        textField.setHorizontalAlignment(JTextField.LEFT);
        textField.setBorder(null);
        textField.setBackground(backgroundColor);
        textField.setLocation(0, 0);


        ExecutorService fixedThreadPool = Executors.newFixedThreadPool(2);
        fixedThreadPool.execute(() -> {
            while (!mainExit) {
                if (!search.isManualUpdate() && !isUsing) {
                    if (search.getRecycleBinSize() > 100) {
                        System.out.println("已检测到回收站过大，自动清理");
                        search.setUsable(false);
                        search.mergeAndClearRecycleBin();
                        search.setUsable(true);
                    }
                    if (search.getLoadListSize() > 100) {
                        System.out.println("加载缓存空间过大，自动清理");
                        search.setUsable(false);
                        search.mergeFileToList();
                        search.setUsable(true);
                    }
                }
                try {
                    Thread.sleep(1);
                } catch (InterruptedException ignored) {

                }
            }
        });
        fixedThreadPool.execute(() -> {
            while (!mainExit) {
                long endTime = System.currentTimeMillis();
                if ((endTime - startTime > 500) && (timer)) {
                    labelCount = 0;
                    clearLabel();
                    if (!textField.getText().equals("")) {
                        label1.setBackground(labelColor);
                    }
                    listResult.clear();
                    String text = textField.getText();
                    if (search.isUsable()) {
                        if (text.equals("#*update*#")) {
                            MainClass.showMessage("提示", "正在更新文件索引");
                            clearTextFieldText();
                            search.setManualUpdate(true);
                            timer = false;
                            continue;
                        }
                        if (text.equals("#*version*#")) {
                            closedTodo();
                            JOptionPane.showMessageDialog(null, "当前版本：" + MainClass.version);
                        }
                        searchPriorityFolder(text);
                        searchCache(text);
                        char firstWord = '\0';
                        try {
                            firstWord = text.charAt(0);
                        } catch (Exception ignored) {

                        }
                        if ('>' == firstWord) {
                            if (isFirstRun || !thread.isAlive()) {
                                isFirstRun = false;
                            } else {
                                thread.interrupt();
                            }
                            thread = new Thread(new AddAllResults(text));
                            thread.start();
                        } else if (19968 <= (int) firstWord && (int) firstWord < 40869) {
                            list = (search.getListUnique());
                            addResult(list, text);
                        } else if ('%' == firstWord) {
                            list = (search.getListPercentSign());
                            addResult(list, text);
                        } else if ('_' == firstWord) {
                            list = (search.getListUnderline());
                            addResult(list, text);

                        } else if (Character.isDigit(firstWord)) {
                            list = (search.getListNum());
                            addResult(list, text);

                        } else if (Character.isAlphabetic(firstWord)) {
                            firstWord = Character.toUpperCase(firstWord);
                            if ('A' == firstWord) {
                                list = (search.getListA());
                                addResult(list, text);


                            } else if ('B' == firstWord) {
                                list = (search.getListB());

                                addResult(list, text);


                            } else if ('C' == firstWord) {
                                list = (search.getListC());

                                addResult(list, text);


                            } else if ('D' == firstWord) {
                                list = (search.getListD());

                                addResult(list, text);


                            } else if ('E' == firstWord) {
                                list = (search.getListE());
                                addResult(list, text);


                            } else if ('F' == firstWord) {
                                list = (search.getListF());

                                addResult(list, text);


                            } else if ('G' == firstWord) {
                                list = (search.getListG());

                                addResult(list, text);


                            } else if ('H' == firstWord) {
                                list = (search.getListH());
                                addResult(list, text);


                            } else if ('I' == firstWord) {
                                list = (search.getListI());

                                addResult(list, text);


                            } else if ('J' == firstWord) {
                                list = (search.getListJ());
                                addResult(list, text);


                            } else if ('K' == firstWord) {
                                list = (search.getListK());
                                addResult(list, text);

                            } else if ('L' == firstWord) {
                                list = (search.getListL());

                                addResult(list, text);


                            } else if ('M' == firstWord) {
                                list = (search.getListM());

                                addResult(list, text);


                            } else if ('N' == firstWord) {
                                list = (search.getListN());

                                addResult(list, text);


                            } else if ('O' == firstWord) {
                                list = (search.getListO());
                                addResult(list, text);


                            } else if ('P' == firstWord) {
                                list = (search.getListP());

                                addResult(list, text);


                            } else if ('Q' == firstWord) {
                                list = (search.getListQ());

                                addResult(list, text);


                            } else if ('R' == firstWord) {
                                list = (search.getListR());

                                addResult(list, text);


                            } else if ('S' == firstWord) {
                                list = (search.getListS());

                                addResult(list, text);
                            } else if ('T' == firstWord) {
                                list = (search.getListT());

                                addResult(list, text);

                            } else if ('U' == firstWord) {
                                list = (search.getListU());
                                addResult(list, text);


                            } else if ('V' == firstWord) {
                                list = (search.getListV());

                                addResult(list, text);

                            } else if ('W' == firstWord) {
                                list = (search.getListW());

                                addResult(list, text);

                            } else if ('X' == firstWord) {
                                list = (search.getListX());

                                addResult(list, text);

                            } else if ('Y' == firstWord) {
                                list = (search.getListY());
                                addResult(list, text);

                            } else if ('Z' == firstWord) {
                                list = (search.getListZ());
                                addResult(list, text);
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
                            label1.setText("搜索中...");
                        }
                    }
                    timer = false;
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
                startTime = System.currentTimeMillis();
                timer = true;
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                startTime = System.currentTimeMillis();
                timer = true;
            }

            @Override
            public void changedUpdate(DocumentEvent e) {

            }
        });

        textField.addKeyListener(new KeyListener() {

            @Override
            public void keyPressed(KeyEvent arg0) {
                if (!listResult.isEmpty()) {
                    int key = arg0.getKeyCode();
                    if (38 == key) {
                        //上键被点击
                        labelCount--;
                        if (labelCount < 0) {
                            labelCount = 0;
                        }

                        //System.out.println(labelCount);
                        if (labelCount >= listResult.size()) {
                            labelCount = listResult.size() - 1;
                        }
                        if (labelCount < 3) {
                            //未到最上端
                            if (0 == labelCount) {
                                label1.setBackground(labelColor);
                                try {
                                    String text = label2.getText();
                                    if (text.equals("")) {
                                        label2.setBackground(null);
                                    } else {
                                        label2.setBackground(backgroundColor);
                                    }
                                } catch (NullPointerException e) {
                                    label2.setBackground(null);
                                }
                                try {
                                    String text = label3.getText();
                                    if (text.equals("")) {
                                        label3.setBackground(null);
                                    } else {
                                        label3.setBackground(backgroundColorLight);
                                    }
                                } catch (NullPointerException e) {
                                    label3.setBackground(null);
                                }
                                try {
                                    String text = label4.getText();
                                    if (text.equals("")) {
                                        label4.setBackground(null);
                                    } else {
                                        label4.setBackground(backgroundColor);
                                    }
                                } catch (NullPointerException e) {
                                    label4.setBackground(null);
                                }
                                showResult();
                            } else if (1 == labelCount) {
                                label1.setBackground(backgroundColorLight);
                                label2.setBackground(labelColor);
                                try {
                                    String text = label3.getText();
                                    if (text.equals("")) {
                                        label3.setBackground(null);
                                    } else {
                                        label3.setBackground(backgroundColorLight);
                                    }
                                } catch (NullPointerException e) {
                                    label3.setBackground(null);
                                }
                                try {
                                    String text = label4.getText();
                                    if (text.equals("")) {
                                        label4.setBackground(null);
                                    } else {
                                        label4.setBackground(backgroundColor);
                                    }
                                } catch (NullPointerException e) {
                                    label4.setBackground(null);
                                }
                                showResult();
                            } else if (2 == labelCount) {
                                label1.setBackground(backgroundColorLight);
                                label2.setBackground(backgroundColor);
                                label3.setBackground(labelColor);
                                try {
                                    String text = label4.getText();
                                    if (text.equals("")) {
                                        label4.setBackground(null);
                                    } else {
                                        label4.setBackground(backgroundColor);
                                    }
                                } catch (NullPointerException e) {
                                    label4.setBackground(null);
                                }
                                showResult();
                            }
                        } else {
                            //到达最下端
                            label1.setBackground(backgroundColorLight);
                            label2.setBackground(backgroundColor);
                            label3.setBackground(backgroundColorLight);
                            label4.setBackground(labelColor);
                            String path = listResult.get(labelCount - 3);
                            String name = getFileName(listResult.get(labelCount - 3));
                            ImageIcon icon;
                            if (isDirectory(path) || isFile(path)) {
                                icon = (ImageIcon) getSmallIcon(path);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label1.setIcon(icon);
                                label1.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                            } else {
                                label1.setIcon(null);
                                label1.setText("无效文件");
                                search.addToRecycleBin(path);
                            }
                            path = listResult.get(labelCount - 2);
                            name = getFileName(listResult.get(labelCount - 2));

                            if (isDirectory(path) || isFile(path)) {
                                icon = (ImageIcon) getSmallIcon(path);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label2.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label2.setIcon(icon);
                            } else {
                                label2.setIcon(null);
                                label2.setText("无效文件");
                                search.addToRecycleBin(path);
                            }
                            path = listResult.get(labelCount - 1);
                            name = getFileName(listResult.get(labelCount - 1));


                            if (isDirectory(path) || isFile(path)) {
                                icon = (ImageIcon) getSmallIcon(path);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label3.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label3.setIcon(icon);
                            } else {
                                label3.setIcon(null);
                                label3.setText("无效文件");
                                search.addToRecycleBin(path);
                            }
                            path = listResult.get(labelCount);
                            name = getFileName(listResult.get(labelCount));


                            if (isDirectory(path) || isFile(path)) {
                                icon = (ImageIcon) getSmallIcon(path);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label4.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label4.setIcon(icon);
                            } else {
                                label4.setIcon(null);
                                label4.setText("无效文件");
                                search.addToRecycleBin(path);
                            }
                        }
                    } else if (40 == key) {
                        //下键被点击
                        labelCount++;
                        if (labelCount < 0) {
                            labelCount = 0;
                        }

                        //System.out.println(labelCount);
                        if (labelCount >= listResult.size()) {
                            labelCount = listResult.size() - 1;
                        }
                        if (labelCount < 3) {
                            //未到最下端
                            if (0 == labelCount) {
                                label1.setBackground(labelColor);
                                try {
                                    String text = label2.getText();
                                    if (text.equals("")) {
                                        label2.setBackground(null);
                                    } else {
                                        label2.setBackground(backgroundColor);
                                    }
                                } catch (NullPointerException e) {
                                    label2.setBackground(null);
                                }
                                try {
                                    String text = label3.getText();
                                    if (text.equals("")) {
                                        label3.setBackground(null);
                                    } else {
                                        label3.setBackground(backgroundColorLight);
                                    }
                                } catch (NullPointerException e) {
                                    label3.setBackground(null);
                                }
                                try {
                                    String text = label4.getText();
                                    if (text.equals("")) {
                                        label4.setBackground(null);
                                    } else {
                                        label4.setBackground(backgroundColor);
                                    }
                                } catch (NullPointerException e) {
                                    label4.setBackground(null);
                                }
                                showResult();
                            } else if (1 == labelCount) {
                                label1.setBackground(backgroundColorLight);
                                label2.setBackground(labelColor);
                                try {
                                    String text = label3.getText();
                                    if (text.equals("")) {
                                        label3.setBackground(null);
                                    } else {
                                        label3.setBackground(backgroundColorLight);
                                    }
                                } catch (NullPointerException e) {
                                    label3.setBackground(null);
                                }
                                try {
                                    String text = label4.getText();
                                    if (text.equals("")) {
                                        label4.setBackground(null);
                                    } else {
                                        label4.setBackground(backgroundColor);
                                    }
                                } catch (NullPointerException e) {
                                    label4.setBackground(null);
                                }
                                showResult();
                            } else if (2 == labelCount) {
                                label1.setBackground(backgroundColorLight);
                                label2.setBackground(backgroundColor);
                                label3.setBackground(labelColor);
                                try {
                                    String text = label4.getText();
                                    if (text.equals("")) {
                                        label4.setBackground(null);
                                    } else {
                                        label4.setBackground(backgroundColor);
                                    }
                                } catch (NullPointerException e) {
                                    label4.setBackground(null);
                                }
                                showResult();
                            }
                        } else {
                            //到最下端
                            label1.setBackground(backgroundColorLight);
                            label2.setBackground(backgroundColor);
                            label3.setBackground(backgroundColorLight);
                            label4.setBackground(labelColor);
                            String path = listResult.get(labelCount - 3);
                            String name = getFileName(listResult.get(labelCount - 3));
                            ImageIcon icon;
                            if (isDirectory(path) || isFile(path)) {
                                icon = (ImageIcon) getSmallIcon(path);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label1.setIcon(icon);
                                label1.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                            } else {
                                label1.setText("无效文件");
                                search.addToRecycleBin(path);
                            }
                            path = listResult.get(labelCount - 2);
                            name = getFileName(listResult.get(labelCount - 2));


                            if (isDirectory(path) || isFile(path)) {
                                icon = (ImageIcon) getSmallIcon(path);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label2.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label2.setIcon(icon);
                            } else {
                                label2.setText("无效文件");
                                search.addToRecycleBin(path);
                            }
                            path = listResult.get(labelCount - 1);
                            name = getFileName(listResult.get(labelCount - 1));


                            if (isDirectory(path) || isFile(path)) {
                                icon = (ImageIcon) getSmallIcon(path);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label3.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label3.setIcon(icon);
                            } else {
                                label3.setText("无效文件");
                                search.addToRecycleBin(path);
                            }
                            path = listResult.get(labelCount);
                            name = getFileName(listResult.get(labelCount));


                            if (isDirectory(path) || isFile(path)) {
                                icon = (ImageIcon) getSmallIcon(path);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label4.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label4.setIcon(icon);
                            } else {
                                label4.setText("无效文件");
                                search.addToRecycleBin(path);
                            }
                        }
                    } else if (10 == key) {
                        //enter被点击
                        closedTodo();
                        if (isCtrlPressed) {
                            //打开上级文件夹
                            File open = new File(listResult.get(labelCount));
                            try {
                                Runtime.getRuntime().exec("explorer.exe /select, \"" + open.getAbsolutePath() + "\"");
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        } else {
                            open(listResult.get(labelCount));
                        }
                        saveCache(listResult.get(labelCount) + ';');
                    } else if (17 == key) {
                        //ctrl被点击
                        isCtrlPressed = true;
                    }
                }

            }

            @Override
            public void keyReleased(KeyEvent arg0) {
                int key = arg0.getKeyCode();
                if (17 == key) {
                    //复位CTRL状态
                    isCtrlPressed = false;
                }
            }

            @Override
            public void keyTyped(KeyEvent arg0) {

            }
        });


        //labels
        Font font = new Font("Microsoft JhengHei", Font.BOLD, (int) ((height * 0.1) / 96 * 72) / 4);
        Color fontColor = new Color(73, 162, 255, 255);
        label1.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label1.setLocation(0, (int) (searchBarHeight * 0.2));
        label1.setFont(font);
        label1.setForeground(fontColor);
        label1.setBackground(null);
        label1.setOpaque(true);


        label2.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label2.setLocation(0, (int) (searchBarHeight * 0.4));
        label2.setFont(font);
        label2.setForeground(fontColor);
        label2.setBackground(null);
        label2.setOpaque(true);


        label3.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label3.setLocation(0, (int) (searchBarHeight * 0.6));
        label3.setFont(font);
        label3.setForeground(fontColor);
        label3.setBackground(null);
        label3.setOpaque(true);


        label4.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label4.setLocation(0, (int) (searchBarHeight * 0.8));
        label4.setFont(font);
        label4.setForeground(fontColor);
        label4.setBackground(null);
        label4.setOpaque(true);


        //panel
        panel.setLayout(null);
        panel.setBackground(new Color(0, 0, 0, 0));
        panel.add(textField);
        panel.add(label1);
        panel.add(label2);
        panel.add(label3);
        panel.add(label4);


        URL icon = TaskBar.class.getResource("/icons/taskbar_32x32.png");
        Image image = new ImageIcon(icon).getImage();
        searchBar.setIconImage(image);
        searchBar.setBackground(new Color(0, 0, 0, 0));
    }

    public static SearchBar getInstance() {
        return searchBarInstance;
    }

    private static Icon getSmallIcon(String file) {
        File f = new File(file);
        if (f.exists()) {
            FileSystemView fsv = FileSystemView.getFileSystemView();
            return (fsv.getSystemIcon(f));
        }
        return (null);
    }

    public boolean isUsing() {
        return this.isUsing;
    }


    private void addResult(LinkedHashSet<String> list, String text) {
        String[] strings = text.split(";");
        String searchText = strings[0];
        int length = strings.length;
        if (search.isUsable()) {
            label:
            for (String fileInList : list) {
                if (length != 2 && match(getFileName(fileInList), searchText)) {
                    listResult.add(fileInList);
                    if (listResult.size() > 100) {
                        break;
                    }
                } else if (match(getFileName(fileInList), searchText) && length == 2) {
                    switch (strings[1].toUpperCase()) {
                        case "FILE":
                            if (isFile(fileInList)) {

                                listResult.add(fileInList);
                                if (listResult.size() > 100) {
                                    break label;
                                }
                            }
                            break;
                        case "FOLDER":
                            if (isDirectory(fileInList)) {

                                listResult.add(fileInList);
                                if (listResult.size() > 100) {
                                    break label;
                                }
                            }
                            break;
                        case "FULL":
                            if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                listResult.add(fileInList);
                                if (listResult.size() > 100) {
                                    break label;
                                }
                            }
                            break;
                    }
                }
            }
        }
        delRepeated(listResult);
        int size = listResult.size();
        if (size == 1) {
            label1.setBackground(labelColor);
            label2.setBackground(null);
            label3.setBackground(null);
            label4.setBackground(null);
        } else if (size == 2) {
            label1.setBackground(labelColor);
            label2.setBackground(backgroundColor);
            label3.setBackground(null);
            label4.setBackground(null);
        } else if (size == 3) {
            label1.setBackground(labelColor);
            label2.setBackground(backgroundColor);
            label3.setBackground(backgroundColorLight);
            label4.setBackground(null);
        } else if (size >= 4) {
            label1.setBackground(labelColor);
            label2.setBackground(backgroundColor);
            label3.setBackground(backgroundColorLight);
            label4.setBackground(backgroundColor);
        }
        showResult();
    }

    public void showSearchbar() {
        textField.setCaretPosition(0);
        //添加更新文件
        System.out.println("正在添加更新文件");
        search.setUsable(false);
        search.mergeFileToList();
        search.setUsable(true);
        isUsing = true;
        searchBar.setVisible(true);
    }

    private void showResult() {
        try {
            String path = listResult.get(0);
            String name = getFileName(listResult.get(0));
            ImageIcon icon;
            if (isDirectory(path) || isFile(path)) {
                icon = (ImageIcon) getSmallIcon(path);
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label1.setIcon(icon);
                label1.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
            } else {
                label1.setIcon(null);
                label1.setText("无效文件");
                search.addToRecycleBin(path);
            }
            path = listResult.get(1);
            name = getFileName(listResult.get(1));


            if (isDirectory(path) || isFile(path)) {
                icon = (ImageIcon) getSmallIcon(path);
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label2.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                label2.setIcon(icon);
            } else {
                label2.setIcon(null);
                label2.setText("无效文件");
                search.addToRecycleBin(path);
            }
            path = listResult.get(2);
            name = getFileName(listResult.get(2));


            if (isDirectory(path) || isFile(path)) {
                icon = (ImageIcon) getSmallIcon(path);
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label3.setIcon(icon);
                label3.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
            } else {
                label3.setIcon(null);
                label3.setText("无效文件");
                search.addToRecycleBin(path);
            }
            path = listResult.get(3);
            name = getFileName(listResult.get(3));


            if (isDirectory(path) || isFile(path)) {
                icon = (ImageIcon) getSmallIcon(path);
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label4.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                label4.setIcon(icon);
            } else {
                label4.setIcon(null);
                label4.setText("无效文件");
                search.addToRecycleBin(path);
            }
        } catch (java.lang.IndexOutOfBoundsException ignored) {

        }
    }


    private void clearLabel() {
        label1.setIcon(null);
        label2.setIcon(null);
        label3.setIcon(null);
        label4.setIcon(null);
        label1.setText(null);
        label2.setText(null);
        label3.setText(null);
        label4.setText(null);
        label1.setBackground(null);
        label2.setBackground(null);
        label3.setBackground(null);
        label4.setBackground(null);
    }

    private void open(String path) {
        File name = new File(path);
        if (name.exists()) {
            try {
                try {
                    Runtime.getRuntime().exec("\"" + name.getAbsolutePath() + "\"", null, name.getParentFile());
                } catch (IOException e) {
                    Desktop desktop;
                    if (Desktop.isDesktopSupported()) {
                        desktop = Desktop.getDesktop();
                        desktop.open(name);
                    }
                }
            } catch (IOException e) {
                //打开上级文件夹
                try {
                    Runtime.getRuntime().exec("explorer.exe /select, \"" + name.getAbsolutePath() + "\"");
                } catch (IOException ignored) {

                }
            }
        }
    }

    private String getFileName(String path) {
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

    public void delRepeated(LinkedList<String> list) {
        LinkedHashSet<String> set = new LinkedHashSet<>();
        try {
            set.addAll(list);
        } catch (Exception ignored) {

        }
        list.clear();
        list.addAll(set);
    }

    private void saveCache(String content) {
        int cacheNum = 0;
        File cache = new File("cache.dat");
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
            isRepeated = match(oldCaches.toString() + ";", (content));
            if (!isRepeated) {
                try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File("cache.dat"), true)))) {
                    out.write(content + "\r\n");
                } catch (Exception ignored) {

                }
            }
        }
    }

    private void delCache(LinkedList<String> cache) {
        File cacheFile = new File("cache.dat");
        StringBuilder allCaches = new StringBuilder();
        String eachLine;
        if (cacheFile.exists()) {
            try (BufferedReader br = new BufferedReader(new FileReader(cacheFile))) {
                while ((eachLine = br.readLine()) != null) {
                    String[] each = eachLine.split(";");
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

    private void searchCache(String searchFile) {
        String cacheResult;
        LinkedList<String> cachesToDel = new LinkedList<>();
        File cache = new File("cache.dat");
        if (cache.exists()) {
            try (BufferedReader reader = new BufferedReader(new FileReader(cache))) {
                while ((cacheResult = reader.readLine()) != null) {
                    String[] caches = cacheResult.split(";");
                    for (String cach : caches) {
                        if (!(new File(cach).exists())) {
                            cachesToDel.add(cach);
                        } else {
                            String eachCacheName = getFileName(cach);
                            if (eachCacheName.toLowerCase().contains(searchFile.toLowerCase())) {
                                listResult.addFirst(cach);
                            }
                        }
                    }
                }
            } catch (IOException ignored) {

            }
        }
        delCache(cachesToDel);
    }

    private boolean match(String srcText, String txt) {
        srcText = srcText.toLowerCase();
        txt = txt.toLowerCase();
        if (srcText.length() >= txt.length()) {
            return srcText.contains(txt);
        }
        return false;
    }

    private void searchPriorityFolder(String text) {
        File path = new File(SettingsFrame.priorityFolder);
        boolean exist = path.exists();
        LinkedList<File> listRemain = new LinkedList<>();
        if (exist) {
            File[] files = path.listFiles();
            if (!(null == files || files.length == 0)) {
                for (File each : files) {
                    if (match(getFileName(each.getAbsolutePath()), text)) {
                        listResult.addFirst(each.getAbsolutePath());
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
                        if (match(getFileName(each.getAbsolutePath()), text)) {
                            listResult.addFirst(each.getAbsolutePath());
                        }
                        if (each.isDirectory()) {
                            listRemain.add(each);
                        }
                    }
                }
            }
        }
    }

    public Container getPanel() {
        return this.panel;
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
            CheckHotKey.setShowSearchBar(false);
            search.setUsable(false);
            search.mergeAndClearRecycleBin();
            search.setUsable(true);
            clearLabel();
            isUsing = false;
            labelCount = 0;
            listResult.clear();
            textField.setText(null);
            try {
                thread.interrupt();
                searchWaiter.interrupt();
            } catch (NullPointerException ignored) {

            }
        };
        SwingUtilities.invokeLater(todo);
    }

    private boolean isFile(String text) {
        File file = new File(text);
        return file.isFile();
    }

    private boolean isDirectory(String text) {
        File file = new File(text);
        return file.isDirectory();
    }

    class AddAllResults implements Runnable {
        private String text;
        private String searchText;
        private int length;
        private String[] strings;

        AddAllResults(String txt) {
            this.text = txt.substring(1);
            strings = this.text.split(";");
            try {
                searchText = strings[0];
            } catch (ArrayIndexOutOfBoundsException e) {
                searchText = "";
            }
            length = strings.length;
        }

        @Override
        public void run() {
            if (search.isUsable()) {
                if (!this.text.equals("")) {
                    label:
                    for (String fileInList : search.getListA()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList).toLowerCase().equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label1:
                    for (String fileInList : search.getListB()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label1;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label1;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label1;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label2:
                    for (String fileInList : search.getListC()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label2;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label2;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label2;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label3:
                    for (String fileInList : search.getListD()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label3;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label3;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label3;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label4:
                    for (String fileInList : search.getListE()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label4;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label4;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label4;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label5:
                    for (String fileInList : search.getListF()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label5;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label5;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label5;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label6:
                    for (String fileInList : search.getListG()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label6;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label6;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label6;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label7:
                    for (String fileInList : search.getListH()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label7;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label7;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label7;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label8:
                    for (String fileInList : search.getListI()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label8;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label8;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label8;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label9:
                    for (String fileInList : search.getListJ()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label9;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label9;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label9;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label10:
                    for (String fileInList : search.getListK()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label10;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label10;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label10;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label11:
                    for (String fileInList : search.getListL()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label11;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label11;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label11;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label12:
                    for (String fileInList : search.getListM()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label12;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label12;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label12;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label13:
                    for (String fileInList : search.getListN()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label13;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label13;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label13;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label14:
                    for (String fileInList : search.getListO()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label14;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label14;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label14;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label15:
                    for (String fileInList : search.getListP()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label15;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label15;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label15;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label16:
                    for (String fileInList : search.getListQ()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label16;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label16;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label16;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label17:
                    for (String fileInList : search.getListR()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label17;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label17;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label17;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label18:
                    for (String fileInList : search.getListS()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label18;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label18;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label18;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label19:
                    for (String fileInList : search.getListT()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label19;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label19;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label19;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label20:
                    for (String fileInList : search.getListU()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label20;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label20;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label20;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label21:
                    for (String fileInList : search.getListV()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label21;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label21;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label21;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label22:
                    for (String fileInList : search.getListW()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label22;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label22;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label22;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label23:
                    for (String fileInList : search.getListX()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label23;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label23;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label23;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label24:
                    for (String fileInList : search.getListY()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label24;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label24;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label24;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label25:
                    for (String fileInList : search.getListZ()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label25;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label25;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label25;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label26:
                    for (String fileInList : search.getListNum()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label26;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label26;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label26;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label27:
                    for (String fileInList : search.getListUnderline()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label27;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label27;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label27;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label28:
                    for (String fileInList : search.getListUnique()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label28;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label28;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label28;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    label29:
                    for (String fileInList : search.getListPercentSign()) {

                        if (length != 2 && match(getFileName(fileInList), searchText)) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList), searchText) && length == 2) {
                            switch (strings[1].toUpperCase()) {
                                case "FILE":
                                    if (isFile(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label29;
                                        }
                                    }
                                    break;
                                case "FOLDER":
                                    if (isDirectory(fileInList)) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label29;
                                        }
                                    }
                                    break;
                                case "FULL":
                                    if (getFileName(fileInList.toLowerCase()).equals(searchText.toLowerCase())) {
                                        listResult.add(fileInList);
                                        if (listResult.size() > 100) {
                                            break label29;
                                        }
                                    }
                                    break;
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                }
                if (!Thread.currentThread().isInterrupted()) {
                    delRepeated(listResult);
                    int size = listResult.size();
                    if (size == 1) {
                        label1.setBackground(labelColor);
                        label2.setBackground(null);
                        label3.setBackground(null);
                        label4.setBackground(null);
                    } else if (size == 2) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(null);
                        label4.setBackground(null);
                    } else if (size == 3) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColorLight);
                        label4.setBackground(null);
                    } else if (size >= 4) {
                        label1.setBackground(labelColor);
                        label2.setBackground(backgroundColor);
                        label3.setBackground(backgroundColorLight);
                        label4.setBackground(backgroundColor);
                    }
                    showResult();
                }
            }
        }
    }
}


