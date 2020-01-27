package frame;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.*;
import java.net.URL;
import java.util.LinkedHashSet;
import java.util.LinkedList;

import search.*;


public class SearchBar {
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
    private String path;
    private Search search = new Search();
    private Color labelColor = new Color(255, 125, 29, 255);
    private LinkedList<String> list = new LinkedList<>();
    private Thread thread;
    private boolean isFirstRun = true;


    public SearchBar() {
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
        int width = screenSize.width;
        int height = screenSize.height;
        int positionX = (int) (width * 0.15);
        int positionY = (int) (height * 0.2);
        int searchBarWidth = (int) (width * 0.7);
        int searchBarHeight = (int) (height * 0.5);


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
        textField.setForeground(Color.white);
        textField.setHorizontalAlignment(JTextField.LEFT);
        textField.setBorder(null);
        textField.setBackground(new Color(75, 75, 75, 255));
        textField.setLocation(0, 0);
        textField.addFocusListener(new FocusListener() {

            @Override
            public void focusGained(FocusEvent arg0) {
                textField.setCaretPosition(textField.getText().length());
            }

            @Override
            public void focusLost(FocusEvent arg0) {
                searchBar.setVisible(false);
                clearLabel();
                labelCount = 0;
                listResult.clear();
                textField.setText(null);
                search.setFocusLostStatus(true);
            }
        });
        //当textField文件变动时开始
        textField.getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent arg0) {
                labelCount = 0;
                clearLabel();
                label1.setBackground(labelColor);
                listResult.clear();
                String text = textField.getText();
                if (!text.equals("")) {
                    if (text.equals("#*update*#")) {
                        clearTextFieldText();
                        search.setSearch(true);
                        return;
                    }
                    text = text.toUpperCase();
                    searchFilesFolder(text);
                    searchCache(text);
                    char firstWord = text.charAt(0);
                    if ('?' == firstWord) {
                        if (isFirstRun || !thread.isAlive()) {
                            isFirstRun = false;
                        } else {
                            thread.interrupt();
                        }
                        thread = new Thread(new AddAllResult(text));
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
                }

            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                labelCount = 0;
                clearLabel();
                listResult.clear();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                labelCount = 0;
                clearLabel();
                label1.setBackground(labelColor);
                listResult.clear();
                String text = textField.getText();
                if (!text.equals("")) {
                    if (text.equals("#*update*#")) {
                        clearTextFieldText();
                        search.setSearch(true);
                        return;
                    }
                    text = text.toUpperCase();
                    searchCache(text);
                    char firstWord;

                    searchFilesFolder(text);
                    firstWord = text.charAt(0);
                    if ('?' == firstWord) {
                        if (isFirstRun || !thread.isAlive()) {
                            isFirstRun = false;
                        } else {
                            thread.interrupt();
                        }
                        thread = new Thread(new AddAllResult(text));
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
                    clearLabel();
                    labelCount = 0;
                    listResult.clear();
                }
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

                        System.out.println(labelCount);
                        if (labelCount >= listResult.size()) {
                            labelCount = listResult.size() - 1;
                        }
                        if (labelCount < 3) {
                            //未到最上端
                            if (0 == labelCount) {
                                label1.setBackground(labelColor);
                                label2.setBackground(null);
                                label3.setBackground(null);
                                label4.setBackground(null);
                                showResult();
                            } else if (1 == labelCount) {
                                label1.setBackground(null);
                                label2.setBackground(labelColor);
                                label3.setBackground(null);
                                label4.setBackground(null);
                                showResult();
                            } else if (2 == labelCount) {
                                label1.setBackground(null);
                                label2.setBackground(null);
                                label3.setBackground(labelColor);
                                label4.setBackground(null);
                                showResult();
                            }
                        } else {
                            //到达最下端
                            label1.setBackground(null);
                            label2.setBackground(null);
                            label3.setBackground(null);
                            label4.setBackground(labelColor);
                            path = listResult.get(labelCount - 3);
                            String name = getFileName(listResult.get(labelCount - 3));
                            File file = new File(path);
                            ImageIcon icon;
                            if (file.isDirectory() || file.isFile()) {
                                icon = (ImageIcon) getSmallIcon(file);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label1.setIcon(icon);
                                label1.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                            } else {
                                label1.setIcon(null);
                                label1.setText("无效文件");
                            }
                            path = listResult.get(labelCount - 2);
                            name = getFileName(listResult.get(labelCount - 2));
                            file = new File(path);

                            if (file.isDirectory() || file.isFile()) {
                                icon = (ImageIcon) getSmallIcon(file);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label2.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label2.setIcon(icon);
                            } else {
                                label2.setIcon(null);
                                label2.setText("无效文件");
                            }
                            path = listResult.get(labelCount - 1);
                            name = getFileName(listResult.get(labelCount - 1));
                            file = new File(path);

                            if (file.isDirectory() || file.isFile()) {
                                icon = (ImageIcon) getSmallIcon(file);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label3.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label3.setIcon(icon);
                            } else {
                                label3.setIcon(null);
                                label3.setText("无效文件");
                            }
                            path = listResult.get(labelCount);
                            name = getFileName(listResult.get(labelCount));
                            file = new File(path);

                            if (file.isDirectory() || file.isFile()) {
                                icon = (ImageIcon) getSmallIcon(file);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label4.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label4.setIcon(icon);
                            } else {
                                label4.setIcon(null);
                                label4.setText("无效文件");
                            }
                        }
                    } else if (40 == key) {
                        //下键被点击
                        labelCount++;
                        if (labelCount < 0) {
                            labelCount = 0;
                        }

                        System.out.println(labelCount);
                        if (labelCount >= listResult.size()) {
                            labelCount = listResult.size() - 1;
                        }
                        if (labelCount < 3) {
                            //未到最下端
                            if (0 == labelCount) {
                                label1.setBackground(labelColor);
                                label2.setBackground(null);
                                label3.setBackground(null);
                                label4.setBackground(null);
                                showResult();
                            } else if (1 == labelCount) {
                                label1.setBackground(null);
                                label2.setBackground(labelColor);
                                label3.setBackground(null);
                                label4.setBackground(null);
                                showResult();
                            } else if (2 == labelCount) {
                                label1.setBackground(null);
                                label2.setBackground(null);
                                label3.setBackground(labelColor);
                                label4.setBackground(null);
                                showResult();
                            }
                        } else {
                            //到最下端
                            label1.setBackground(null);
                            label2.setBackground(null);
                            label3.setBackground(null);
                            label4.setBackground(labelColor);
                            path = listResult.get(labelCount - 3);
                            String name = getFileName(listResult.get(labelCount - 3));
                            ImageIcon icon;
                            File file = new File(path);
                            if (file.isDirectory() || file.isFile()) {
                                icon = (ImageIcon) getSmallIcon(file);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label1.setIcon(icon);
                                label1.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                            } else {
                                label1.setText("无效文件");
                            }
                            path = listResult.get(labelCount - 2);
                            name = getFileName(listResult.get(labelCount - 2));
                            file = new File(path);

                            if (file.isDirectory() || file.isFile()) {
                                icon = (ImageIcon) getSmallIcon(file);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label2.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label2.setIcon(icon);
                            } else {
                                label2.setText("无效文件");
                            }
                            path = listResult.get(labelCount - 1);
                            name = getFileName(listResult.get(labelCount - 1));
                            file = new File(path);

                            if (file.isDirectory() || file.isFile()) {
                                icon = (ImageIcon) getSmallIcon(file);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label3.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label3.setIcon(icon);
                            } else {
                                label3.setText("无效文件");
                            }
                            path = listResult.get(labelCount);
                            name = getFileName(listResult.get(labelCount));
                            file = new File(path);

                            if (file.isDirectory() || file.isFile()) {
                                icon = (ImageIcon) getSmallIcon(file);
                                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                label4.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                label4.setIcon(icon);
                            } else {
                                label4.setText("无效文件");
                            }
                        }
                    } else if (10 == key) {
                        //enter被点击
                        searchBar.setVisible(false);
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

    private static Icon getSmallIcon(File f) {
        if (f != null && f.exists()) {
            FileSystemView fsv = FileSystemView.getFileSystemView();
            return (fsv.getSystemIcon(f));
        }
        return (null);
    }

    private void addResult(LinkedList<String> list, String text) {
        String[] strings = text.split(";");
        String searchText = strings[0];
        int length = strings.length;
        if (!search.isSearch()) {
            for (String fileInList : list) {
                if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                    listResult.add(fileInList);
                    if (listResult.size() > 100) {
                        break;
                    }
                } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                    if (strings[1].toUpperCase().equals("FILE")) {
                        if (new File(fileInList).isFile()) {

                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        }
                    } else if (strings[1].toUpperCase().equals("FOLDER")) {
                        if (new File(fileInList).isDirectory()) {

                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        }
                    }
                }
            }
        }
        delRepeated(listResult);
        showResult();
    }

    public void showSearchbar() {
        textField.grabFocus();
        searchBar.setVisible(true);
    }

    private void showResult() {
        try {
            path = listResult.get(0);
            String name = getFileName(listResult.get(0));
            File file = new File(path);
            ImageIcon icon;
            if (file.isDirectory() || file.isFile()) {
                icon = (ImageIcon) getSmallIcon(file);
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label1.setIcon(icon);
                label1.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
            } else {
                label1.setIcon(null);
                label1.setText("无效文件");
            }
            path = listResult.get(1);
            name = getFileName(listResult.get(1));
            file = new File(path);

            if (file.isDirectory() || file.isFile()) {
                icon = (ImageIcon) getSmallIcon(file);
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label2.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                label2.setIcon(icon);
            } else {
                label2.setIcon(null);
                label2.setText("无效文件");
            }
            path = listResult.get(2);
            name = getFileName(listResult.get(2));
            file = new File(path);

            if (file.isDirectory() || file.isFile()) {
                icon = (ImageIcon) getSmallIcon(file);
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label3.setIcon(icon);
                label3.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
            } else {
                label3.setIcon(null);
                label3.setText("无效文件");
            }
            path = listResult.get(3);
            name = getFileName(listResult.get(3));
            file = new File(path);

            if (file.isDirectory() || file.isFile()) {
                icon = (ImageIcon) getSmallIcon(file);
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label4.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                label4.setIcon(icon);
            } else {
                label4.setIcon(null);
                label4.setText("无效文件");
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
                    Runtime.getRuntime().exec(name.getAbsolutePath(), null, new File(name.getParent()));
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
        String placeHolder = "";
        String[] names;
        String name;
        try {
            names = path.split("\\\\");
            name = names[names.length - 1];
            if (name.contains(":")) {
                name = name.substring(name.indexOf(':') + 2);
            }
            return name;
        } catch (Exception e) {
            return placeHolder;
        }
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
                            if (eachCacheName.toUpperCase().contains(searchFile)) {
                                listResult.addFirst(cach);
                                //System.out.println("adding"+cach);
                            }
                        }
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            //e.printStackTrace();
        }
        delCache(cachesToDel);
    }

    private boolean match(String srcText, String txt) {
        if (srcText.length() >= txt.length()) {
            while (srcText.length() >= txt.length()) {
                String each = srcText.substring(0, txt.length());
                if (each.toUpperCase().equals(txt.toUpperCase())) {
                    return true;
                } else {
                    srcText = srcText.substring(1);
                }
            }
        }
        return false;
    }

    private void searchFilesFolder(String text) {
        File path = new File("Files");
        boolean exist = path.exists();
        LinkedList<File> listRemain = new LinkedList<>();
        if (exist) {
            File[] files = path.listFiles();
            if (!(null == files || files.length == 0)) {
                for (File each : files) {
                    if (match(getFileName(each.getAbsolutePath()).toUpperCase(), text.toUpperCase())) {
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
                        if (match(getFileName(each.getAbsolutePath()).toUpperCase(), text.toUpperCase())) {
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

    class AddAllResult implements Runnable {
        private String text;
        private String searchText;
        private int length;
        private String[] strings;

        AddAllResult(String txt) {
            this.text = txt.substring(1);
            strings = this.text.split(";");
            searchText = strings[0];
            length = strings.length;
        }

        @Override
        public void run() {
            if (!search.isSearch()) {
                if (!this.text.equals("")) {
                    for (String fileInList : search.getListA()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);
                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListB()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListC()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListD()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListE()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListF()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListG()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListH()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListI()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListJ()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListK()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListL()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListM()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListN()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListO()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListP()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListQ()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListR()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListS()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListT()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListU()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListV()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListW()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListX()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListY()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListZ()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListNum()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListPercentSign()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListUnderline()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                    for (String fileInList : search.getListUnique()) {
                        if (length != 2 && match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase())) {
                            listResult.add(fileInList);

                            if (listResult.size() > 100) {
                                break;
                            }
                        } else if (match(getFileName(fileInList).toUpperCase(), searchText.toUpperCase()) && length == 2) {
                            if (strings[1].toUpperCase().equals("FILE")) {
                                if (new File(fileInList).isFile()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            } else if (strings[1].toUpperCase().equals("FOLDER")) {
                                if (new File(fileInList).isDirectory()) {
                                    listResult.add(fileInList);

                                    if (listResult.size() > 100) {
                                        break;
                                    }
                                }
                            }
                        }
                        if (Thread.currentThread().isInterrupted()) {
                            return;
                        }
                    }
                }
            }
            if (!Thread.currentThread().isInterrupted()) {
                delRepeated(listResult);
                showResult();
            }
        }
    }
}


