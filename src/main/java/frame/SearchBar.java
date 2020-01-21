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
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.LinkedList;

import search.*;


public class SearchBar {
    private JFrame searchBar = new JFrame();
    private LinkedList<String> listResult = new LinkedList<>();
    private JLabel label1 = new JLabel();
    private JLabel label2 = new JLabel();
    private JLabel label3 = new JLabel();
    private JLabel label4 = new JLabel();
    private boolean isCtrlPressed = false;
    private int labelCount = 0;
    private JTextField textField;
    private Container panel;
    private Thread thread;
    private boolean isFirstRun = true;



    private void addResult(ArrayList<String> list, String text) {
        if (!new Search().isSearch()) {
            for (String fileInList : list) {
                if (match(getFileName(fileInList).toUpperCase(), text.toUpperCase())) {
                    listResult.add(fileInList);
                    //System.out.println("adding "+fileInList.toString());
                }
            }
        }
        delRepeated(listResult);
        showResult();
    }


    class AddAllResult implements Runnable {
        private String text;

        AddAllResult(String txt) {
            this.text = txt.substring(1);
        }

        @Override
        public void run() {
            if (!new Search().isSearch()) {
                if (!this.text.equals("")) {
                    for (String fileInList : Search.listA) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }
                        if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listB) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }
                        if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listC) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }
                        if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listD) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }
                        if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listE) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }
                        if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listF) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }
                        if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listG) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listH) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listI) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listJ) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listK) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listL) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listM) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listN) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listO) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listP) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listQ) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listR) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listS) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listT) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listU) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listV) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listW) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listX) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listY) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listZ) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listNum) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listPercentSign) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listUnderline) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
                            return;
                        }
                    }
                    for (String fileInList : Search.listUnique) {
                        if (match(getFileName(fileInList).toUpperCase(), this.text.toUpperCase())) {
                            listResult.add(fileInList);
                            //System.out.println("adding "+fileInList.toString());
                        }if (Thread.currentThread().isInterrupted()){
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
        textField.setFont(new Font("微软雅黑", Font.BOLD, (int) ((height * 0.1) / 96 * 72)));
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
            }
        });
        //当textField文本变动时开始
        textField.getDocument().addDocumentListener(new javax.swing.event.DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent arg0) {
                labelCount = 0;
                clearLabel();
                label1.setBackground(new Color(255, 125, 29, 255));
                listResult.clear();
                String text = textField.getText();
                if (!text.equals("")) {
                    if (text.equals("#*update*#")) {
                        clearTextFieldText();
                        new Search().setSearch(true);
                        return;
                    }
                    text = text.toUpperCase();
                    searchFilesFolder(text);
                    searchCache(text);
                    char firstWord = text.charAt(0);
                    if ('?' == firstWord) {
                        if (isFirstRun || !thread.isAlive()) {
                            isFirstRun = false;
                        }else{
                            thread.interrupt();
                        }
                        thread = new Thread(new AddAllResult(text));
                        thread.start();

                    } else {
                        if (19968 <= (int) firstWord && (int) firstWord < 40869) {
                            addResult(Search.listUnique, text);

                        } else if ('%' == firstWord) {

                            addResult(Search.listPercentSign, text);

                        } else if ('_' == firstWord) {

                            addResult(Search.listUnderline, text);

                        } else if (Character.isDigit(firstWord)) {

                            addResult(Search.listNum, text);


                        } else if (Character.isAlphabetic(firstWord)) {
                            firstWord = Character.toUpperCase(firstWord);
                            if ('A' == firstWord) {
                                addResult(Search.listA, text);


                            } else if ('B' == firstWord) {

                                addResult(Search.listB, text);


                            } else if ('C' == firstWord) {

                                addResult(Search.listC, text);


                            } else if ('D' == firstWord) {

                                addResult(Search.listD, text);


                            } else if ('E' == firstWord) {
                                addResult(Search.listE, text);


                            } else if ('F' == firstWord) {

                                addResult(Search.listF, text);


                            } else if ('G' == firstWord) {

                                addResult(Search.listG, text);


                            } else if ('H' == firstWord) {

                                addResult(Search.listH, text);


                            } else if ('I' == firstWord) {

                                addResult(Search.listI, text);


                            } else if ('J' == firstWord) {
                                addResult(Search.listJ, text);


                            } else if ('K' == firstWord) {

                                addResult(Search.listK, text);


                            } else if ('L' == firstWord) {

                                addResult(Search.listL, text);


                            } else if ('M' == firstWord) {

                                addResult(Search.listM, text);


                            } else if ('N' == firstWord) {

                                addResult(Search.listN, text);


                            } else if ('O' == firstWord) {
                                addResult(Search.listO, text);


                            } else if ('P' == firstWord) {

                                addResult(Search.listP, text);


                            } else if ('Q' == firstWord) {

                                addResult(Search.listQ, text);


                            } else if ('R' == firstWord) {

                                addResult(Search.listR, text);


                            } else if ('S' == firstWord) {

                                addResult(Search.listS, text);


                            } else if ('T' == firstWord) {

                                addResult(Search.listT, text);


                            } else if ('U' == firstWord) {

                                addResult(Search.listU, text);


                            } else if ('V' == firstWord) {

                                addResult(Search.listV, text);

                            } else if ('W' == firstWord) {

                                addResult(Search.listW, text);

                            } else if ('X' == firstWord) {

                                addResult(Search.listX, text);


                            } else if ('Y' == firstWord) {

                                addResult(Search.listY, text);

                            } else if ('Z' == firstWord) {

                                addResult(Search.listZ, text);

                            }
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
                label1.setBackground(new Color(255, 125, 29, 255));
                listResult.clear();
                String text = textField.getText();
                if (!text.equals("")) {
                    if (text.equals("#*update*#")) {
                        clearTextFieldText();
                        new Search().setSearch(true);
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
                        }else{
                            thread.interrupt();
                        }
                        thread = new Thread(new AddAllResult(text));
                        thread.start();
                    } else {
                        if (19968 <= (int) firstWord && (int) firstWord < 40869) {
                            addResult(Search.listUnique, text);

                        } else if ('%' == firstWord) {

                            addResult(Search.listPercentSign, text);

                        } else if ('_' == firstWord) {

                            addResult(Search.listUnderline, text);

                        } else if (Character.isDigit(firstWord)) {

                            addResult(Search.listNum, text);


                        } else if (Character.isAlphabetic(firstWord)) {
                            firstWord = Character.toUpperCase(firstWord);
                            if ('A' == firstWord) {
                                addResult(Search.listA, text);


                            } else if ('B' == firstWord) {

                                addResult(Search.listB, text);


                            } else if ('C' == firstWord) {

                                addResult(Search.listC, text);


                            } else if ('D' == firstWord) {

                                addResult(Search.listD, text);


                            } else if ('E' == firstWord) {
                                addResult(Search.listE, text);


                            } else if ('F' == firstWord) {

                                addResult(Search.listF, text);


                            } else if ('G' == firstWord) {

                                addResult(Search.listG, text);


                            } else if ('H' == firstWord) {

                                addResult(Search.listH, text);


                            } else if ('I' == firstWord) {

                                addResult(Search.listI, text);


                            } else if ('J' == firstWord) {
                                addResult(Search.listJ, text);


                            } else if ('K' == firstWord) {

                                addResult(Search.listK, text);


                            } else if ('L' == firstWord) {

                                addResult(Search.listL, text);


                            } else if ('M' == firstWord) {

                                addResult(Search.listM, text);


                            } else if ('N' == firstWord) {

                                addResult(Search.listN, text);


                            } else if ('O' == firstWord) {
                                addResult(Search.listO, text);


                            } else if ('P' == firstWord) {

                                addResult(Search.listP, text);


                            } else if ('Q' == firstWord) {

                                addResult(Search.listQ, text);


                            } else if ('R' == firstWord) {

                                addResult(Search.listR, text);


                            } else if ('S' == firstWord) {

                                addResult(Search.listS, text);


                            } else if ('T' == firstWord) {

                                addResult(Search.listT, text);


                            } else if ('U' == firstWord) {

                                addResult(Search.listU, text);


                            } else if ('V' == firstWord) {

                                addResult(Search.listV, text);

                            } else if ('W' == firstWord) {

                                addResult(Search.listW, text);

                            } else if ('X' == firstWord) {

                                addResult(Search.listX, text);


                            } else if ('Y' == firstWord) {

                                addResult(Search.listY, text);

                            } else if ('Z' == firstWord) {

                                addResult(Search.listZ, text);

                            }
                        }
                    }
                }else{
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
                                //未到最下层
                                if (0 == labelCount) {
                                    label1.setBackground(new Color(255, 125, 29, 255));
                                    label2.setBackground(null);
                                    label3.setBackground(null);
                                    label4.setBackground(null);
                                    showResult();
                                } else if (1 == labelCount) {
                                    label1.setBackground(null);
                                    label2.setBackground(new Color(255, 125, 29, 255));
                                    label3.setBackground(null);
                                    label4.setBackground(null);
                                    showResult();
                                } else if (2 == labelCount) {
                                    label1.setBackground(null);
                                    label2.setBackground(null);
                                    label3.setBackground(new Color(255, 125, 29, 255));
                                    label4.setBackground(null);
                                    showResult();
                                }
                            } else {
                                //到达最下层
                                label1.setBackground(null);
                                label2.setBackground(null);
                                label3.setBackground(null);
                                label4.setBackground(new Color(255, 125, 29, 255));
                                String path = listResult.get(labelCount - 3);
                                String name = getFileName(listResult.get(labelCount - 3));
                                if (name.indexOf(':') != -1) {
                                    name = name.substring(name.indexOf(':') + 2);
                                }
                                ImageIcon icon;
                                if (new File(path).isDirectory() || new File(path).isFile()) {
                                    icon = (ImageIcon) getSmallIcon(new File(path));
                                    icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                    label1.setIcon(icon);
                                    label1.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                } else {
                                    label1.setIcon(null);
                                    label1.setText("无效文件");
                                }
                                path = listResult.get(labelCount - 2);
                                name = getFileName(listResult.get(labelCount - 2));
                                if (name.indexOf(':') != -1) {
                                    name = name.substring(name.indexOf(':') + 2);
                                }
                                if (new File(path).isDirectory() || new File(path).isFile()) {
                                    icon = (ImageIcon) getSmallIcon(new File(path));
                                    icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                    label2.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                    label2.setIcon(icon);
                                } else {
                                    label2.setIcon(null);
                                    label2.setText("无效文件");
                                }
                                path = listResult.get(labelCount - 1);
                                name = getFileName(listResult.get(labelCount - 1));
                                if (name.indexOf(':') != -1) {
                                    name = name.substring(name.indexOf(':') + 2);
                                }
                                if (new File(path).isDirectory() || new File(path).isFile()) {
                                    icon = (ImageIcon) getSmallIcon(new File(path));
                                    icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                    label3.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                    label3.setIcon(icon);
                                } else {
                                    label3.setIcon(null);
                                    label3.setText("无效文件");
                                }
                                path = listResult.get(labelCount);
                                name = getFileName(listResult.get(labelCount));
                                if (name.indexOf(':') != -1) {
                                    name = name.substring(name.indexOf(':') + 2);
                                }
                                if (new File(path).isDirectory() || new File(path).isFile()) {
                                    icon = (ImageIcon) getSmallIcon(new File(path));
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
                                //未到最下层
                                if (0 == labelCount) {
                                    label1.setBackground(new Color(255, 125, 29, 255));
                                    label2.setBackground(null);
                                    label3.setBackground(null);
                                    label4.setBackground(null);
                                    showResult();
                                } else if (1 == labelCount) {
                                    label1.setBackground(null);
                                    label2.setBackground(new Color(255, 125, 29, 255));
                                    label3.setBackground(null);
                                    label4.setBackground(null);
                                    showResult();
                                } else if (2 == labelCount) {
                                    label1.setBackground(null);
                                    label2.setBackground(null);
                                    label3.setBackground(new Color(255, 125, 29, 255));
                                    label4.setBackground(null);
                                    showResult();
                                }
                            } else {
                                //到达最下层
                                label1.setBackground(null);
                                label2.setBackground(null);
                                label3.setBackground(null);
                                label4.setBackground(new Color(255, 125, 29, 255));
                                String path = listResult.get(labelCount - 3);
                                String name = getFileName(listResult.get(labelCount - 3));
                                if (name.indexOf(':') != -1) {
                                    name = name.substring(name.indexOf(':') + 2);
                                }
                                ImageIcon icon;
                                if (new File(path).isDirectory() || new File(path).isFile()) {
                                    icon = (ImageIcon) getSmallIcon(new File(path));
                                    icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                    label1.setIcon(icon);
                                    label1.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                } else {
                                    label1.setText("无效文件");
                                }
                                path = listResult.get(labelCount - 2);
                                name = getFileName(listResult.get(labelCount - 2));
                                if (name.indexOf(':') != -1) {
                                    name = name.substring(name.indexOf(':') + 2);
                                }
                                if (new File(path).isDirectory() || new File(path).isFile()) {
                                    icon = (ImageIcon) getSmallIcon(new File(path));
                                    icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                    label2.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                    label2.setIcon(icon);
                                } else {
                                    label2.setText("无效文件");
                                }
                                path = listResult.get(labelCount - 1);
                                name = getFileName(listResult.get(labelCount - 1));
                                if (name.indexOf(':') != -1) {
                                    name = name.substring(name.indexOf(':') + 2);
                                }
                                if (new File(path).isDirectory() || new File(path).isFile()) {
                                    icon = (ImageIcon) getSmallIcon(new File(path));
                                    icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                                    label3.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                                    label3.setIcon(icon);
                                } else {
                                    label3.setText("无效文件");
                                }
                                path = listResult.get(labelCount);
                                name = getFileName(listResult.get(labelCount));
                                if (name.indexOf(':') != -1) {
                                    name = name.substring(name.indexOf(':') + 2);
                                }
                                if (new File(path).isDirectory() || new File(path).isFile()) {
                                    icon = (ImageIcon) getSmallIcon(new File(path));
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

        label1.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label1.setLocation(0, (int) (searchBarHeight * 0.2));
        label1.setFont(new Font("微软雅黑", Font.BOLD, (int) ((height * 0.05) / 96 * 72 * 0.5)));
        label1.setForeground(new Color(73, 162, 255, 255));
        label1.setBackground(null);
        label1.setOpaque(true);


        label2.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label2.setLocation(0, (int) (searchBarHeight * 0.4));
        label2.setFont(new Font("微软雅黑", Font.BOLD, (int) ((height * 0.05) / 96 * 72 * 0.5)));
        label2.setForeground(new Color(73, 162, 255, 255));
        label2.setBackground(null);
        label2.setOpaque(true);


        label3.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label3.setLocation(0, (int) (searchBarHeight * 0.6));
        label3.setFont(new Font("微软雅黑", Font.BOLD, (int) ((height * 0.05) / 96 * 72 * 0.5)));
        label3.setForeground(new Color(73, 162, 255, 255));
        label3.setBackground(null);
        label3.setOpaque(true);


        label4.setSize(searchBarWidth, (int) (searchBarHeight * 0.2));
        label4.setLocation(0, (int) (searchBarHeight * 0.8));
        label4.setFont(new Font("微软雅黑", Font.BOLD, (int) ((height * 0.05) / 96 * 72 * 0.5)));
        label4.setForeground(new Color(73, 162, 255, 255));
        label4.setBackground(null);
        label4.setOpaque(true);


        //panel
        panel.setLayout(null);
        panel.setBackground(null);
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

    public void showSearchbar() {
        textField.grabFocus();
        searchBar.setVisible(true);
    }

    private void showResult() {
        try {
            String path = listResult.get(0);
            String name = getFileName(listResult.get(0));
            if (name.indexOf(':') != -1) {
                name = name.substring(name.indexOf(':') + 2);
            }
            ImageIcon icon;
            if (new File(path).isDirectory() || new File(path).isFile()) {
                icon = (ImageIcon) getSmallIcon(new File(path));
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label1.setIcon(icon);
                label1.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
            }else{
                label1.setIcon(null);
                label1.setText("无效文件");
            }
            path = listResult.get(1);
            name = getFileName(listResult.get(1));
            if (name.indexOf(':') != -1) {
                name = name.substring(name.indexOf(':') + 2);
            }
            if (new File(path).isDirectory() || new File(path).isFile()) {
                icon = (ImageIcon) getSmallIcon(new File(path));
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label2.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                label2.setIcon(icon);
            }else{
                label2.setIcon(null);
                label2.setText("无效文件");
            }
            path = listResult.get(2);
            name = getFileName(listResult.get(2));
            if (name.indexOf(':') != -1) {
                name = name.substring(name.indexOf(':') + 2);
            }
            if (new File(path).isDirectory() || new File(path).isFile()) {
                icon = (ImageIcon) getSmallIcon(new File(path));
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label3.setIcon(icon);
                label3.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
            }else{
                label3.setIcon(null);
                label3.setText("无效文件");
            }
            path = listResult.get(3);
            name = getFileName(listResult.get(3));
            if (name.indexOf(':') != -1) {
                name = name.substring(name.indexOf(':') + 2);
            }
            if (new File(path).isDirectory() || new File(path).isFile()) {
                icon = (ImageIcon) getSmallIcon(new File(path));
                icon = changeIcon(icon, label1.getHeight() - 60, label1.getHeight() - 60);
                label4.setText("<html><body>" + name + "<br>" + ">>>" + path + "</body></html>");
                label4.setIcon(icon);
            }else{
                label4.setIcon(null);
                label4.setText("无效文件");
            }
        } catch (java.lang.IndexOutOfBoundsException e) {
            //e.printStackTrace();
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
                Desktop desktop = null;
                if (Desktop.isDesktopSupported()) {
                    desktop = Desktop.getDesktop();
                }
                assert desktop != null;
                desktop.open(name);
            } catch (IOException e) {
                //打开上级文件夹
                try {
                    Runtime.getRuntime().exec("explorer.exe /select, \"" + name.getAbsolutePath() + "\"");
                } catch (IOException ex) {
                    //ex.printStackTrace();
                }
            }
        }
    }

    private String getFileName(String path) {
        String placeHolder = "";
        String[] name;
        try {
            name = path.split("\\\\");
            return name[name.length - 1];
        } catch (Exception e) {
            System.err.println(path);
            return placeHolder;
        }
    }


    /**
     * 获取程序小图标
     *
     * @param f 文件路径
     * @return icon
     */
    private static Icon getSmallIcon(File f) {
        if (f != null && f.exists()) {
            FileSystemView fsv = FileSystemView.getFileSystemView();
            return (fsv.getSystemIcon(f));
        }
        return (null);
    }


    private ImageIcon changeIcon(ImageIcon icon, int width, int height) {
        try {
            Image image = icon.getImage().getScaledInstance(width, height, Image.SCALE_DEFAULT);
            return new ImageIcon(image);
        }catch (NullPointerException e){
            return null;
        }
    }

    private void saveCache(String content) {
        int cacheNum = 0;
        File cache = new File("cache.dat");
        StringBuilder oldCaches = new StringBuilder();
        boolean isRepeated;
        if (cache.exists()) {
            try (BufferedReader reader = new BufferedReader(new FileReader(cache))){
                String eachLine;
                while ((eachLine = reader.readLine()) != null) {
                    oldCaches.append(eachLine);
                    cacheNum++;
                }
            } catch (IOException e) {
                //e.printStackTrace();
            }
        }
        if (cacheNum < SettingsFrame.cacheNumLimit) {
            isRepeated = match(oldCaches.toString() + ";", (content));
            if (!isRepeated) {
                try(BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File("cache.dat"), true)))) {
                    out.write(content + "\r\n");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }


    public static void delRepeated(LinkedList<String> list) {
        LinkedHashSet<String> set = new LinkedHashSet<>();
        try {
            set.addAll(list);
        }catch (Exception ignored){

        }
        list.clear();
        list.addAll(set);
    }


    private void delCache(LinkedList<String> cache){
        File cacheFile = new File("cache.dat");
        StringBuilder allCaches = new StringBuilder();
        String eachLine;
        if (cacheFile.exists()){
            try(BufferedReader br = new BufferedReader(new FileReader(cacheFile))){
                while ((eachLine = br.readLine()) != null){
                    String[] each = eachLine.split(";");
                    for (String eachCache: each) {
                        if (!(cache.contains(eachCache))) {
                            allCaches.append(eachCache).append(";\n");
                        }
                    }
                }
            } catch (IOException ignored) {

            }
            try(BufferedWriter bw  = new BufferedWriter(new FileWriter(cacheFile))){
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
                        if (!(new File(cach).exists())){
                            cachesToDel.add(cach);
                        }else{
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
            if (srcText.length() == txt.length()) {
                return srcText.toUpperCase().equals(txt.toUpperCase());
            }
            while (srcText.length() > txt.length()) {
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


    public Container getPanel(){
        return this.panel;
    }
    private void clearTextFieldText() {
        Runnable clear = () -> textField.setText(null);
        SwingUtilities.invokeLater(clear);
    }
}
	


