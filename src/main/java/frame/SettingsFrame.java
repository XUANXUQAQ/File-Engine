package frame;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.*;
import java.net.URL;
import com.alibaba.fastjson.*;
import main.Main;
import moveFiles.*;
import search.Search;


public class SettingsFrame {
    private JTextField textField1;
    private JTextField textField2;
    private JTextArea textArea1;
    private JTextField textField4;
    private JCheckBox checkBox1;
    private JButton button1;
    private JLabel label3;
    private JLabel label1;
    private JLabel label2;
    JFrame frame = new JFrame("SettingsFrame");
    private Search searchObj = new Search();

    public void showWindow() {
        URL frameIcon = SettingsFrame.class.getResource("/icons/frame.png");
        frame.setIconImage(new ImageIcon(frameIcon).getImage());
        frame.setContentPane(new SettingsFrame().panel);
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        Dimension screenSize =Toolkit.getDefaultToolkit().getScreenSize(); // ��ȡ��ǰ�ֱ���
        int width = screenSize.width;
        int height = screenSize.height;
        frame.setLocation((int)(width*0.25), (int)(height*0.3));
        frame.setSize(800, 400);
        frame.setBackground(Color.white);
        panel.setBackground(Color.white);
        panel.setOpaque(true);
        frame.setBackground(Color.white);
        frame.setVisible(true);
    }

    private JLabel label4;
    private JLabel label5;
    private JLabel label6;
    private JPanel panel;
    private JLabel label7;
    private JButton button2;
    private JLabel labelPlaceHoder1;
    private JButton Button3;
    private JScrollPane scrollpane;
    private JLabel labelplaceholder2;
    private JLabel labelplacehoder3;
    private JLabel labelTip;
    private JTextField textField3;
    private boolean isStartup;
    public static int cacheNumLimit;
    public static String hotkey;
    private int updateTimeLimit;
    private String ignorePath;
    private static File settings = new File("settings.json");
    private static CheckHotKey HotKeyListener;

    public static void initSettings(){
        try(BufferedReader buffR = new BufferedReader(new FileReader(settings))) {
            String line;
            StringBuilder result = new StringBuilder();
            while (null != (line = buffR.readLine())){
                result.append(line);
            }
            JSONObject settings = JSON.parseObject(result.toString());
            cacheNumLimit = settings.getInteger("cacheNumLimit");
            hotkey = settings.getString("hotkey");
            HotKeyListener = new CheckHotKey();
        } catch (IOException ignored) {

        }
    }


    public SettingsFrame() {
        String lookAndFeel = "com.sun.java.swing.plaf.windows.WindowsLookAndFeel";
        try {
            UIManager.setLookAndFeel(lookAndFeel);
        } catch (ClassNotFoundException | UnsupportedLookAndFeelException | IllegalAccessException | InstantiationException ignored) {

        }
        //��ȡ�������ò�����checkBox״̬
        label1.setText("�ļ���������ʱ�䣺");
        label3.setText("���ú����ļ��У�");
        label4.setText("������󻺴�������");
        label6.setText("������ȣ�");
        button1.setText("����");
        button2.setText("���ݲ��Ƴ����������ļ�");
        textArea1.setLineWrap(true);
        textArea1.setWrapStyleWord(true);

        try(BufferedReader buffR = new BufferedReader(new FileReader(settings))) {
            String line;
            StringBuilder result = new StringBuilder();
            while (null != (line = buffR.readLine())){
                result.append(line);
            }
            JSONObject settings = JSON.parseObject(result.toString());
            isStartup = settings.getBoolean("isStartup");
            if (isStartup){
                checkBox1.setSelected(true);
            }else{
                checkBox1.setSelected(false);
            }
            updateTimeLimit = settings.getInteger("updateTimeLimit");
            String MaxUpdateTime = "";
            MaxUpdateTime = MaxUpdateTime + updateTimeLimit;
            textField1.setText(MaxUpdateTime);
            ignorePath = settings.getString("ignorePath");
            ignorePath = ignorePath.replaceAll(",", ",\n");
            textArea1.setText(ignorePath);
            cacheNumLimit = settings.getInteger("cacheNumLimit");
            String MaxCacheNum = "";
            MaxCacheNum = MaxCacheNum + cacheNumLimit;
            textField2.setText(MaxCacheNum);
            String searchDepth = "";
            int searchDepthInSettings = settings.getInteger("searchDepth");
            searchDepth = searchDepth + searchDepthInSettings;
            textField4.setText(searchDepth);
            textField3.setText(hotkey);
        } catch (IOException ignored) {

        }


        button1.addActionListener(e -> {
            JSONObject allSettings = new JSONObject();
            String MaxUpdateTime = textField1.getText();
            try {
                updateTimeLimit = Integer.parseInt(MaxUpdateTime);
            }catch (Exception e1){
                updateTimeLimit = -1; // ���벻��ȷ
            }
            if (updateTimeLimit > 3600 || updateTimeLimit <= 0){
                JOptionPane.showMessageDialog(null, "�ļ������������ô��������");
                return;
            }
            isStartup = checkBox1.isSelected();
            String MaxCacheNum = textField2.getText();
            try {
                cacheNumLimit = Integer.parseInt(MaxCacheNum);
            }catch (Exception e1){
                cacheNumLimit = -1;
            }
            if (cacheNumLimit > 10000 || cacheNumLimit <= 0){
                JOptionPane.showMessageDialog(null, "�����������ô��������");
                return;
            }
            ignorePath = textArea1.getText();
            ignorePath = ignorePath.replaceAll("\n", "");
            String[] paths = ignorePath.split(",");
            if (!ignorePath.equals("")) {
                if (ignorePath.contains("\\")){
                    JOptionPane.showMessageDialog(null, "�����ļ��и�ʽ���󣬸�ʽΪ   \nC:\\Windows,\nD\\Test,");
                    return;
                }
                for (String each : paths) {
                    if (each.charAt(0) == ' ' || each.charAt(each.length() - 1) == ' ') {
                        JOptionPane.showMessageDialog(null, "�����ļ��и�ʽ���󣬸�ʽΪ   \nC:\\Windows,\nD\\Test,");
                        return;
                    }
                }
            }
            int searchDepth;
            try {
                searchDepth = Integer.parseInt(textField4.getText());
            }catch (Exception e1){
                searchDepth = -1;
            }

            if (searchDepth > 10 || searchDepth <= 0){
                JOptionPane.showMessageDialog(null, "����������ô��������");
                return;
            }

            String _hotkey = textField3.getText();
            if (_hotkey.length() == 1){
                JOptionPane.showMessageDialog(null, "��ݼ����ô���");
                return;
            }else{
                if (!(64 < _hotkey.charAt(_hotkey.length()-1) && _hotkey.charAt(_hotkey.length()-1) < 91)){
                    JOptionPane.showMessageDialog(null, "��ݼ����ô���");
                    return;
                }
            }
            hotkey = textField3.getText();
            HotKeyListener.registHotkey(hotkey);
            Main.setIgnorePath(ignorePath);
            Main.setSearchDepth(searchDepth);
            Main.setUpdateTimeLimit(updateTimeLimit*1000); //ʹ������Ч
            allSettings.put("hotkey", hotkey);
            allSettings.put("isStartup", isStartup);
            allSettings.put("cacheNumLimit", cacheNumLimit);
            allSettings.put("updateTimeLimit", updateTimeLimit);
            allSettings.put("ignorePath", ignorePath);
            allSettings.put("searchDepth", searchDepth);
            try(BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
                buffW.write(allSettings.toJSONString());
            } catch (IOException ignored) {

            }
        });
        checkBox1.addActionListener(e -> {
            Process p;
            File superSearch = new File("search_x64.exe"); //TODO �޸İ汾
            if (checkBox1.isSelected()){
                try {
                    p = Runtime.getRuntime().exec("reg add \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\" /v superSearch /t REG_SZ /d "+"\"" + superSearch.getAbsolutePath() +"\"" + " /f");
                    p.waitFor();
                    BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                    String line;
                    StringBuilder result = new StringBuilder();
                    while ((line = outPut.readLine()) != null){
                        result.append(line);
                    }
                    if (!result.toString().equals("")){
                        checkBox1.setSelected(false);
                        JOptionPane.showMessageDialog(null, "��ӵ���������ʧ�ܣ��볢���Թ���Ա�������");
                    }else{
                        Runtime.getRuntime().exec("reg add \"HKEY_CURRENT_USER\\Software\\Microsoft\\Windows NT\\CurrentVersion\\AppCompatFlags\\Layers\" /v " + "\""+superSearch.getAbsolutePath()+ "\"" + " /t REG_SZ /d RUNASADMIN /f");
                    }
                } catch (IOException | InterruptedException ignored) {

                }
            }else{
                try {
                    p = Runtime.getRuntime().exec("reg delete \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\" /v superSearch /f");
                    p.waitFor();
                    BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                    String line;
                    StringBuilder result = new StringBuilder();
                    while ((line = outPut.readLine()) != null){
                        result.append(line);
                    }
                    if (!result.toString().equals("")){
                        checkBox1.setSelected(true);
                        JOptionPane.showMessageDialog(null, "ɾ����������ʧ�ܣ��볢���Թ���Ա�������");
                    }else{
                        Runtime.getRuntime().exec("reg delete \"HKEY_CURRENT_USER\\Software\\Microsoft\\Windows NT\\CurrentVersion\\AppCompatFlags\\Layers\" /v " + "\""+superSearch.getAbsolutePath()+ "\" /f");
                    }
                } catch (IOException | InterruptedException ignored) {

                }
            }
            boolean isSelected = checkBox1.isSelected();
            JSONObject allSettings = null;
            try(BufferedReader buffR1 = new BufferedReader(new FileReader(settings))){
                String line;
                StringBuilder result = new StringBuilder();
                while (null != (line = buffR1.readLine())){
                    result.append(line);
                }
                allSettings = JSON.parseObject(result.toString());
                allSettings.put("isStartup", isSelected);
            } catch (IOException ignored) {

            }
            try(BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))){
                assert allSettings != null;
                buffW.write(allSettings.toJSONString());
            } catch (IOException ignored) {

            }
        });
        button2.addActionListener(e -> {
            String currentFolder = new File("").getAbsolutePath();
            if (currentFolder.equals(FileSystemView.getFileSystemView().getHomeDirectory().getAbsolutePath()) || currentFolder.equals("C:\\Users\\Public\\Desktop")){
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
                textArea1.append(file.getAbsolutePath() + ",\n");
            }
        });
        textField3.addKeyListener(new KeyListener() {
            boolean reset = false;

            @Override
            public void keyTyped(KeyEvent e) {

            }

            @Override
            public void keyPressed(KeyEvent e) {
                int key = e.getKeyCode();
                if (reset){
                    textField3.setText(null);
                    reset = false;
                }
                textField3.setCaretPosition(textField3.getText().length());
                if (key == 17) {
                    if (!textField3.getText().contains("Ctrl + ")) {
                        textField3.setText(textField3.getText() + "Ctrl + ");
                    }
                } else if (key == 18) {
                    if (!textField3.getText().contains("Alt + ")) {
                        textField3.setText(textField3.getText() + "Alt + ");
                    }
                } else if (key == 524) {
                    if (!textField3.getText().contains("Win + ")) {
                        textField3.setText(textField3.getText() + "Win + ");
                    }
                } else if (key == 16) {
                    if (!textField3.getText().contains("Shift + ")) {
                        textField3.setText(textField3.getText() + "Shift + ");
                    }
                } else if (64 < key && key < 91) {
                    String txt = textField3.getText();
                    if (!txt.equals("")) {
                        if (64 < txt.charAt(txt.length() - 1) && txt.charAt(txt.length() - 1) < 91) {
                            String text = txt.substring(0, txt.length() - 1);
                            textField3.setText(text + (char) key);
                        } else {
                            textField3.setText(txt + (char) key);
                        }
                    }
                    if (txt.length() == 1){
                        textField3.setText(null);
                    }
                }
            }

            @Override
            public void keyReleased(KeyEvent e) {
                reset = true;
            }
        });
    }
    class moveDesktopFiles implements Runnable{

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