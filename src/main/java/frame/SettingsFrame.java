package frame;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.*;
import java.net.URL;
import com.alibaba.fastjson.*;
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


    public void showWindow() {
        URL frameIcon = SettingsFrame.class.getResource("/icons/frame.png");
        frame.setIconImage(new ImageIcon(frameIcon).getImage());
        frame.setContentPane(new SettingsFrame().panel);
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        Dimension screenSize =Toolkit.getDefaultToolkit().getScreenSize(); // 获取当前分辨率
        int width = screenSize.width;
        int height = screenSize.height;
        frame.setLocation((int)(width*0.25), (int)(height*0.3));
        frame.setSize(800, 400);
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
    private boolean isStartup;
    public static int cacheNumLimit;
    private int updateTimeLimit;
    private String ignorePath;
    private static File settings = new File("settings.json");

    public static void initCacheLimit(){
        try(BufferedReader buffR = new BufferedReader(new FileReader(settings))) {
            String line;
            StringBuilder result = new StringBuilder();
            while (null != (line = buffR.readLine())){
                result.append(line);
            }
            JSONObject settings = JSON.parseObject(result.toString());
            cacheNumLimit = settings.getInteger("cacheNumLimit");
        } catch (IOException e) {
            //e.printStackTrace();
        }
    }


    public SettingsFrame() {
        String lookAndFeel = "com.sun.java.swing.plaf.windows.WindowsLookAndFeel";
        try {
            UIManager.setLookAndFeel(lookAndFeel);
        } catch (ClassNotFoundException | UnsupportedLookAndFeelException | IllegalAccessException | InstantiationException e) {
            //e.printStackTrace();
        }
        //读取所有设置并设置checkBox状态
        label1.setText("文件索引更新时间：");
        label2.setText("快捷键：CTRL+ALT+J");
        label3.setText("设置忽略文件夹：");
        label4.setText("设置最大缓存数量：");
        label6.setText("搜索深度：");
        button1.setText("保存");
        button2.setText("备份并移除所有桌面文件");
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
        } catch (IOException e) {
            //e.printStackTrace();
        }


        button1.addActionListener(e -> {
            JSONObject allSettings = new JSONObject();
            String MaxUpdateTime = textField1.getText();
            try {
                updateTimeLimit = Integer.parseInt(MaxUpdateTime);
            }catch (Exception e1){
                updateTimeLimit = -1; // 输入不正确
            }
            if (updateTimeLimit > 3600 || updateTimeLimit <= 0){
                JOptionPane.showMessageDialog(null, "文件索引更新设置错误，请更改");
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
                JOptionPane.showMessageDialog(null, "缓存容量设置错误，请更改");
                return;
            }
            ignorePath = textArea1.getText();
            ignorePath = ignorePath.replaceAll("\n", "");
            String[] paths = ignorePath.split(",");
            if (!ignorePath.equals("")) {
                for (String each : paths) {
                    if (each.charAt(0) == ' ' || each.charAt(each.length() - 1) == ' ') {
                        JOptionPane.showMessageDialog(null, "忽略文件夹格式错误，格式为   \nC:\\Windows,\nD\\Test,");
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
                JOptionPane.showMessageDialog(null, "搜索深度设置错误，请更改");
                return;
            }
            allSettings.put("isStartup", isStartup);
            allSettings.put("cacheNumLimit", cacheNumLimit);
            allSettings.put("updateTimeLimit", updateTimeLimit);
            allSettings.put("ignorePath", ignorePath);
            allSettings.put("searchDepth", searchDepth);
            try(BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
                buffW.write(allSettings.toJSONString());
            } catch (IOException ex) {
                //ex.printStackTrace();
            }
        });
        checkBox1.addActionListener(e -> {
            Process p;
            if (checkBox1.isSelected()){
                try {
                    File superSearch = new File("search_x64.exe"); //TODO 更改版本号
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
                        JOptionPane.showMessageDialog(null, "添加到开机启动失败，请尝试以管理员身份运行");
                    }
                } catch (IOException | InterruptedException ex) {
                    //ex.printStackTrace();
                }
            }else{
                try {
                    p = Runtime.getRuntime().exec("reg delete \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\" /v superSearch /f");
                    BufferedReader outPut = new BufferedReader(new InputStreamReader(p.getErrorStream()));
                    String line;
                    StringBuilder result = new StringBuilder();
                    while ((line = outPut.readLine()) != null){
                        result.append(line);
                    }
                    if (!result.toString().equals("")){
                        checkBox1.setSelected(true);
                        JOptionPane.showMessageDialog(null, "删除开机启动失败，请尝试以管理员身份运行");
                    }
                } catch (IOException ex) {
                    //ex.printStackTrace();
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
            } catch (IOException ex) {
                //ex.printStackTrace();
            }
            try(BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))){
                assert allSettings != null;
                buffW.write(allSettings.toJSONString());
            } catch (IOException ex) {
                //ex.printStackTrace();
            }
        });
        button2.addActionListener(e -> {
            String currentFolder = new File("").getAbsolutePath();
            if (currentFolder.equals(Search.desktop) || currentFolder.equals("C:\\Users\\Public\\Desktop")){
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
            fileChooser.showDialog(new JLabel(), "选择");
            File file = fileChooser.getSelectedFile();
            if (file != null) {
                textArea1.append(file.getAbsolutePath() + ",\n");
            }
        });
    }
    static class moveDesktopFiles implements Runnable{

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
            new Search().setSearch(true);
        }
    }
}