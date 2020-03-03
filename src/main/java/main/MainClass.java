package main;

import com.alee.laf.WebLookAndFeel;
import com.alibaba.fastjson.JSONObject;
import fileMonitor.FileMonitor;
import frames.SearchBar;
import frames.SettingsFrame;
import frames.TaskBar;
import hotkeyListener.CheckHotKey;
import search.Search;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.*;
import java.util.Objects;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainClass {
    public static final String version = "1.6"; //TODO ���İ汾��
    public static boolean mainExit = false;
    public static String name;
    private static Search search = Search.getInstance();
    private static TaskBar taskBar = null;

    public static void setMainExit(boolean b) {
        mainExit = b;
    }

    public static void showMessage(String caption, String message) {
        if (taskBar != null) {
            taskBar.showMessage(caption, message);
        }
    }

    public static boolean isAdmin() {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("cmd.exe");
            Process process = processBuilder.start();
            PrintStream printStream = new PrintStream(process.getOutputStream(), true);
            Scanner scanner = new Scanner(process.getInputStream());
            printStream.println("@echo off");
            printStream.println(">nul 2>&1 \"%SYSTEMROOT%\\system32\\cacls.exe\" \"%SYSTEMROOT%\\system32\\config\\system\"");
            printStream.println("echo %errorlevel%");

            boolean printedErrorlevel = false;
            while (true) {
                String nextLine = scanner.nextLine();
                if (printedErrorlevel) {
                    int errorlevel = Integer.parseInt(nextLine);
                    return errorlevel == 0;
                } else if (nextLine.equals("echo %errorlevel%")) {
                    printedErrorlevel = true;
                }
            }
        } catch (IOException e) {
            return false;
        }
    }

    public static void copyFile(InputStream source, File dest) {
        try (OutputStream os = new FileOutputStream(dest); BufferedInputStream bis = new BufferedInputStream(source); BufferedOutputStream bos = new BufferedOutputStream(os)) {
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

    public static void deleteDir(String path) {
        File file = new File(path);
        if (!file.exists()) {//�ж��Ƿ��ɾ��Ŀ¼�Ƿ����
            return;
        }

        String[] content = file.list();//ȡ�õ�ǰĿ¼�������ļ����ļ���
        if (content != null) {
            for (String name : content) {
                File temp = new File(path, name);
                if (temp.isDirectory()) {//�ж��Ƿ���Ŀ¼
                    deleteDir(temp.getAbsolutePath());//�ݹ���ã�ɾ��Ŀ¼�������
                    temp.delete();//ɾ����Ŀ¼
                } else {
                    if (!temp.delete()) {//ֱ��ɾ���ļ�
                        System.err.println("Failed to delete " + name);
                    }
                }
            }
        }
    }


    public static void main(String[] args) {
        try {
            System.setProperty("sun.java2d.noddraw", "true");
            System.setProperty("jna.library.path", new File("user").getAbsolutePath() + "\\");
            org.jb2011.lnf.beautyeye.BeautyEyeLNFHelper.launchBeautyEyeLNF();
            UIManager.put("RootPane.setupButtonVisible", false);
            WebLookAndFeel.initializeManagers();
        } catch (Exception e) {
            e.printStackTrace();
        }
        String osArch = System.getProperty("os.arch");
        if (osArch.contains("64")) {
            name = "File-Engine-x64.exe";
        } else {
            name = "File-Engine-x86.exe";
        }

        File user = new File("user");
        if (!user.exists()) {
            user.mkdir();
        }
        File[] userFiles = user.listFiles();
        boolean isUpdate = false;
        if (userFiles != null) {
            for (File each : userFiles) {
                if (each.getName().equals("update")) {
                    isUpdate = true;
                    break;
                }
            }
        }
        if (isUpdate) {
            boolean isCopied = false;
            String currentPath = System.getProperty("user.dir");
            File[] files = new File(currentPath).listFiles();
            assert files != null;
            for (File i : files) {
                if (i.getName().startsWith("_File-Engine")) {
                    isCopied = true;
                    new File("user/update").delete();
                }
            }
            if (!isCopied) {
                try {
                    new File("user/fileMonitor.dll").delete();
                    new File("user/fileSearcher.exe").delete();
                    new File("user/getAscII.dll").delete();
                    new File("user/fileOpener.exe").delete();
                    new File("user/hotkeyListener.dll").delete();
                    File originFile = new File(name);
                    File updated = new File("_" + name);
                    copyFile(new FileInputStream(originFile), updated);
                    Desktop desktop;
                    if (Desktop.isDesktopSupported()) {
                        desktop = Desktop.getDesktop();
                        //���Ʋ��������Լ����˳����򿪸��ƺ�ĳ���
                        desktop.open(updated);
                        System.exit(0);
                    }
                } catch (Exception ignored) {

                }
            } else {
                try {
                    copyFile(new FileInputStream(new File("tmp\\" + name)), new File(name));
                    Desktop desktop;
                    if (Desktop.isDesktopSupported()) {
                        desktop = Desktop.getDesktop();
                        //���Ʋ��������Լ����˳����򿪸��ƺ�ĳ���
                        desktop.open(new File(name));
                        System.exit(0);
                    }
                } catch (IOException ignored) {

                }
            }
        } else {
            File pre = new File("_" + name);
            if (pre.exists()) {
                int count = 0;
                while (count < 500) {
                    count++;
                    try {
                        pre.delete();
                        Thread.sleep(10);
                    } catch (Exception ignored) {

                    }
                }
            }
        }

        File settings = SettingsFrame.settings;
        File caches = new File("user/cache.dat");
        File data = new File("data");

        if (!settings.exists()) {
            String ignorePath = "C:\\Windows,";
            JSONObject json = new JSONObject();
            json.put("hotkey", "Ctrl + Alt + J");
            json.put("ignorePath", ignorePath);
            json.put("isStartup", false);
            json.put("updateTimeLimit", 5);
            json.put("cacheNumLimit", 1000);
            json.put("searchDepth", 6);
            json.put("priorityFolder", "");
            json.put("dataPath", data.getAbsolutePath());
            json.put("isDefaultAdmin", false);
            json.put("isLoseFocusClose", true);
            json.put("openLastFolderKeyCode", 17);
            json.put("runAsAdminKeyCode", 16);
            json.put("copyPathKeyCode", 18);
            try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
                buffW.write(json.toJSONString());
            } catch (IOException ignored) {

            }
        }


        //���tmp
        MainClass.deleteDir(SettingsFrame.tmp.getAbsolutePath());

        File target;
        InputStream fileSearcher64 = MainClass.class.getResourceAsStream("/fileSearcher64.exe");
        InputStream fileSearcher86 = MainClass.class.getResourceAsStream("/fileSearcher86.exe");
        InputStream fileOpener64 = MainClass.class.getResourceAsStream("/fileOpener64.exe");
        InputStream fileOpener86 = MainClass.class.getResourceAsStream("/fileOpener86.exe");
        InputStream fileMonitor64Dll = MainClass.class.getResourceAsStream("/win32-x86-64/fileMonitor64.dll");
        InputStream fileMonitor86Dll = MainClass.class.getResourceAsStream("/win32-x86/fileMonitor86.dll");
        InputStream getAscII64Dll = MainClass.class.getResourceAsStream("/win32-x86-64/getAscII64.dll");
        InputStream getAscII86Dll = MainClass.class.getResourceAsStream("/win32-x86/getAscII86.dll");
        InputStream hotkeyListener64Dll = MainClass.class.getResourceAsStream("/win32-x86-64/hotkeyListener64.dll");
        InputStream hotkeyListener86Dll = MainClass.class.getResourceAsStream("/win32-x86/hotkeyListener86.dll");
        InputStream everything64Dll = MainClass.class.getResourceAsStream("/win32-x86-64/Everything64.dll");
        InputStream everything86Dll = MainClass.class.getResourceAsStream("/win32-x86/Everything32.dll");

        boolean is64Bit = name.contains("x64");

        if (is64Bit) {
            target = new File("user/fileMonitor.dll");
            if (!target.exists()) {
                copyFile(fileMonitor64Dll, target);
            }
            target = new File("user/getAscII.dll");
            if (!target.exists()) {
                copyFile(getAscII64Dll, target);
            }
            target = new File("user/hotkeyListener.dll");
            if (!target.exists()) {
                copyFile(hotkeyListener64Dll, target);
            }
            target = new File("user/Everything.dll");
            if (!target.exists()) {
                copyFile(everything64Dll, target);
            }
        } else {
            target = new File("user/fileMonitor.dll");
            if (!target.exists()) {
                copyFile(fileMonitor86Dll, target);
            }
            target = new File("user/getAscII.dll");
            if (!target.exists()) {
                copyFile(getAscII86Dll, target);
            }
            target = new File("user/hotkeyListener.dll");
            if (!target.exists()) {
                copyFile(hotkeyListener86Dll, target);
            }
            target = new File("user/Everything.dll");
            if (!target.exists()) {
                copyFile(everything86Dll, target);
            }
        }
        SettingsFrame.initSettings();
        SearchBar searchBar = SearchBar.getInstance();

        target = new File("user/fileSearcher.exe");
        if (!target.exists()) {
            if (is64Bit) {
                copyFile(fileSearcher64, target);
                System.out.println("�Ѽ���64λfileSearcher");
            } else {
                copyFile(fileSearcher86, target);
                System.out.println("�Ѽ���32λfileSearcher");
            }
        }
        target = new File(SettingsFrame.dataPath);
        if (!target.exists()) {
            target.mkdir();
        }
        target = new File("user/fileOpener.exe");
        if (!target.exists()) {
            if (is64Bit) {
                copyFile(fileOpener64, target);
                System.out.println("�Ѽ���64λfileOpener");
            } else {
                copyFile(fileOpener86, target);
                System.out.println("�Ѽ���32λfileOpener");
            }
        }

        if (!caches.exists()) {
            try {
                caches.createNewFile();
            } catch (IOException e) {
                JOptionPane.showMessageDialog(null, "���������ļ�ʧ�ܣ����������˳�");
                mainExit = true;
            }
        }
        if (!SettingsFrame.tmp.exists()) {
            SettingsFrame.tmp.mkdir();
        }
        File fileAdded = new File(SettingsFrame.tmp.getAbsolutePath() + "\\fileAdded.txt");
        File fileRemoved = new File(SettingsFrame.tmp.getAbsolutePath() + "\\fileRemoved.txt");
        if (!fileAdded.exists()) {
            try {
                fileAdded.createNewFile();
            } catch (IOException ignored) {

            }
        }
        if (!fileRemoved.exists()) {
            try {
                fileRemoved.createNewFile();
            } catch (IOException ignored) {

            }
        }
        taskBar = new TaskBar();
        taskBar.showTaskBar();


        File[] roots = File.listRoots();
        ExecutorService fixedThreadPool = Executors.newFixedThreadPool(roots.length + 4);


        data = new File(SettingsFrame.dataPath);
        if (data.isDirectory() && data.exists()) {
            if (Objects.requireNonNull(data.listFiles()).length != 26) {
                System.out.println("��⵽data�ļ��𻵣������������ؽ�");
                search.setManualUpdate(true);
            }
        } else {
            System.out.println("��data�ļ��������������ؽ�");
            search.setManualUpdate(true);
        }

        FileSystemView sys = FileSystemView.getFileSystemView();
        if (isAdmin()) {
            for (File root : roots) {
                String dirveType = sys.getSystemTypeDescription(root);
                if (dirveType.equals("���ش���")) {
                    fixedThreadPool.execute(() -> FileMonitor.INSTANCE.monitor(root.getAbsolutePath(), SettingsFrame.tmp.getAbsolutePath(), SettingsFrame.tmp.getAbsolutePath() + "\\CLOSE"));
                }
            }
        } else {
            System.out.println("�޹���ԱȨ�ޣ��ļ���ع����ѹر�");
            taskBar.showMessage("����", "�޹���ԱȨ�ޣ��ļ���ع����ѹر�");
        }

        //����Ƿ�Ϊ���°汾
        try {
            JSONObject info = SettingsFrame.getInfo();
            String latestVersion = info.getString("version");
            if (Double.parseDouble(latestVersion) > Double.parseDouble(version)) {
                showMessage("��ʾ", "���°汾�ɸ���");
            }
        } catch (IOException ignored) {

        }


        fixedThreadPool.execute(() -> {
            //����ļ��Ķ��߳�
            String filesToAdd;
            String filesToRemove;
            BufferedReader readerAdd = null;
            BufferedReader readerRemove = null;
            try {
                readerAdd = new BufferedReader(new InputStreamReader(new FileInputStream(new File(SettingsFrame.tmp.getAbsolutePath() + "\\fileAdded.txt"))));
                readerRemove = new BufferedReader(new InputStreamReader(new FileInputStream(new File(SettingsFrame.tmp.getAbsolutePath() + "\\fileRemoved.txt"))));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            //�ָ��ַ���
            while (!mainExit) {
                if (!search.isManualUpdate()) {
                    try {
                        if (readerAdd != null) {
                            while ((filesToAdd = readerAdd.readLine()) != null) {
                                if (!filesToAdd.contains(SettingsFrame.dataPath)) {
                                    search.addFileToLoadBin(filesToAdd);
                                    System.out.println("���" + filesToAdd);
                                }
                            }
                        }
                        if (readerRemove != null) {
                            while ((filesToRemove = readerRemove.readLine()) != null) {
                                if (!filesToRemove.contains(SettingsFrame.dataPath)) {
                                    search.addToRecycleBin(filesToRemove);
                                }
                            }
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                try {
                    Thread.sleep(50);
                } catch (InterruptedException ignored) {

                }
            }
            if (readerAdd != null) {
                try {
                    readerAdd.close();
                } catch (IOException ignored) {

                }
            }
            if (readerRemove != null) {
                try {
                    readerRemove.close();
                } catch (IOException ignored) {

                }
            }
        });


        fixedThreadPool.execute(() -> {
            // ʱ�����߳�
            long count = 0;
            long gcCount = 0;
            while (!mainExit) {
                boolean isUsing = searchBar.isUsing();
                count++;
                gcCount++;
                if (count >= (SettingsFrame.updateTimeLimit << 10) && !isUsing && !search.isManualUpdate()) {
                    count = 0;
                    if (search.isUsable() && (!searchBar.isUsing())) {
                        search.mergeFileToList();
                    }
                }
                if (gcCount > 600000) {
                    System.out.println("�����ͷ��ڴ�ռ�");
                    System.gc();
                    gcCount = 0;
                }
                try {
                    Thread.sleep(1);
                } catch (InterruptedException ignore) {

                }
            }
        });


        //�����߳�
        fixedThreadPool.execute(() -> {
            while (!mainExit) {
                if (search.isManualUpdate()) {
                    search.setUsable(false);
                    search.updateLists(SettingsFrame.ignorePath, SettingsFrame.searchDepth);
                }
                try {
                    Thread.sleep(1);
                } catch (InterruptedException ignored) {

                }
            }
        });

        while (true) {
            // ��ѭ����ʼ
            try {
                Thread.sleep(1);
            } catch (InterruptedException ignored) {

            }
            if (mainExit) {
                CheckHotKey.getInstance().stopListen();
                File CLOSEDLL = new File(SettingsFrame.tmp.getAbsolutePath() + "\\CLOSE");
                try {
                    CLOSEDLL.createNewFile();
                } catch (IOException ignored) {

                }
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException ignored) {

                }
                fixedThreadPool.shutdown();
                System.exit(0);
            }
        }
    }
}
