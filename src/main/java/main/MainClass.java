package main;

import com.alibaba.fastjson.JSONObject;
import fileMonitor.FileMonitor;
import frame.SearchBar;
import frame.SettingsFrame;
import frame.TaskBar;
import search.Search;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.io.*;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainClass {
    public static String version = "2.6"; //TODO ���İ汾��
    public static boolean mainExit = false;
    public static String name;
    private static Search search = new Search();
    private static SearchBar searchBar = SearchBar.getInstance();
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

    private static void copyFile(InputStream source, File dest) {
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
            org.jb2011.lnf.beautyeye.BeautyEyeLNFHelper.launchBeautyEyeLNF();
            UIManager.put("RootPane.setupButtonVisible",false);
        } catch (Exception e) {
            e.printStackTrace();
        }
        String osArch = System.getProperty("os.arch");
        if (osArch.contains("64")) {
            name = "search_x64.exe";
        } else {
            name = "search_x86.exe";
        }
        File settings = new File(System.getenv("Appdata") + "/settings.json");
        File caches = new File("cache.dat");
        File data = new File("data");
        //���tmp
        deleteDir(SettingsFrame.tmp.getAbsolutePath());
        if (!settings.exists()) {
            String ignorePath = "";
            JSONObject json = new JSONObject();
            json.put("hotkey", "Ctrl + Alt + J");
            json.put("ignorePath", ignorePath);
            json.put("isStartup", false);
            json.put("updateTimeLimit", 300);
            json.put("cacheNumLimit", 1000);
            json.put("searchDepth", 6);
            json.put("priorityFolder", "");
            json.put("dataPath", data.getAbsolutePath());
            try (BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
                buffW.write(json.toJSONString());
            } catch (IOException ignored) {

            }
        }
        File target;
        InputStream fileMonitorDll64 = MainClass.class.getResourceAsStream("/fileMonitor64.dll");
        InputStream fileMonitorDll32 = MainClass.class.getResourceAsStream("/fileMonitor32.dll");
        InputStream fileSearcherDll64 = MainClass.class.getResourceAsStream("/fileSearcher64.dll");
        InputStream fileSearcherDll32 = MainClass.class.getResourceAsStream("/fileSearcher32.dll");

        target = new File("fileMonitor.dll");
        if (!target.exists()) {
            File dllMonitor;
            if (name.contains("x64")) {
                copyFile(fileMonitorDll64, target);
                System.out.println("�Ѽ���64λfileMonitor");
                dllMonitor = new File("fileMonitor64.dll");
            } else {
                copyFile(fileMonitorDll32, target);
                System.out.println("�Ѽ���32λfileMonitor");
                dllMonitor = new File("fileMonitor32.dll");
            }
            dllMonitor.renameTo(target);
        }
        target = new File("fileSearcher.dll");
        if (!target.exists()) {
            File dllSearcher;
            if (name.contains("x64")) {
                copyFile(fileSearcherDll64, target);
                System.out.println("�Ѽ���64ΪfileSearcher");
                dllSearcher = new File("fileSearcher64.dll");
            } else {
                copyFile(fileSearcherDll32, target);
                System.out.println("�Ѽ���32λfileSearcher");
                dllSearcher = new File("fileSearcher32.dll");
            }
            dllSearcher.renameTo(target);
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

        SettingsFrame.initSettings();

        data = new File(SettingsFrame.dataPath + "\\data.dat");
        if (data.isFile() && data.exists()) {
            System.out.println("��⵽data�ļ������ڶ�ȡ");
            //showMessage("��ʾ", "��⵽data�ļ������ڶ�ȡ");
            search.setUsable(false);
            try {
                search.loadAllLists();
                search.setUsable(true);
                System.out.println("��ȡ���");
                showMessage("��ʾ", "��ȡ���");
            } catch (Exception e) {
                System.out.println("��⵽data�ļ��𻵣���ʼ����������data�ļ�");
                showMessage("��ʾ", "���⵽data�ļ��𻵣���ʼ����������data�ļ�");
                search.setManualUpdate(true);
            }
        }else {
            System.out.println("δ��⵽data�ļ�����ʼ����������data�ļ�");
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
                                search.addFileToLoadBin(filesToAdd);
                                System.out.println("���" + filesToAdd);
                            }
                        }
                        if (readerRemove != null) {
                            while ((filesToRemove = readerRemove.readLine()) != null) {
                                search.addToRecycleBin(filesToRemove);
                            }
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    e.printStackTrace();
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
            while (!mainExit) {
                boolean isUsing = searchBar.isUsing();
                count++;
                if (count >= (SettingsFrame.updateTimeLimit << 10) && !isUsing && !search.isManualUpdate()) {
                    count = 0;
                    System.out.println("���ڸ��±�������data�ļ�");
                    if (search.isUsable() && (!searchBar.isUsing())) {
                        deleteDir(SettingsFrame.dataPath);
                        search.saveLists();
                    }
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
                fixedThreadPool.shutdown();
                if (search.isUsable()) {
                    File CLOSEDLL = new File(SettingsFrame.tmp.getAbsolutePath() + "\\CLOSE");
                    try {
                        CLOSEDLL.createNewFile();
                    } catch (IOException ignored) {

                    }
                    System.out.println("�����˳������������ļ��б�data");
                    search.mergeFileToList();
                    deleteDir(SettingsFrame.dataPath);
                    search.saveLists();
                }
                System.exit(0);
            }
        }
    }
}
