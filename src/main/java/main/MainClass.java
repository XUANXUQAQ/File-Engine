package main;

import com.alibaba.fastjson.JSONObject;
import fileMonitor.FileMonitor;
import frame.CheckHotKey;
import frame.SearchBar;
import frame.SettingsFrame;
import frame.TaskBar;
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
    public static final String version = "1.4"; //TODO 更改版本号
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

    private static void copyFile(InputStream source, File dest) {
        try (OutputStream os = new FileOutputStream(dest); BufferedInputStream bis = new BufferedInputStream(source); BufferedOutputStream bos = new BufferedOutputStream(os)) {
            byte[] buffer = new byte[8192];
            int count = bis.read(buffer);
            while (count != -1) {
                //使用缓冲流写数据
                bos.write(buffer, 0, count);
                //刷新
                bos.flush();
                count = bis.read(buffer);
            }
        } catch (IOException ignored) {

        }
    }

    public static void deleteDir(String path) {
        File file = new File(path);
        if (!file.exists()) {//判断是否待删除目录是否存在
            return;
        }

        String[] content = file.list();//取得当前目录下所有文件和文件夹
        if (content != null) {
            for (String name : content) {
                File temp = new File(path, name);
                if (temp.isDirectory()) {//判断是否是目录
                    deleteDir(temp.getAbsolutePath());//递归调用，删除目录里的内容
                    temp.delete();//删除空目录
                } else {
                    if (!temp.delete()) {//直接删除文件
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
            UIManager.put("RootPane.setupButtonVisible", false);
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
                    File originFile = new File(name);
                    File updated = new File("_" + name);
                    copyFile(new FileInputStream(originFile), updated);
                    Desktop desktop;
                    if (Desktop.isDesktopSupported()) {
                        desktop = Desktop.getDesktop();
                        //复制并重命名自己后退出，打开复制后的程序
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
                        //复制并重命名自己后退出，打开复制后的程序
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
        File caches = new File("./user/cache.dat");
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

        SettingsFrame.initSettings();
        SearchBar searchBar = SearchBar.getInstance();

        //清空tmp
        MainClass.deleteDir(SettingsFrame.tmp.getAbsolutePath());

        File target;
        InputStream fileMonitorDll64 = MainClass.class.getResourceAsStream("/fileMonitor64.dll");
        InputStream fileMonitorDll86 = MainClass.class.getResourceAsStream("/fileMonitor86.dll");
        InputStream fileSearcherDll64 = MainClass.class.getResourceAsStream("/fileSearcher64.exe");
        InputStream fileSearcherDll86 = MainClass.class.getResourceAsStream("/fileSearcher86.exe");
        InputStream fileOpener64 = MainClass.class.getResourceAsStream("/fileOpener64.exe");
        InputStream fileOpener86 = MainClass.class.getResourceAsStream("/fileOpener86.exe");
        InputStream getAscIIDll64 = MainClass.class.getResourceAsStream("/getAscII64.dll");
        InputStream getAscIIDll86 = MainClass.class.getResourceAsStream("/getAscII86.dll");

        target = new File("./user/fileMonitor.dll");
        if (!target.exists()) {
            if (name.contains("x64")) {
                copyFile(fileMonitorDll64, target);
                System.out.println("已加载64位fileMonitor");
            } else {
                copyFile(fileMonitorDll86, target);
                System.out.println("已加载32位fileMonitor");
            }
        }
        target = new File("./user/fileSearcher.exe");
        if (!target.exists()) {
            if (name.contains("x64")) {
                copyFile(fileSearcherDll64, target);
                System.out.println("已加载64位fileSearcher");
            } else {
                copyFile(fileSearcherDll86, target);
                System.out.println("已加载32位fileSearcher");
            }
        }
        target = new File(SettingsFrame.dataPath);
        if (!target.exists()) {
            target.mkdir();
        }
        target = new File("./user/fileOpener.exe");
        if (!target.exists()) {
            if (name.contains("x64")) {
                copyFile(fileOpener64, target);
                System.out.println("已加载64位fileOpener");
            } else {
                copyFile(fileOpener86, target);
                System.out.println("已加载32位fileOpener");
            }
        }
        target = new File("./user/getAscII.dll");
        if (!target.exists()) {
            if (name.contains("x64")) {
                copyFile(getAscIIDll64, target);
                System.out.println("已加载64位getAscII");
            } else {
                copyFile(getAscIIDll86, target);
                System.out.println("已加载32位getAscII");
            }
        }

        if (!caches.exists()) {
            try {
                caches.createNewFile();
            } catch (IOException e) {
                JOptionPane.showMessageDialog(null, "创建缓存文件失败，程序正在退出");
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
                System.out.println("检测到data文件损坏，正在搜索并重建");
                search.setManualUpdate(true);
            }
        } else {
            System.out.println("无data文件，正在搜索并重建");
            search.setManualUpdate(true);
        }

        FileSystemView sys = FileSystemView.getFileSystemView();
        if (isAdmin()) {
            for (File root : roots) {
                String dirveType = sys.getSystemTypeDescription(root);
                if (dirveType.equals("本地磁盘")) {
                    fixedThreadPool.execute(() -> FileMonitor.INSTANCE.monitor(root.getAbsolutePath(), SettingsFrame.tmp.getAbsolutePath(), SettingsFrame.tmp.getAbsolutePath() + "\\CLOSE"));
                }
            }
        } else {
            System.out.println("无管理员权限，文件监控功能已关闭");
            taskBar.showMessage("警告", "无管理员权限，文件监控功能已关闭");
        }

        //检测是否为最新版本
        try {
            JSONObject info = SettingsFrame.getInfo();
            String latestVersion = info.getString("version");
            if (Double.parseDouble(latestVersion) > Double.parseDouble(version)) {
                showMessage("提示", "有新版本可更新");
            }
        } catch (IOException ignored) {

        }


        fixedThreadPool.execute(() -> {
            //检测文件改动线程
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
            //分割字符串
            while (!mainExit) {
                if (!search.isManualUpdate()) {
                    try {
                        if (readerAdd != null) {
                            while ((filesToAdd = readerAdd.readLine()) != null) {
                                if (!filesToAdd.contains(SettingsFrame.dataPath)) {
                                    search.addFileToLoadBin(filesToAdd);
                                    System.out.println("添加" + filesToAdd);
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
            // 时间检测线程
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
                    System.out.println("正在释放内存空间");
                    System.gc();
                    gcCount = 0;
                }
                try {
                    Thread.sleep(1);
                } catch (InterruptedException ignore) {

                }
            }
        });


        //搜索线程
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
            // 主循环开始
            try {
                Thread.sleep(1);
            } catch (InterruptedException ignored) {

            }
            if (mainExit) {
                CheckHotKey.getInstance().unRegisterHotkey();
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
