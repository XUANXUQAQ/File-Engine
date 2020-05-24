package main;

import DllInterface.FileMonitor;
import DllInterface.IsLocalDisk;
import com.alibaba.fastjson.JSONObject;
import frames.SearchBar;
import frames.SettingsFrame;
import frames.TaskBar;
import hotkeyListener.CheckHotKey;
import mdlaf.MaterialLookAndFeel;
import search.Search;

import javax.swing.*;
import java.io.*;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainClass {
    public static final String version = "2.0"; //TODO 更改版本号
    public static volatile boolean mainExit = false;
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
                    scanner.close();
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

    private static void deleteDir(String path) {
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
                        System.out.println("Failed to delete " + name);
                    }
                }
            }
        }
    }


    public static void main(String[] args) {
        try {
            System.setProperty("sun.java2d.noddraw", "true");
            System.setProperty("jna.library.path", new File("user").getAbsolutePath() + "\\");
            UIManager.setLookAndFeel(new MaterialLookAndFeel(new MaterialDesign.materialLookAndFeel()));
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
                    each.delete();
                    break;
                }
            }
        }
        //复制updater.exe
        File updaterExe = new File("updater.exe");
        if (isUpdate) {
            InputStream updater;
            if (name.contains("x64")) {
                updater = MainClass.class.getResourceAsStream("/updater64.exe");
            } else {
                updater = MainClass.class.getResourceAsStream("/updater86.exe");
            }
            copyFile(updater, updaterExe);
            String absPath = updaterExe.getAbsolutePath();
            String path = absPath.substring(0, 2) + "\"" + absPath.substring(2) + "\"";
            String command = "cmd /c " + path + " " + "\"" + name + "\"";
            try {
                Runtime.getRuntime().exec(command);
                System.exit(0);
            } catch (Exception ignored) {

            }
        } else {
            updaterExe.delete();
        }

        File settings = SettingsFrame.getSettings();
        File caches = new File("user/cache.dat");

        if (!settings.exists()) {
            try {
                settings.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }


        //清空tmp
        MainClass.deleteDir(SettingsFrame.getTmp().getAbsolutePath());

        File target;

        boolean is64Bit = name.contains("x64");

        target = new File("user/fileMonitor.dll");
        if (is64Bit) {
            if (!target.exists()) {
                InputStream fileMonitor64Dll = MainClass.class.getResourceAsStream("/win32-x86-64/fileMonitor64.dll");
                copyFile(fileMonitor64Dll, target);
            }
            target = new File("user/getAscII.dll");
            if (!target.exists()) {
                InputStream getAscII64Dll = MainClass.class.getResourceAsStream("/win32-x86-64/getAscII64.dll");
                copyFile(getAscII64Dll, target);
            }
            target = new File("user/hotkeyListener.dll");
            if (!target.exists()) {
                InputStream hotkeyListener64Dll = MainClass.class.getResourceAsStream("/win32-x86-64/hotkeyListener64.dll");
                copyFile(hotkeyListener64Dll, target);
            }
            target = new File("user/isLocalDisk.dll");
            if (!target.exists()) {
                InputStream isLocakDisk64Dll = MainClass.class.getResourceAsStream("/win32-x86-64/isLocalDisk64.dll");
                copyFile(isLocakDisk64Dll, target);
            }
        } else {
            if (!target.exists()) {
                InputStream fileMonitor86Dll = MainClass.class.getResourceAsStream("/win32-x86/fileMonitor86.dll");
                copyFile(fileMonitor86Dll, target);
            }
            target = new File("user/getAscII.dll");
            if (!target.exists()) {
                InputStream getAscII86Dll = MainClass.class.getResourceAsStream("/win32-x86/getAscII86.dll");
                copyFile(getAscII86Dll, target);
            }
            target = new File("user/hotkeyListener.dll");
            if (!target.exists()) {
                InputStream hotkeyListener86Dll = MainClass.class.getResourceAsStream("/win32-x86/hotkeyListener86.dll");
                copyFile(hotkeyListener86Dll, target);
            }
            target = new File("user/isLocalDisk.dll");
            if (!target.exists()) {
                InputStream isLocalDisk86Dll = MainClass.class.getResourceAsStream("/win32-x86/isLocalDisk86.dll");
                copyFile(isLocalDisk86Dll, target);
            }
        }

        SettingsFrame.initSettings();
        SearchBar searchBar = SearchBar.getInstance();

        target = new File("user/fileSearcher.exe");
        if (!target.exists()) {
            InputStream fileSearcher64 = MainClass.class.getResourceAsStream("/fileSearcher64.exe");
            InputStream fileSearcher86 = MainClass.class.getResourceAsStream("/fileSearcher86.exe");
            if (is64Bit) {
                copyFile(fileSearcher64, target);
            } else {
                copyFile(fileSearcher86, target);
            }
        }
        target = new File(SettingsFrame.getDataPath());
        if (!target.exists()) {
            target.mkdir();
        }
        target = new File("user/shortcutGenerator.vbs");
        if (!target.exists()) {
            InputStream shortcutGen = MainClass.class.getResourceAsStream("/shortcutGenerator.vbs");
            copyFile(shortcutGen, target);
        }
        target = new File("user/restart.exe");
        if (!target.exists()) {
            InputStream restart64 = MainClass.class.getResourceAsStream("/restart64.exe");
            InputStream restart86 = MainClass.class.getResourceAsStream("/restart86.exe");
            if (is64Bit) {
                copyFile(restart64, target);
            } else {
                copyFile(restart86, target);
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
        File temp = SettingsFrame.getTmp();
        if (!temp.exists()) {
            temp.mkdir();
        }
        File fileAdded = new File(temp.getAbsolutePath() + "\\fileAdded.txt");
        File fileRemoved = new File(temp.getAbsolutePath() + "\\fileRemoved.txt");
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
        ExecutorService fixedThreadPool = Executors.newFixedThreadPool(roots.length + 5);


        File data = new File(SettingsFrame.getDataPath());
        File[] files = data.listFiles();
        if (!data.exists()) {
            System.out.println("无data文件，正在搜索并重建");
            search.setManualUpdate(true);
        } else if (files == null) {
            System.out.println("data文件损坏，正在搜索并重建");
            search.setManualUpdate(true);
        } else {
            assert false;
            if (files.length != 26) {
                System.out.println("data文件损坏，正在搜索并重建");
                search.setManualUpdate(true);
            }
        }

        if (isAdmin()) {
            for (File root : roots) {
                boolean isLocal = IsLocalDisk.INSTANCE.isLocalDisk(root.getAbsolutePath());
                if (isLocal) {
                    fixedThreadPool.execute(() -> FileMonitor.INSTANCE.monitor(root.getAbsolutePath(), SettingsFrame.getTmp().getAbsolutePath(), SettingsFrame.getTmp().getAbsolutePath() + "\\CLOSE"));
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
            //检测文件添加线程
            String filesToAdd;
            try (BufferedReader readerAdd = new BufferedReader(new FileReader(SettingsFrame.getTmp().getAbsolutePath() + "\\fileAdded.txt"))) {
                while (!mainExit) {
                    if (!search.isManualUpdate()) {
                        if ((filesToAdd = readerAdd.readLine()) != null) {
                            if (!filesToAdd.contains(SettingsFrame.getDataPath())) {
                                search.addFileToLoadBin(filesToAdd);
                                System.out.println("添加" + filesToAdd);
                            }
                        }
                    }
                    Thread.sleep(100);
                }
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        });

        fixedThreadPool.execute(() -> {
            String filesToRemove;
            try (BufferedReader readerRemove = new BufferedReader(new FileReader(SettingsFrame.getTmp().getAbsolutePath() + "\\fileRemoved.txt"))) {
                while (!mainExit) {
                    if (!search.isManualUpdate()) {
                        if ((filesToRemove = readerRemove.readLine()) != null) {
                            if (!filesToRemove.contains(SettingsFrame.getDataPath())) {
                                search.addToRecycleBin(filesToRemove);
                                System.out.println("删除" + filesToRemove);
                            }
                        }
                    }
                    Thread.sleep(100);
                }
            } catch (InterruptedException | IOException e) {
                e.printStackTrace();
            }
        });


        fixedThreadPool.execute(() -> {
            // 时间检测线程
            long count = 0;
            try {
                while (!mainExit) {
                    boolean isUsing = searchBar.isUsing();
                    count += 100;
                    if (count >= (SettingsFrame.getUpdateTimeLimit() << 10) && !isUsing && !search.isManualUpdate()) {
                        count = 0;
                        if (search.isUsable() && (!searchBar.isUsing())) {
                            search.mergeFileToList();
                        }
                    }
                    Thread.sleep(100);
                }
            } catch (InterruptedException ignore) {

            }
        });


        //搜索线程
        fixedThreadPool.execute(() -> {
            try {
                while (!mainExit) {
                    if (search.isManualUpdate()) {
                        search.setUsable(false);
                        SearchBar.getInstance().closeAllConnection();
                        search.updateLists(SettingsFrame.getIgnorePath(), SettingsFrame.getSearchDepth());
                    }
                    Thread.sleep(100);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        try {
            while (true) {
                // 主循环开始
                Thread.sleep(100);
                if (mainExit) {
                    CheckHotKey.getInstance().stopListen();
                    File CLOSEDLL = new File(SettingsFrame.getTmp().getAbsolutePath() + "\\CLOSE");
                    CLOSEDLL.createNewFile();
                    Thread.sleep(8000);
                    fixedThreadPool.shutdown();
                    System.exit(0);
                }
            }

        } catch (InterruptedException | IOException ignored) {

        }
    }
}
