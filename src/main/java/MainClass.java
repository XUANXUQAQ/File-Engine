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
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainClass {
    private static Search search;
    private static SearchBar searchBar;

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

    private static void deleteFile(String path) {
        File file = new File(path);
        file.delete();
    }


    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(new MaterialLookAndFeel(new MaterialDesign.materialLookAndFeel()));
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
        String osArch = System.getProperty("os.arch");
        if (osArch.contains("64")) {
            SettingsFrame.name = "File-Engine-x64.exe";
        } else {
            SettingsFrame.name = "File-Engine-x86.exe";
        }

        if (!initSettingsJson()) {
            System.err.println("initialize failed");
            System.exit(-1);
        }
        SettingsFrame.getInstance();

        startOrIgnoreUpdateAndExit(isUpdateSignExist());

        //清空tmp
        deleteDir(SettingsFrame.getTmp().getAbsolutePath());

        if (!initFoldersAndFiles()) {
            System.err.println("initialize failed");
            System.exit(-1);
        }

        searchBar = SearchBar.getInstance();
        search = Search.getInstance();

        TaskBar taskBar = TaskBar.getInstance();
        taskBar.showTaskBar();


        if (searchBar.isDataDamaged()) {
            System.out.println("无data文件，正在搜索并重建");
            search.setManualUpdate(true);
        }

        if (!isLatest()) {
            taskBar.showMessage(SettingsFrame.getTranslation("Info"), SettingsFrame.getTranslation("New version can be updated"));
        }


        File[] roots = File.listRoots();
        int size = roots.length + 5;
        ExecutorService fixedThreadPool = Executors.newFixedThreadPool(size);
        if (SettingsFrame.isAdmin()) {
            for (File root : roots) {
                boolean isLocal = IsLocalDisk.INSTANCE.isLocalDisk(root.getAbsolutePath());
                if (isLocal) {
                    fixedThreadPool.execute(() -> FileMonitor.INSTANCE.monitor(root.getAbsolutePath(), SettingsFrame.getTmp().getAbsolutePath(), SettingsFrame.getTmp().getAbsolutePath() + "/CLOSE"));
                }
            }
        } else {
            System.out.println("Not administrator, file monitoring function is turned off");
            taskBar.showMessage(SettingsFrame.getTranslation("Warning"), SettingsFrame.getTranslation("Not administrator, file monitoring function is turned off"));
        }

        fixedThreadPool.execute(() -> {
            //检测文件添加线程
            String filesToAdd;
            try (BufferedReader readerAdd = new BufferedReader(new InputStreamReader(
                    new FileInputStream(SettingsFrame.getTmp().getAbsolutePath() + File.separator + "fileAdded.txt"), StandardCharsets.UTF_8))) {
                while (!SettingsFrame.getMainExit()) {
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
            } catch (IOException | InterruptedException ignored) {

            }
        });

        fixedThreadPool.execute(() -> {
            String filesToRemove;
            try (BufferedReader readerRemove = new BufferedReader(new InputStreamReader(
                    new FileInputStream(SettingsFrame.getTmp().getAbsolutePath() + File.separator + "fileRemoved.txt"), StandardCharsets.UTF_8))) {
                while (!SettingsFrame.getMainExit()) {
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
            } catch (InterruptedException | IOException ignored) {

            }
        });


        fixedThreadPool.execute(() -> {
            // 时间检测线程
            long count = 0;
            try {
                while (!SettingsFrame.getMainExit()) {
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
                while (!SettingsFrame.getMainExit()) {
                    if (search.isManualUpdate()) {
                        search.setUsable(false);
                        searchBar.closeAllConnection();
                        search.updateLists(SettingsFrame.getIgnorePath(), SettingsFrame.getSearchDepth());
                    }
                    Thread.sleep(100);
                }
            } catch (InterruptedException ignored) {

            }
        });

        try {
            while (!SettingsFrame.getMainExit()) {
                // 主循环开始
                Thread.sleep(100);
            }
            CheckHotKey.getInstance().stopListen();
            File CLOSESign = new File(SettingsFrame.getTmp().getAbsolutePath() + File.separator + "CLOSE");
            CLOSESign.createNewFile();
            fixedThreadPool.shutdownNow();
            deleteFile(SettingsFrame.getTmp().getAbsolutePath() + File.separator + "fileAdded.txt");
            deleteFile(SettingsFrame.getTmp().getAbsolutePath() + File.separator + "fileRemoved.txt");
            searchBar.closeThreadPool();
            Thread.sleep(8000);
            System.exit(0);
        } catch (InterruptedException | IOException ignored) {

        }
    }

    private static void releaseAllDependence(boolean is64Bit) {
        if (is64Bit) {
            copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86-64/fileMonitor64.dll");
            copyOrIgnoreFile("user/getAscII.dll", "/win32-x86-64/getAscII64.dll");
            copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86-64/hotkeyListener64.dll");
            copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86-64/isLocalDisk64.dll");
            copyOrIgnoreFile("user/fileSearcher.exe", "/fileSearcher64.exe");
            copyOrIgnoreFile("user/restart.exe", "/restart64.exe");
        } else {
            copyOrIgnoreFile("user/fileMonitor.dll", "/win32-x86/fileMonitor86.dll");
            copyOrIgnoreFile("user/getAscII.dll", "/win32-x86/getAscII86.dll");
            copyOrIgnoreFile("user/hotkeyListener.dll", "/win32-x86/hotkeyListener86.dll");
            copyOrIgnoreFile("user/isLocalDisk.dll", "/win32-x86/isLocalDisk86.dll");
            copyOrIgnoreFile("user/fileSearcher.exe", "/fileSearcher86.exe");
            copyOrIgnoreFile("user/restart.exe", "/restart86.exe");
        }
        copyOrIgnoreFile("user/shortcutGenerator.vbs", "/shortcutGenerator.vbs");
    }

    private static void copyOrIgnoreFile(String path, String rootPath) {
        File target;
        target = new File(path);
        if (!target.exists()) {
            InputStream resource = MainClass.class.getResourceAsStream(rootPath);
            copyFile(resource, target);
            try {
                resource.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static void startOrIgnoreUpdateAndExit(boolean isUpdate) {
        //复制updater.exe
        if (isUpdate) {
            if (SettingsFrame.name.contains("x64")) {
                copyOrIgnoreFile("updater.exe", "/updater64.exe");
            } else {
                copyOrIgnoreFile("updater.exe", "/updater86.exe");
            }
            File updaterExe = new File("updater.exe");
            String absPath = updaterExe.getAbsolutePath();
            String path = absPath.substring(0, 2) + "\"" + absPath.substring(2) + "\"";
            String command = "cmd.exe /c " + path + " " + "\"" + SettingsFrame.name + "\"";
            try {
                Runtime.getRuntime().exec(command);
                System.exit(0);
            } catch (Exception ignored) {

            }
        } else {
            deleteFile("updater.exe");
        }
    }

    private static boolean initFoldersAndFiles() {
        boolean isFailed;
        //user
        isFailed = createFileOrFolder("user", false);
        //tmp
        File tmp = SettingsFrame.getTmp();
        String tempPath = tmp.getAbsolutePath();
        isFailed = isFailed && createFileOrFolder(tmp, false);
        isFailed = isFailed && createFileOrFolder(tempPath + File.separator + "fileAdded.txt", true);
        isFailed = isFailed && createFileOrFolder(tempPath + File.separator + "fileRemoved.txt", true);
        //cache.dat
        isFailed = isFailed && createFileOrFolder("user/cache.dat", true);
        releaseAllDependence(SettingsFrame.name.contains("x64"));
        return isFailed;
    }

    private static boolean initSettingsJson() {
        return createFileOrFolder("user/settings.json", true);
    }

    private static boolean createFileOrFolder(File file, boolean isFile) {
        boolean result;
        try {
            if (!file.exists()) {
                if (isFile) {
                    result = file.createNewFile();
                } else {
                    result = file.mkdirs();
                }
            } else {
                result = true;
            }
        } catch (IOException e) {
            result = false;
        }
        return result;
    }

    private static boolean createFileOrFolder(String path, boolean isFile) {
        File file = new File(path);
        return createFileOrFolder(file, isFile);
    }

    private static boolean isLatest() {
        //检测是否为最新版本
        try {
            JSONObject info = SettingsFrame.getInfo();
            String latestVersion = info.getString("version");
            if (Double.parseDouble(latestVersion) > Double.parseDouble(SettingsFrame.version)) {
                return false;
            }
        } catch (IOException ignored) {

        }
        return true;
    }

    private static boolean isUpdateSignExist() {
        File user = new File("user");
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
        return isUpdate;
    }
}
