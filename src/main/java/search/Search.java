package search;

import DllInterface.IsLocalDisk;
import frames.SearchBar;
import frames.SettingsFrame;
import main.MainClass;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ConcurrentModificationException;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


public class Search {
    private volatile static boolean isUsable = true;
    private static boolean isManualUpdate = false;
    private Set<String> listToLoad = ConcurrentHashMap.newKeySet();
    private Set<String> set0 = ConcurrentHashMap.newKeySet();
    private Set<String> set100 = ConcurrentHashMap.newKeySet();
    private Set<String> set200 = ConcurrentHashMap.newKeySet();
    private Set<String> set300 = ConcurrentHashMap.newKeySet();
    private Set<String> set400 = ConcurrentHashMap.newKeySet();
    private Set<String> set500 = ConcurrentHashMap.newKeySet();
    private Set<String> set600 = ConcurrentHashMap.newKeySet();
    private Set<String> set700 = ConcurrentHashMap.newKeySet();
    private Set<String> set800 = ConcurrentHashMap.newKeySet();
    private Set<String> set900 = ConcurrentHashMap.newKeySet();
    private Set<String> set1000 = ConcurrentHashMap.newKeySet();
    private Set<String> set1100 = ConcurrentHashMap.newKeySet();
    private Set<String> set1200 = ConcurrentHashMap.newKeySet();
    private Set<String> set1300 = ConcurrentHashMap.newKeySet();
    private Set<String> set1400 = ConcurrentHashMap.newKeySet();
    private Set<String> set1500 = ConcurrentHashMap.newKeySet();
    private Set<String> set1600 = ConcurrentHashMap.newKeySet();
    private Set<String> set1700 = ConcurrentHashMap.newKeySet();
    private Set<String> set1800 = ConcurrentHashMap.newKeySet();
    private Set<String> set1900 = ConcurrentHashMap.newKeySet();
    private Set<String> set2000 = ConcurrentHashMap.newKeySet();
    private Set<String> set2100 = ConcurrentHashMap.newKeySet();
    private Set<String> set2200 = ConcurrentHashMap.newKeySet();
    private Set<String> set2300 = ConcurrentHashMap.newKeySet();
    private Set<String> set2400 = ConcurrentHashMap.newKeySet();
    private Set<String> set2500 = ConcurrentHashMap.newKeySet();
    private SearchBar instance = SearchBar.getInstance();

    private static class SearchBuilder {
        private static Search instance = new Search();
    }

    private Search() {
    }

    public static Search getInstance() {
        return SearchBuilder.instance;
    }

    private static void writeRecordToFile(String record, String srcPath) {
        if (!isContained(record, srcPath)) {
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(srcPath, true), StandardCharsets.UTF_8))) {
                bw.write(record);
                bw.write("\n");
            } catch (IOException ignored) {

            }
        }
    }

    private static boolean isContained(String record, String filePath) {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String eachLine;
            while ((eachLine = br.readLine()) != null) {
                if (eachLine.equals(record)) {
                    return true;
                }
            }
        } catch (IOException ignored) {

        }
        return false;
    }

    public int getRecycleBinSize() {
        return set0.size() + set100.size() + set200.size() + set300.size() + set400.size() + set500.size() + set600.size()
                + set700.size() + set800.size() + set900.size() + set1000.size() + set1100.size() + set1200.size() + set1300.size()
                + set1400.size() + set1500.size() + set1600.size() + set1700.size() + set1800.size() + set1900.size() + set2000.size()
                + set2100.size() + set2200.size() + set2300.size() + set2400.size() + set2500.size();
    }

    public void addToRecycleBin(String path) {
        int ascII = instance.getAscIISum(instance.getFileName(path));
        if (0 <= ascII && ascII <= 100) {
            set0.add(path);
        } else if (100 < ascII && ascII <= 200) {
            set100.add(path);
        } else if (200 < ascII && ascII <= 300) {
            set200.add(path);
        } else if (300 < ascII && ascII <= 400) {
            set300.add(path);
        } else if (400 < ascII && ascII <= 500) {
            set400.add(path);
        } else if (500 < ascII && ascII <= 600) {
            set500.add(path);
        } else if (600 < ascII && ascII <= 700) {
            set600.add(path);
        } else if (700 < ascII && ascII <= 800) {
            set700.add(path);
        } else if (800 < ascII && ascII <= 900) {
            set800.add(path);
        } else if (900 < ascII && ascII <= 1000) {
            set900.add(path);
        } else if (1000 < ascII && ascII <= 1100) {
            set1000.add(path);
        } else if (1100 < ascII && ascII <= 1200) {
            set1100.add(path);
        } else if (1200 < ascII && ascII <= 1300) {
            set1200.add(path);
        } else if (1300 < ascII && ascII <= 1400) {
            set1300.add(path);
        } else if (1400 < ascII && ascII <= 1500) {
            set1400.add(path);
        } else if (1500 < ascII && ascII <= 1600) {
            set1500.add(path);
        } else if (1600 < ascII && ascII <= 1700) {
            set1600.add(path);
        } else if (1700 < ascII && ascII <= 1800) {
            set1700.add(path);
        } else if (1800 < ascII && ascII <= 1900) {
            set1800.add(path);
        } else if (1900 < ascII && ascII <= 2000) {
            set1900.add(path);
        } else if (2000 < ascII && ascII <= 2100) {
            set2000.add(path);
        } else if (2100 < ascII && ascII <= 2200) {
            set2100.add(path);
        } else if (2200 < ascII && ascII <= 2300) {
            set2200.add(path);
        } else if (2300 < ascII && ascII <= 2400) {
            set2300.add(path);
        } else if (2400 < ascII && ascII <= 2500) {
            set2400.add(path);
        } else {
            set2500.add(path);
        }
    }

    public void mergeAndClearRecycleBin() {
        if (!isManualUpdate) {
            isUsable = false;
            try {
                generateAllNewLocalRecord();
            } catch (ConcurrentModificationException ignored) {

            } finally {
                //清空回收站
                clearAllSets();
                isUsable = true;
            }
        }
    }

    private void generateAllNewLocalRecord() {
        String srcPath;
        String destPath;

        if (!set0.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list0-100.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list0-100.dat";
            generateNewLocalRecord(srcPath, destPath, set0);
        }
        if (!set100.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list100-200.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list100-200.dat";
            generateNewLocalRecord(srcPath, destPath, set100);
        }
        if (!set200.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list200-300.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list200-300.dat";
            generateNewLocalRecord(srcPath, destPath, set200);
        }
        if (!set300.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list300-400.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list300-400.dat";
            generateNewLocalRecord(srcPath, destPath, set300);
        }
        if (!set400.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list400-500.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list400-500.dat";
            generateNewLocalRecord(srcPath, destPath, set400);
        }
        if (!set500.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list500-600.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list500-600.dat";
            generateNewLocalRecord(srcPath, destPath, set500);
        }
        if (!set600.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list600-700.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list600-700.dat";
            generateNewLocalRecord(srcPath, destPath, set600);
        }
        if (!set700.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list700-800.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list700-800.dat";
            generateNewLocalRecord(srcPath, destPath, set700);
        }
        if (!set800.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list800-900.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list800-900.dat";
            generateNewLocalRecord(srcPath, destPath, set800);
        }
        if (!set900.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list900-1000.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list900-1000.dat";
            generateNewLocalRecord(srcPath, destPath, set900);
        }
        if (!set1000.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1000-1100.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1000-1100.dat";
            generateNewLocalRecord(srcPath, destPath, set1000);
        }
        if (!set1100.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1100-1200.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1100-1200.dat";
            generateNewLocalRecord(srcPath, destPath, set1100);
        }
        if (!set1200.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1200-1300.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1200-1300.dat";
            generateNewLocalRecord(srcPath, destPath, set1200);
        }
        if (!set1300.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1300-1400.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1300-1400.dat";
            generateNewLocalRecord(srcPath, destPath, set1300);
        }
        if (!set1400.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1400-1500.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1400-1500.dat";
            generateNewLocalRecord(srcPath, destPath, set1400);
        }
        if (!set1500.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1500-1600.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1500-1600.dat";
            generateNewLocalRecord(srcPath, destPath, set1500);
        }
        if (!set1600.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1600-1700.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1600-1700.dat";
            generateNewLocalRecord(srcPath, destPath, set1600);
        }
        if (!set1700.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1700-1800.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1700-1800.dat";
            generateNewLocalRecord(srcPath, destPath, set1700);
        }
        if (!set1800.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1800-1900.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1800-1900.dat";
            generateNewLocalRecord(srcPath, destPath, set1800);
        }
        if (!set1900.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list1900-2000.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list1900-2000.dat";
            generateNewLocalRecord(srcPath, destPath, set1900);
        }
        if (!set2000.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list2000-2100.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list2000-2100.dat";
            generateNewLocalRecord(srcPath, destPath, set2000);
        }
        if (!set2100.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list2100-2200.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list2100-2200.dat";
            generateNewLocalRecord(srcPath, destPath, set2100);
        }
        if (!set2200.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list2200-2300.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list2200-2300.dat";
            generateNewLocalRecord(srcPath, destPath, set2200);
        }
        if (!set2300.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list2300-2400.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list2300-2400.dat";
            generateNewLocalRecord(srcPath, destPath, set2300);
        }
        if (!set2400.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list2400-2500.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list2400-2500.dat";
            generateNewLocalRecord(srcPath, destPath, set2400);
        }
        if (!set2500.isEmpty()) {
            srcPath = SettingsFrame.getDataPath() + "\\list2500-.dat";
            destPath = SettingsFrame.getDataPath() + "\\_list2500-.dat";
            generateNewLocalRecord(srcPath, destPath, set2500);
        }
    }

    private void clearAllSets() {
        set0.clear();
        set100.clear();
        set200.clear();
        set300.clear();
        set400.clear();
        set500.clear();
        set600.clear();
        set700.clear();
        set800.clear();
        set900.clear();
        set1000.clear();
        set1100.clear();
        set1200.clear();
        set1300.clear();
        set1400.clear();
        set1500.clear();
        set1600.clear();
        set1700.clear();
        set1800.clear();
        set1900.clear();
        set2000.clear();
        set2100.clear();
        set2200.clear();
        set2300.clear();
        set2400.clear();
        set2500.clear();
    }

    //将旧文件内容复制到新文件，需要删除的不复制
    private void generateNewLocalRecord(String srcPath, String destPath, Set<String> set) {
        File src, target;
        deleteRecordInFile(set, srcPath, destPath);
        src = new File(srcPath);
        src.delete();
        target = new File(destPath);
        target.renameTo(src);
    }

    private void deleteRecordInFile(Set<String> recordToDel, String srcText, String destText) {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(srcText), StandardCharsets.UTF_8));
             BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(destText, true), StandardCharsets.UTF_8))) {
            String eachLine;
            while ((eachLine = br.readLine()) != null) {
                if (!recordToDel.contains(eachLine)) {
                    bw.write(eachLine);
                    bw.write("\n");
                }
            }
        } catch (IOException ignored) {

        }
    }

    public void addFileToLoadBin(String path) {
        listToLoad.add(path);
    }

    public void mergeFileToList() {
        if (!isManualUpdate) {
            isUsable = false;
            for (String each : listToLoad) {
                File add = new File(each);
                if (add.exists()) {
                    addRecordToLocal(each);
                }
            }
            isUsable = true;
            listToLoad.clear();
        }
    }

    public boolean isManualUpdate() {
        return isManualUpdate;
    }

    public void setManualUpdate(boolean b) {
        isManualUpdate = b;
    }

    public boolean isUsable() {
        return isUsable;
    }

    public void setUsable(boolean b) {
        if (!isManualUpdate) {
            isUsable = b;
        } else {
            isUsable = false;
        }
    }

    private void addRecordToLocal(String path) {
        File file = new File(path);
        int ascII = SearchBar.getInstance().getAscIISum(file.getName());
        String listPath;
        if (0 < ascII && ascII <= 100) {
            listPath = SettingsFrame.getDataPath() + "\\list0-100.dat";
            writeRecordToFile(path, listPath);
        } else if (100 < ascII && ascII <= 200) {
            listPath = SettingsFrame.getDataPath() + "\\list100-200.dat";
            writeRecordToFile(path, listPath);
        } else if (200 < ascII && ascII <= 300) {
            listPath = SettingsFrame.getDataPath() + "\\list200-300.dat";
            writeRecordToFile(path, listPath);
        } else if (300 < ascII && ascII <= 400) {
            listPath = SettingsFrame.getDataPath() + "\\list300-400.dat";
            writeRecordToFile(path, listPath);
        } else if (400 < ascII && ascII <= 500) {
            listPath = SettingsFrame.getDataPath() + "\\list400-500.dat";

            writeRecordToFile(path, listPath);
        } else if (500 < ascII && ascII <= 600) {
            listPath = SettingsFrame.getDataPath() + "\\list500-600.dat";

            writeRecordToFile(path, listPath);
        } else if (600 < ascII && ascII <= 700) {
            listPath = SettingsFrame.getDataPath() + "\\list600-700.dat";
            writeRecordToFile(path, listPath);
        } else if (700 < ascII && ascII <= 800) {
            listPath = SettingsFrame.getDataPath() + "\\list700-800.dat";

            writeRecordToFile(path, listPath);
        } else if (800 < ascII && ascII <= 900) {
            listPath = SettingsFrame.getDataPath() + "\\list800-900.dat";

            writeRecordToFile(path, listPath);
        } else if (900 < ascII && ascII <= 1000) {
            listPath = SettingsFrame.getDataPath() + "\\list900-1000.dat";
            writeRecordToFile(path, listPath);
        } else if (1000 < ascII && ascII <= 1100) {
            listPath = SettingsFrame.getDataPath() + "\\list1000-1100.dat";

            writeRecordToFile(path, listPath);
        } else if (1100 < ascII && ascII <= 1200) {
            listPath = SettingsFrame.getDataPath() + "\\list1100-1200.dat";
            writeRecordToFile(path, listPath);
        } else if (1200 < ascII && ascII <= 1300) {
            listPath = SettingsFrame.getDataPath() + "\\list1200-1300.dat";
            writeRecordToFile(path, listPath);
        } else if (1300 < ascII && ascII <= 1400) {
            listPath = SettingsFrame.getDataPath() + "\\list1300-1400.dat";

            writeRecordToFile(path, listPath);
        } else if (1400 < ascII && ascII <= 1500) {
            listPath = SettingsFrame.getDataPath() + "\\list1400-1500.dat";

            writeRecordToFile(path, listPath);
        } else if (1500 < ascII && ascII <= 1600) {
            listPath = SettingsFrame.getDataPath() + "\\list1500-1600.dat";

            writeRecordToFile(path, listPath);
        } else if (1600 < ascII && ascII <= 1700) {
            listPath = SettingsFrame.getDataPath() + "\\list1600-1700.dat";

            writeRecordToFile(path, listPath);
        } else if (1700 < ascII && ascII <= 1800) {
            listPath = SettingsFrame.getDataPath() + "\\list1700-1800.dat";

            writeRecordToFile(path, listPath);
        } else if (1800 < ascII && ascII <= 1900) {
            listPath = SettingsFrame.getDataPath() + "\\list1800-1900.dat";

            writeRecordToFile(path, listPath);
        } else if (1900 < ascII && ascII <= 2000) {
            listPath = SettingsFrame.getDataPath() + "\\list1900-2000.dat";

            writeRecordToFile(path, listPath);
        } else if (2000 < ascII && ascII <= 2100) {
            listPath = SettingsFrame.getDataPath() + "\\list2000-2100.dat";

            writeRecordToFile(path, listPath);
        } else if (2100 < ascII && ascII <= 2200) {
            listPath = SettingsFrame.getDataPath() + "\\list2100-2200.dat";

            writeRecordToFile(path, listPath);
        } else if (2200 < ascII && ascII <= 2300) {
            listPath = SettingsFrame.getDataPath() + "\\list2200-2300.dat";

            writeRecordToFile(path, listPath);
        } else if (2300 < ascII && ascII <= 2400) {
            listPath = SettingsFrame.getDataPath() + "\\list2300-2400.dat";

            writeRecordToFile(path, listPath);
        } else if (2400 < ascII && ascII <= 2500) {
            listPath = SettingsFrame.getDataPath() + "\\list2400-2500.dat";

            writeRecordToFile(path, listPath);
        } else {
            listPath = SettingsFrame.getDataPath() + "\\list2500-.dat";
            writeRecordToFile(path, listPath);
        }

        if (!isManualUpdate) {
            isUsable = true;
        }
    }


    private void searchFile(String ignorePath, int searchDepth) {
        File[] roots = File.listRoots();
        ExecutorService pool = Executors.newFixedThreadPool(roots.length);
        //创建搜索结果存放文件夹
        for (int i = 0; i < SettingsFrame.getDiskCount(); i++) {
            File eachResults = new File(SettingsFrame.getDataPath() + "\\" + i);
            if (!eachResults.exists()) {
                eachResults.mkdir();
            }
        }
        int count = 0;
        for (File root : roots) {
            if (IsLocalDisk.INSTANCE.isLocalDisk(root.getAbsolutePath())) {
                String path = root.getAbsolutePath();
                path = path.substring(0, 2);
                String finalPath = path;
                int finalCount = count;
                pool.execute(() -> __searchFile(finalPath, searchDepth, ignorePath, finalCount));
                count++;
            }
        }
        pool.shutdown();
        try {
            pool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        __searchFileIgnoreSearchDepth(getStartMenu(), ignorePath);
        __searchFileIgnoreSearchDepth("C:\\ProgramData\\Microsoft\\Windows\\Start Menu", ignorePath);

        //生成合并搜索结果cmd命令
        StringBuilder strb = new StringBuilder();
        strb.append("cmd.exe /c copy ");
        StringBuilder listPath = new StringBuilder();
        HashSet<String> commands = new HashSet<>();
        for (int i = 0; i < 2500; i += 100) {
            int name = i + 100;
            for (int j = 0; j < SettingsFrame.getDiskCount(); j++) {
                String _temp = SettingsFrame.getDataPath() + "\\" + j + "\\list" + i + "-" + name + ".dat";
                String _tempPath = _temp.substring(0, 2) + "\"" + _temp.substring(2) + "\"";
                listPath.append(_tempPath).append("+");
            }
            strb.append(listPath.toString(), 0, listPath.length() - 1).append(" ").append("\"").append(SettingsFrame.getDataPath())
                    .append("\\list").append(i).append("-").append(name).append(".dat").append("\"");
            commands.add(strb.toString());
            listPath.delete(0, listPath.length());
            strb.delete(0, strb.length());
            strb.append("cmd.exe /c copy ");
        }
        for (int j = 0; j < SettingsFrame.getDiskCount(); j++) {
            String _temp = SettingsFrame.getDataPath() + "\\" + j + "\\list2500-.dat";
            String _tempPath = _temp.substring(0, 2) + "\"" + _temp.substring(2) + "\"";
            listPath.append(_tempPath).append("+");
        }
        strb.append(listPath.toString(), 0, listPath.length() - 1).append(" ").append("\"").append(SettingsFrame.getDataPath())
                .append("\\list2500-.dat").append("\"");
        commands.add(strb.toString());

        //合并所有搜索结果
        for (String each : commands) {
            Process p;
            try {
                p = Runtime.getRuntime().exec(each);
                p.waitFor();
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        }


        //删除之前的结果
        for (int i = 0; i < SettingsFrame.getDiskCount(); i++) {
            String path = SettingsFrame.getDataPath() + "\\" + i;
            deleteDir(path);
            boolean isDeleted = new File(path).delete();
            if (!isDeleted) {
                System.out.println("文件夹" + i + "删除失败");
            }
        }
        MainClass.showMessage("提示", "搜索完成");
        isManualUpdate = false;
        isUsable = true;
    }

    private String getStartMenu() {
        String startMenu;
        BufferedReader bufrIn;
        try {
            Process getStartMenu = Runtime.getRuntime().exec("reg query \"HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders\" " + "/v " + "\"Start Menu\"");
            bufrIn = new BufferedReader(new InputStreamReader(getStartMenu.getInputStream(), StandardCharsets.UTF_8));
            while ((startMenu = bufrIn.readLine()) != null) {
                if (startMenu.contains("REG_SZ")) {
                    startMenu = startMenu.substring(startMenu.indexOf("REG_SZ") + 10);
                    return startMenu;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private void __searchFileIgnoreSearchDepth(String path, String ignorePath) {
        File fileSearcher = new File("user/fileSearcher.exe");
        String absPath = fileSearcher.getAbsolutePath();
        String start = absPath.substring(0, 2);
        String end = "\"" + absPath.substring(2) + "\"";
        String command = "cmd /c " + start + end + " \"" + path + "\"" + " \"6\" " + "\"" + ignorePath + "\" " + "\"" + SettingsFrame.getDataPath() + "\\0" + "\" " + "\"" + "1" + "\"";
        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            while (p.isAlive()) {
                Thread.sleep(10);
            }
        } catch (IOException | InterruptedException ignored) {

        }
    }

    private void __searchFile(String path, int searchDepth, String ignorePath, int count) {
        File fileSearcher = new File("user/fileSearcher.exe");
        String absPath = fileSearcher.getAbsolutePath();
        String start = absPath.substring(0, 2);
        String end = "\"" + absPath.substring(2) + "\"";
        String command = "cmd /c " + start + end + " \"" + path + "\"" + " \"" + searchDepth + "\" " + "\"" + ignorePath + "\" " + "\"" + SettingsFrame.getDataPath() + "\\" + count + "\" " + "\"" + "0" + "\"";
        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            while (p.isAlive()) {
                Thread.sleep(10);
            }
        } catch (IOException | InterruptedException ignored) {

        }
    }


    public void updateLists(String ignorePath, int searchDepth) {
        deleteDir(SettingsFrame.getDataPath());
        searchFile(ignorePath, searchDepth);
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
}