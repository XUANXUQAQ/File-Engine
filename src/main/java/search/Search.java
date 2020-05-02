package search;

import frames.SearchBar;
import frames.SettingsFrame;
import main.MainClass;

import javax.swing.filechooser.FileSystemView;
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
    private static boolean isUsable = true;
    private static boolean isManualUpdate = false;
    private static Search searchInstance = new Search();
    private Set<String> RecycleBin = ConcurrentHashMap.newKeySet();
    private Set<String> listToLoad = ConcurrentHashMap.newKeySet();
    Set<String> set0 = ConcurrentHashMap.newKeySet();
    Set<String> set100 = ConcurrentHashMap.newKeySet();
    Set<String> set200 = ConcurrentHashMap.newKeySet();
    Set<String> set300 = ConcurrentHashMap.newKeySet();
    Set<String> set400 = ConcurrentHashMap.newKeySet();
    Set<String> set500 = ConcurrentHashMap.newKeySet();
    Set<String> set600 = ConcurrentHashMap.newKeySet();
    Set<String> set700 = ConcurrentHashMap.newKeySet();
    Set<String> set800 = ConcurrentHashMap.newKeySet();
    Set<String> set900 = ConcurrentHashMap.newKeySet();
    Set<String> set1000 = ConcurrentHashMap.newKeySet();
    Set<String> set1100 = ConcurrentHashMap.newKeySet();
    Set<String> set1200 = ConcurrentHashMap.newKeySet();
    Set<String> set1300 = ConcurrentHashMap.newKeySet();
    Set<String> set1400 = ConcurrentHashMap.newKeySet();
    Set<String> set1500 = ConcurrentHashMap.newKeySet();
    Set<String> set1600 = ConcurrentHashMap.newKeySet();
    Set<String> set1700 = ConcurrentHashMap.newKeySet();
    Set<String> set1800 = ConcurrentHashMap.newKeySet();
    Set<String> set1900 = ConcurrentHashMap.newKeySet();
    Set<String> set2000 = ConcurrentHashMap.newKeySet();
    Set<String> set2100 = ConcurrentHashMap.newKeySet();
    Set<String> set2200 = ConcurrentHashMap.newKeySet();
    Set<String> set2300 = ConcurrentHashMap.newKeySet();
    Set<String> set2400 = ConcurrentHashMap.newKeySet();
    Set<String> set2500 = ConcurrentHashMap.newKeySet();

    private Search() {
    }

    public static Search getInstance() {
        return searchInstance;
    }

    private static void writeRecordToFile(String record, String srcPath) {
        if (!isContained(record, srcPath)) {
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(srcPath, true))) {
                bw.write(record);
                bw.write("\n");
            } catch (IOException ignored) {

            }
        }
    }

    private static boolean isContained(String record, String filePath) {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
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
        return RecycleBin.size();
    }

    public void addToRecycleBin(String path) {
        RecycleBin.add(path);
    }

    public void mergeAndClearRecycleBin() {
        if (!isManualUpdate) {
            isUsable = false;
            try { //垃圾分类
                for (String i : RecycleBin) {
                    SearchBar instance = SearchBar.getInstance();
                    int ascII = instance.getAscIISum(instance.getFileName(i));
                    if (0 <= ascII && ascII <= 100) {
                        set0.add(i);
                    } else if (100 < ascII && ascII <= 200) {
                        set100.add(i);
                    } else if (200 < ascII && ascII <= 300) {
                        set200.add(i);
                    } else if (300 < ascII && ascII <= 400) {
                        set300.add(i);
                    } else if (400 < ascII && ascII <= 500) {
                        set400.add(i);
                    } else if (500 < ascII && ascII <= 600) {
                        set500.add(i);
                    } else if (600 < ascII && ascII <= 700) {
                        set600.add(i);
                    } else if (700 < ascII && ascII <= 800) {
                        set700.add(i);
                    } else if (800 < ascII && ascII <= 900) {
                        set800.add(i);
                    } else if (900 < ascII && ascII <= 1000) {
                        set900.add(i);
                    } else if (1000 < ascII && ascII <= 1100) {
                        set1000.add(i);
                    } else if (1100 < ascII && ascII <= 1200) {
                        set1100.add(i);
                    } else if (1200 < ascII && ascII <= 1300) {
                        set1200.add(i);
                    } else if (1300 < ascII && ascII <= 1400) {
                        set1300.add(i);
                    } else if (1400 < ascII && ascII <= 1500) {
                        set1400.add(i);
                    } else if (1500 < ascII && ascII <= 1600) {
                        set1500.add(i);
                    } else if (1600 < ascII && ascII <= 1700) {
                        set1600.add(i);
                    } else if (1700 < ascII && ascII <= 1800) {
                        set1700.add(i);
                    } else if (1800 < ascII && ascII <= 1900) {
                        set1800.add(i);
                    } else if (1900 < ascII && ascII <= 2000) {
                        set1900.add(i);
                    } else if (2000 < ascII && ascII <= 2100) {
                        set2000.add(i);
                    } else if (2100 < ascII && ascII <= 2200) {
                        set2100.add(i);
                    } else if (2200 < ascII && ascII <= 2300) {
                        set2200.add(i);
                    } else if (2300 < ascII && ascII <= 2400) {
                        set2300.add(i);
                    } else if (2400 < ascII && ascII <= 2500) {
                        set2400.add(i);
                    } else {
                        set2500.add(i);
                    }
                }
                RecycleBin.clear();
                String srcPath;
                String destPath;

                if (!set0.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list0-100.txt";
                    destPath = SettingsFrame.dataPath + "\\_list0-100.txt";
                    generateNewLocalRecord(srcPath, destPath, set0);
                }
                if (!set100.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list100-200.txt";
                    destPath = SettingsFrame.dataPath + "\\_list100-200.txt";
                    generateNewLocalRecord(srcPath, destPath, set100);
                }
                if (!set200.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list200-300.txt";
                    destPath = SettingsFrame.dataPath + "\\_list200-300.txt";
                    generateNewLocalRecord(srcPath, destPath, set200);
                }
                if (!set300.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list300-400.txt";
                    destPath = SettingsFrame.dataPath + "\\_list300-400.txt";
                    generateNewLocalRecord(srcPath, destPath, set300);
                }
                if (!set400.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list400-500.txt";
                    destPath = SettingsFrame.dataPath + "\\_list400-500.txt";
                    generateNewLocalRecord(srcPath, destPath, set400);
                }
                if (!set500.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list500-600.txt";
                    destPath = SettingsFrame.dataPath + "\\_list500-600.txt";
                    generateNewLocalRecord(srcPath, destPath, set500);
                }
                if (!set600.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list600-700.txt";
                    destPath = SettingsFrame.dataPath + "\\_list600-700.txt";
                    generateNewLocalRecord(srcPath, destPath, set600);
                }
                if (!set700.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list700-800.txt";
                    destPath = SettingsFrame.dataPath + "\\_list700-800.txt";
                    generateNewLocalRecord(srcPath, destPath, set700);
                }
                if (!set800.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list800-900.txt";
                    destPath = SettingsFrame.dataPath + "\\_list800-900.txt";
                    generateNewLocalRecord(srcPath, destPath, set800);
                }
                if (!set900.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list900-1000.txt";
                    destPath = SettingsFrame.dataPath + "\\_list900-1000.txt";
                    generateNewLocalRecord(srcPath, destPath, set900);
                }
                if (!set1000.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1000-1100.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1000-1100.txt";
                    generateNewLocalRecord(srcPath, destPath, set1000);
                }
                if (!set1100.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1100-1200.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1100-1200.txt";
                    generateNewLocalRecord(srcPath, destPath, set1100);
                }
                if (!set1200.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1200-1300.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1200-1300.txt";
                    generateNewLocalRecord(srcPath, destPath, set1200);
                }
                if (!set1300.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1300-1400.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1300-1400.txt";
                    generateNewLocalRecord(srcPath, destPath, set1300);
                }
                if (!set1400.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1400-1500.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1400-1500.txt";
                    generateNewLocalRecord(srcPath, destPath, set1400);
                }
                if (!set1500.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1500-1600.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1500-1600.txt";
                    generateNewLocalRecord(srcPath, destPath, set1500);
                }
                if (!set1600.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1600-1700.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1600-1700.txt";
                    generateNewLocalRecord(srcPath, destPath, set1600);
                }
                if (!set1700.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1700-1800.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1700-1800.txt";
                    generateNewLocalRecord(srcPath, destPath, set1700);
                }
                if (!set1800.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1800-1900.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1800-1900.txt";
                    generateNewLocalRecord(srcPath, destPath, set1800);
                }
                if (!set1900.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list1900-2000.txt";
                    destPath = SettingsFrame.dataPath + "\\_list1900-2000.txt";
                    generateNewLocalRecord(srcPath, destPath, set1900);
                }
                if (!set2000.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list2000-2100.txt";
                    destPath = SettingsFrame.dataPath + "\\_list2000-2100.txt";
                    generateNewLocalRecord(srcPath, destPath, set2000);
                }
                if (!set2100.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list2100-2200.txt";
                    destPath = SettingsFrame.dataPath + "\\_list2100-2200.txt";
                    generateNewLocalRecord(srcPath, destPath, set2100);
                }
                if (!set2200.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list2200-2300.txt";
                    destPath = SettingsFrame.dataPath + "\\_list2200-2300.txt";
                    generateNewLocalRecord(srcPath, destPath, set2200);
                }
                if (!set2300.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list2300-2400.txt";
                    destPath = SettingsFrame.dataPath + "\\_list2300-2400.txt";
                    generateNewLocalRecord(srcPath, destPath, set2300);
                }
                if (!set2400.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list2400-2500.txt";
                    destPath = SettingsFrame.dataPath + "\\_list2400-2500.txt";
                    generateNewLocalRecord(srcPath, destPath, set2400);
                }
                if (!set2500.isEmpty()) {
                    srcPath = SettingsFrame.dataPath + "\\list2500-.txt";
                    destPath = SettingsFrame.dataPath + "\\_list2500-.txt";
                    generateNewLocalRecord(srcPath, destPath, set2500);
                }

            } catch (ConcurrentModificationException ignored) {

            } finally {
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
                isUsable = true;
            }
        }
    }

    private void generateNewLocalRecord(String srcPath, String destPath, Set<String> set) {
        File src, target;
        deleteRecordInFile(set, srcPath, destPath);
        src = new File(srcPath);
        src.delete();
        target = new File(destPath);
        target.renameTo(src);
    }

    private void deleteRecordInFile(Set<String> recordToDel, String srcText, String destText) {
        try (BufferedReader br = new BufferedReader(new FileReader(srcText)); BufferedWriter bw = new BufferedWriter(new FileWriter(destText, true))) {
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
            listPath = SettingsFrame.dataPath + "\\list0-100.txt";
            writeRecordToFile(path, listPath);
        } else if (100 < ascII && ascII <= 200) {
            listPath = SettingsFrame.dataPath + "\\list100-200.txt";
            writeRecordToFile(path, listPath);
        } else if (200 < ascII && ascII <= 300) {
            listPath = SettingsFrame.dataPath + "\\list200-300.txt";
            writeRecordToFile(path, listPath);
        } else if (300 < ascII && ascII <= 400) {
            listPath = SettingsFrame.dataPath + "\\list300-400.txt";
            writeRecordToFile(path, listPath);
        } else if (400 < ascII && ascII <= 500) {
            listPath = SettingsFrame.dataPath + "\\list400-500.txt";

            writeRecordToFile(path, listPath);
        } else if (500 < ascII && ascII <= 600) {
            listPath = SettingsFrame.dataPath + "\\list500-600.txt";

            writeRecordToFile(path, listPath);
        } else if (600 < ascII && ascII <= 700) {
            listPath = SettingsFrame.dataPath + "\\list600-700.txt";
            writeRecordToFile(path, listPath);
        } else if (700 < ascII && ascII <= 800) {
            listPath = SettingsFrame.dataPath + "\\list700-800.txt";

            writeRecordToFile(path, listPath);
        } else if (800 < ascII && ascII <= 900) {
            listPath = SettingsFrame.dataPath + "\\list800-900.txt";

            writeRecordToFile(path, listPath);
        } else if (900 < ascII && ascII <= 1000) {
            listPath = SettingsFrame.dataPath + "\\list900-1000.txt";
            writeRecordToFile(path, listPath);
        } else if (1000 < ascII && ascII <= 1100) {
            listPath = SettingsFrame.dataPath + "\\list1000-1100.txt";

            writeRecordToFile(path, listPath);
        } else if (1100 < ascII && ascII <= 1200) {
            listPath = SettingsFrame.dataPath + "\\list1100-1200.txt";
            writeRecordToFile(path, listPath);
        } else if (1200 < ascII && ascII <= 1300) {
            listPath = SettingsFrame.dataPath + "\\list1200-1300.txt";
            writeRecordToFile(path, listPath);
        } else if (1300 < ascII && ascII <= 1400) {
            listPath = SettingsFrame.dataPath + "\\list1300-1400.txt";

            writeRecordToFile(path, listPath);
        } else if (1400 < ascII && ascII <= 1500) {
            listPath = SettingsFrame.dataPath + "\\list1400-1500.txt";

            writeRecordToFile(path, listPath);
        } else if (1500 < ascII && ascII <= 1600) {
            listPath = SettingsFrame.dataPath + "\\list1500-1600.txt";

            writeRecordToFile(path, listPath);
        } else if (1600 < ascII && ascII <= 1700) {
            listPath = SettingsFrame.dataPath + "\\list1600-1700.txt";

            writeRecordToFile(path, listPath);
        } else if (1700 < ascII && ascII <= 1800) {
            listPath = SettingsFrame.dataPath + "\\list1700-1800.txt";

            writeRecordToFile(path, listPath);
        } else if (1800 < ascII && ascII <= 1900) {
            listPath = SettingsFrame.dataPath + "\\list1800-1900.txt";

            writeRecordToFile(path, listPath);
        } else if (1900 < ascII && ascII <= 2000) {
            listPath = SettingsFrame.dataPath + "\\list1900-2000.txt";

            writeRecordToFile(path, listPath);
        } else if (2000 < ascII && ascII <= 2100) {
            listPath = SettingsFrame.dataPath + "\\list2000-2100.txt";

            writeRecordToFile(path, listPath);
        } else if (2100 < ascII && ascII <= 2200) {
            listPath = SettingsFrame.dataPath + "\\list2100-2200.txt";

            writeRecordToFile(path, listPath);
        } else if (2200 < ascII && ascII <= 2300) {
            listPath = SettingsFrame.dataPath + "\\list2200-2300.txt";

            writeRecordToFile(path, listPath);
        } else if (2300 < ascII && ascII <= 2400) {
            listPath = SettingsFrame.dataPath + "\\list2300-2400.txt";

            writeRecordToFile(path, listPath);
        } else if (2400 < ascII && ascII <= 2500) {
            listPath = SettingsFrame.dataPath + "\\list2400-2500.txt";

            writeRecordToFile(path, listPath);
        } else {
            listPath = SettingsFrame.dataPath + "\\list2500-.txt";
            writeRecordToFile(path, listPath);
        }

        if (!isManualUpdate) {
            isUsable = true;
        }
    }


    private void searchFile(String ignorePath, int searchDepth) {
        File[] roots = File.listRoots();
        FileSystemView sys = FileSystemView.getFileSystemView();
        ExecutorService pool = Executors.newFixedThreadPool(roots.length);
        //创建搜索结果存放文件夹
        for (int i = 0; i < SettingsFrame.diskCount; i++) {
            File eachResults = new File(SettingsFrame.dataPath + "\\" + i);
            if (!eachResults.exists()) {
                eachResults.mkdir();
            }
        }
        int count = 0;
        for (File root : roots) {
            String driveType = sys.getSystemTypeDescription(root);
            if (driveType.equals("本地磁盘")) {
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
            for (int j = 0; j < SettingsFrame.diskCount; j++) {
                String _temp = SettingsFrame.dataPath + "\\" + j + "\\list" + i + "-" + name + ".txt";
                String _tempPath = _temp.substring(0, 2) + "\"" + _temp.substring(2) + "\"";
                listPath.append(_tempPath).append("+");
            }
            strb.append(listPath.toString(), 0, listPath.length() - 1).append(" ").append("\"").append(SettingsFrame.dataPath)
                    .append("\\list").append(i).append("-").append(name).append(".txt").append("\"");
            commands.add(strb.toString());
            listPath.delete(0, listPath.length());
            strb.delete(0, strb.length());
            strb.append("cmd.exe /c copy ");
        }
        for (int j = 0; j < SettingsFrame.diskCount; j++) {
            String _temp = SettingsFrame.dataPath + "\\" + j + "\\list2500-.txt";
            String _tempPath = _temp.substring(0, 2) + "\"" + _temp.substring(2) + "\"";
            listPath.append(_tempPath).append("+");
        }
        strb.append(listPath.toString(), 0, listPath.length() - 1).append(" ").append("\"").append(SettingsFrame.dataPath)
                .append("\\list2500-.txt").append("\"");
        commands.add(strb.toString());

        //合并所有搜索结果
        try {
            for (String each : commands) {
                Process p = Runtime.getRuntime().exec(each);
                p.getOutputStream().close();
                p.getErrorStream().close();
                BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String line;
                while ((line = br.readLine()) != null) {
                    System.out.println(line);
                }
                br.close();
                p.waitFor();
            }
        } catch (IOException | InterruptedException ignored) {

        }


        //删除之前的结果
        for (int i = 0; i < SettingsFrame.diskCount; i++) {
            String path = SettingsFrame.dataPath + "\\" + i;
            MainClass.deleteDir(path);
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
        String command = "cmd /c " + start + end + " \"" + path + "\"" + " \"6\" " + "\"" + ignorePath + "\" " + "\"" + SettingsFrame.dataPath + "\\0" + "\" " + "\"" + "1" + "\"";
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
        String command = "cmd /c " + start + end + " \"" + path + "\"" + " \"" + searchDepth + "\" " + "\"" + ignorePath + "\" " + "\"" + SettingsFrame.dataPath + "\\" + count + "\" " + "\"" + "0" + "\"";
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
        MainClass.deleteDir(SettingsFrame.dataPath);
        searchFile(ignorePath, searchDepth);
    }
}