package search;

import frame.SettingsFrame;
import main.MainClass;
import pinyin.PinYinConverter;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ConcurrentModificationException;
import java.util.HashSet;
import java.util.concurrent.CopyOnWriteArraySet;


public class Search {
    private static boolean isUsable = true;
    private static boolean isManualUpdate = false;
    private CopyOnWriteArraySet<String> RecycleBin = new CopyOnWriteArraySet<>();
    private CopyOnWriteArraySet<String> listToLoad = new CopyOnWriteArraySet<>();
    private MainClass mainInstance = MainClass.getInstance();

    private static void addFileToRecord(String record, String srcPath) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(srcPath, true))) {
            bw.write(record);
        } catch (IOException ignored) {

        }
    }

    public int getRecycleBinSize() {
        return RecycleBin.size();
    }

    public int getLoadListSize() {
        return listToLoad.size();
    }

    public void addToRecycleBin(String path) {
        RecycleBin.add(path);
    }

    public void mergeAndClearRecycleBin() {
        if (!isManualUpdate) {
            isUsable = false;
            try {
                HashSet<String> listChars = new HashSet<>();
                boolean isNumAdded = false;
                for (String each : RecycleBin) { //统计需要在哪些文件中进行修改
                    File file = new File(each);
                    for (char i : PinYinConverter.getPinYin(file.getName()).toCharArray()) {
                        if (!Character.isDigit(i)) {
                            if (Character.isAlphabetic(i)) {
                                listChars.add(String.valueOf(i).toUpperCase());
                            } else {
                                listChars.add("Unique");
                            }
                        }
                        if (Character.isDigit(i) && !isNumAdded) {
                            listChars.add("Num");
                            isNumAdded = true;
                        }
                    }
                }
                deletePathInList(RecycleBin, listChars);
                RecycleBin.clear();
                for (String i : listChars) {
                    String target = SettingsFrame.dataPath + "\\list" + i + ".txt";
                    File old = new File(target);
                    try {
                        old.delete();
                    } catch (Exception ignored) {

                    }
                }
                for (String i : listChars) {
                    String file = SettingsFrame.dataPath + "\\_list" + i + ".txt";
                    File newFile = new File(file);
                    try {
                        newFile.renameTo(new File(SettingsFrame.dataPath + "\\list" + i + ".txt"));
                    } catch (Exception ignored) {

                    }
                }
            } catch (ConcurrentModificationException ignored) {

            } finally {
                isUsable = true;
            }
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
                    addFileToList(each);
                }
            }
            isUsable = true;
            listToLoad.clear();
        }
    }

    private void deletePathInList(CopyOnWriteArraySet<String> path, HashSet<String> listChars) {
        for (String headWord : listChars) {
            headWord = headWord.toUpperCase();

            switch (headWord) {
                case "A":
                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listA.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listA.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }

                    break;
                case "B":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listB.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listB.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }

                    break;
                case "C":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listC.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listC.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "D":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listD.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listD.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "E":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listE.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listE.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "F":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listF.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listF.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "G":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listG.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listG.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "H":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listH.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listH.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "I":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listI.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listI.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "J":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listJ.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listJ.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "K":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listK.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listK.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "L":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listL.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listL.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "M":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listM.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listM.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "N":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listN.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listN.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "O":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listO.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listO.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "P":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listP.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listP.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "Q":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listQ.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listQ.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "R":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listR.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listR.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "S":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listS.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listS.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "T":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listT.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listT.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "U":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listU.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listU.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "V":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listV.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listV.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "W":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listW.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listW.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "X":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listX.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listX.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "Y":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listY.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listY.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "Z":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listZ.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listZ.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }


                    break;
                case "_":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listUnderline.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listUnderline.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }

                    break;
                case "%":

                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listPercentSign.txt"));
                         BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listPercentSign.txt", true))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!path.contains(each)) {
                                bw.write(each + "\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }

                    break;
                default:
                    if (Character.isDigit(headWord.charAt(0))) {

                        try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listNum.txt"));
                             BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listNum.txt", true))) {
                            String each;
                            while ((each = br.readLine()) != null) {
                                if (!path.contains(each)) {
                                    bw.write(each + "\n");
                                }
                            }
                        } catch (Exception ignored) {

                        }

                    } else {

                        try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listUnique.txt"));
                             BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\_listUnique.txt", true))) {
                            String each;
                            while ((each = br.readLine()) != null) {
                                if (!path.contains(each)) {
                                    bw.write(each + "\n");
                                }
                            }
                        } catch (Exception ignored) {

                        }

                    }
                    break;
            }
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

    private void addFileToList(String path) {
        File file = new File(path);
        HashSet<String> listChars = new HashSet<>();
        boolean isNumAdded = false;
        for (char i : PinYinConverter.getPinYin(file.getName()).toCharArray()) {
            if (Character.isAlphabetic(i)) {
                listChars.add(String.valueOf(i));
            }
            if (Character.isDigit(i) && !isNumAdded) {
                listChars.add(String.valueOf(i));
                isNumAdded = true;
            }
        }
        for (String headWord : listChars) {
            headWord = headWord.toUpperCase();

            switch (headWord) {
                case "A":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listA.txt");
                    break;
                case "B":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listB.txt");


                    break;
                case "C":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listC.txt");


                    break;
                case "D":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listD.txt");


                    break;
                case "E":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listE.txt");


                    break;
                case "F":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listF.txt");


                    break;
                case "G":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listG.txt");


                    break;
                case "H":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listH.txt");


                    break;
                case "I":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listI.txt");


                    break;
                case "J":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listJ.txt");


                    break;
                case "K":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listK.txt");


                    break;
                case "L":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listL.txt");


                    break;
                case "M":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listM.txt");


                    break;
                case "N":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listN.txt");


                    break;
                case "O":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listO.txt");


                    break;
                case "P":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listP.txt");


                    break;
                case "Q":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listQ.txt");


                    break;
                case "R":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listR.txt");


                    break;
                case "S":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listS.txt");


                    break;
                case "T":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listT.txt");


                    break;
                case "U":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listU.txt");


                    break;
                case "V":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listV.txt");


                    break;
                case "W":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listW.txt");


                    break;
                case "X":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listX.txt");


                    break;
                case "Y":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listY.txt");


                    break;
                case "Z":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listZ.txt");


                    break;
                case "_":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listUnderline.txt");

                    break;
                case "%":
                    addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listPercentSign.txt");
                    break;
                default:
                    if (Character.isDigit(headWord.charAt(0))) {
                        addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listNum.txt");

                    } else {
                        addFileToRecord(path + "\n", SettingsFrame.dataPath + "\\listUnique.txt");
                    }

                    break;
            }
        }
        if (!isManualUpdate) {
            isUsable = true;
        }
    }


    private void searchFile(String ignorePath, int searchDepth) {
        File[] roots = File.listRoots();
        for (File root : roots) {
            String path = root.getAbsolutePath();
            path = path.substring(0, 2);
            __searchFile(path, searchDepth, ignorePath);
        }
        __searchFileIgnoreSearchDepth(getStartMenu(), ignorePath);
        __searchFileIgnoreSearchDepth("C:\\ProgramData\\Microsoft\\Windows\\Start Menu", ignorePath);
        mainInstance.showMessage("提示", "搜索完成");
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
        String command = "cmd /c fileSearcher.exe \"" + path + "\"" + " \"6\" " + "\"" + ignorePath + "\" " + "\"" + SettingsFrame.dataPath + "\" " + "\"" + "1" + "\"";
        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            while (p.isAlive()) {
                Thread.sleep(1);
            }
        } catch (IOException | InterruptedException ignored) {

        }
    }

    private void __searchFile(String path, int searchDepth, String ignorePath) {
        String command = "cmd /c fileSearcher.exe \"" + path + "\"" + " \"" + searchDepth + "\" " + "\"" + ignorePath + "\" " + "\"" + SettingsFrame.dataPath + "\" " + "\"" + "0" + "\"";
        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            while (p.isAlive()) {
                Thread.sleep(1);
            }
        } catch (IOException | InterruptedException ignored) {

        }
    }


    public void updateLists(String ignorePath, int searchDepth) {
        mainInstance.deleteDir(SettingsFrame.dataPath);
        searchFile(ignorePath, searchDepth);
    }

}