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
    private static boolean isUsable = false;
    private static boolean isManualUpdate = false;
    private static CopyOnWriteArraySet<String> RecycleBin = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listToLoad = new CopyOnWriteArraySet<>();

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
                for (String i : RecycleBin) {
                    deletePathInList(i);
                }
                RecycleBin.clear();
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

    private void deletePathInList(String path) {
        File file = new File(path);
        char firstWord = '\0';
        try {
            firstWord = PinYinConverter.getPinYin(file.getName()).charAt(0);
        } catch (Exception ignored) {

        }
        char headWord = Character.toUpperCase(firstWord);
        switch (headWord) {
            case 'A':
                StringBuilder strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listA.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listA.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'B':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listB.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listB.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'C':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listC.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listC.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'D':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listD.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listD.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'E':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listE.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listE.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'F':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listF.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listF.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'G':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listG.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listG.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'H':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listH.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listH.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'I':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listI.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listI.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'J':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listJ.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listJ.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'K':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listK.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listK.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'L':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listL.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listL.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'M':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listM.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listM.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'N':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listN.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listN.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'O':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listO.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listO.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'P':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listP.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listP.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'Q':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listQ.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listQ.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'R':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listR.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listR.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'S':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listS.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listS.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'T':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listT.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listT.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'U':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listU.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listU.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'V':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listV.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listV.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'W':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listW.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listW.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'X':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listX.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listX.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'Y':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listY.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listY.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case 'Z':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listZ.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listZ.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case '_':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listUnderline.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listUnderline.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            case '%':
                strb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listPercentSign.txt"))) {
                    String each;
                    while ((each = br.readLine()) != null) {
                        if (!each.equals(path)) {
                            strb.append(each).append("\n");
                        }
                    }
                } catch (Exception ignored) {

                }
                try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listPercentSign.txt"))) {
                    bw.write(strb.toString());
                } catch (IOException ignored) {

                }

                break;
            default:
                if (Character.isDigit(headWord)) {
                    strb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listNum.txt"))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!each.equals(path)) {
                                strb.append(each).append("\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listNum.txt"))) {
                        bw.write(strb.toString());
                    } catch (IOException ignored) {

                    }

                    break;
                } else {
                    strb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new FileReader(SettingsFrame.dataPath + "\\listUnique.txt"))) {
                        String each;
                        while ((each = br.readLine()) != null) {
                            if (!each.equals(path)) {
                                strb.append(each).append("\n");
                            }
                        }
                    } catch (Exception ignored) {

                    }
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listUnique.txt"))) {
                        bw.write(strb.toString());
                    } catch (IOException ignored) {

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
        char firstWord = PinYinConverter.getPinYin(file.getName()).charAt(0);
        if (firstWord != '$' && firstWord != '.') {
            char headWord = Character.toUpperCase(firstWord);
            switch (headWord) {
                case 'A':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listA.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'B':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listB.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'C':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listC.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'D':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listD.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'E':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listE.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'F':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listF.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'G':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listG.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'H':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listH.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'I':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listI.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'J':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listJ.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'K':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listK.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'L':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listL.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'M':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listM.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'N':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listN.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'O':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listO.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'P':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listP.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'Q':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listQ.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'R':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listR.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'S':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listS.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'T':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listT.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'U':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listU.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'V':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listV.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'W':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listW.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'X':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listX.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'Y':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listY.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'Z':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listZ.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case '_':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listUnderline.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }
                    break;
                case '%':
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listPercentSign.txt", true))) {
                        bw.write(path);
                    } catch (Exception ignored) {

                    }
                default:
                    if (Character.isDigit(headWord)) {
                        try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listNum.txt", true))) {
                            bw.write(path);
                        } catch (Exception ignored) {

                        }
                    } else {
                        try (BufferedWriter bw = new BufferedWriter(new FileWriter(SettingsFrame.dataPath + "\\listUnique.txt", true))) {
                            bw.write(path);
                        } catch (Exception ignored) {

                        }
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
            try {
                __searchFile(path, searchDepth, ignorePath);
            } catch (IOException ignored) {

            }
        }
        try {
            __searchFileIgnoreSearchDepth(getStartMenu(), ignorePath);
            __searchFileIgnoreSearchDepth("C:\\ProgramData\\Microsoft\\Windows\\Start Menu", ignorePath);
        } catch (IOException ignored) {

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

    private void __searchFileIgnoreSearchDepth(String path, String ignorePath) throws IOException {
        File searchResults = new File(SettingsFrame.tmp + "\\searchResults.txt");
        String command = "cmd /c fileSearcher.exe \"" + path + "\"" + " \"6\" " + "\"" + ignorePath + "\" " + "\"" + searchResults.getAbsolutePath() + "\" " + "\"" + "1" + "\"";
        String each;
        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            while (p.isAlive()) {
                Thread.sleep(1);
            }
        } catch (IOException | InterruptedException ignored) {

        }
        BufferedWriter bwA = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listA.txt", true)));
        BufferedWriter bwB = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listB.txt", true)));
        BufferedWriter bwC = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listC.txt", true)));
        BufferedWriter bwD = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listD.txt", true)));
        BufferedWriter bwE = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listE.txt", true)));
        BufferedWriter bwF = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listF.txt", true)));
        BufferedWriter bwG = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listG.txt", true)));
        BufferedWriter bwH = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listH.txt", true)));
        BufferedWriter bwI = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listI.txt", true)));
        BufferedWriter bwJ = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listJ.txt", true)));
        BufferedWriter bwK = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listK.txt", true)));
        BufferedWriter bwL = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listL.txt", true)));
        BufferedWriter bwM = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listM.txt", true)));
        BufferedWriter bwN = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listN.txt", true)));
        BufferedWriter bwO = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listO.txt", true)));
        BufferedWriter bwP = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listP.txt", true)));
        BufferedWriter bwQ = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listQ.txt", true)));
        BufferedWriter bwR = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listR.txt", true)));
        BufferedWriter bwS = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listS.txt", true)));
        BufferedWriter bwT = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listT.txt", true)));
        BufferedWriter bwU = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listU.txt", true)));
        BufferedWriter bwV = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listV.txt", true)));
        BufferedWriter bwW = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listW.txt", true)));
        BufferedWriter bwX = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listX.txt", true)));
        BufferedWriter bwY = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listY.txt", true)));
        BufferedWriter bwZ = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listZ.txt", true)));
        BufferedWriter bwNum = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listNum.txt", true)));
        BufferedWriter bwUnderline = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listUnderline.txt", true)));
        BufferedWriter bwPercentSign = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listPercentSign.txt", true)));
        BufferedWriter bwUnique = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listUnique.txt", true)));
        try (BufferedReader buffr = new BufferedReader(new InputStreamReader(new FileInputStream(searchResults)))) {
            while ((each = buffr.readLine()) != null) {
                File tmp = new File(each);
                String name = tmp.getName();
                name = PinYinConverter.getPinYin(name);
                HashSet<String> set = new HashSet<>();
                boolean isNumAdded = false;
                for (char i : name.toCharArray()) {
                    if (Character.isAlphabetic(i)) {
                        set.add(String.valueOf(i));
                    }
                    if (Character.isDigit(i) && !isNumAdded) {
                        set.add(String.valueOf(i));
                        isNumAdded = true;
                    }
                    if (set.size() > 7) {
                        break;
                    }
                }
                for (String headWord : set) {
                    headWord = headWord.toUpperCase();
                    switch (headWord) {
                        case "A":
                            bwA.write(each + "\n");
                            break;
                        case "B":
                            bwB.write(each + "\n");
                            break;
                        case "C":
                            bwC.write(each + "\n");
                            break;
                        case "D":
                            bwD.write(each + "\n");
                            break;
                        case "E":
                            bwE.write(each + "\n");
                            break;
                        case "F":
                            bwF.write(each + "\n");
                            break;
                        case "G":
                            bwG.write(each + "\n");
                            break;
                        case "H":
                            bwH.write(each + "\n");
                            break;
                        case "I":
                            bwI.write(each + "\n");
                            break;
                        case "J":
                            bwJ.write(each + "\n");
                            break;
                        case "K":
                            bwK.write(each + "\n");
                            break;
                        case "L":
                            bwL.write(each + "\n");
                            break;
                        case "M":
                            bwM.write(each + "\n");
                            break;
                        case "N":
                            bwN.write(each + "\n");
                            break;
                        case "O":
                            bwO.write(each + "\n");
                            break;
                        case "P":
                            bwP.write(each + "\n");
                            break;
                        case "Q":
                            bwQ.write(each + "\n");
                            break;
                        case "R":
                            bwR.write(each + "\n");
                            break;
                        case "S":
                            bwS.write(each + "\n");
                            break;
                        case "T":
                            bwT.write(each + "\n");
                            break;
                        case "U":
                            bwU.write(each + "\n");
                            break;
                        case "V":
                            bwV.write(each + "\n");
                            break;
                        case "W":
                            bwW.write(each + "\n");
                            break;
                        case "X":
                            bwX.write(each + "\n");
                            break;
                        case "Y":
                            bwY.write(each + "\n");
                            break;
                        case "Z":
                            bwZ.write(each + "\n");
                            break;
                        case "_":
                            bwUnderline.write(each + "\n");
                            break;
                        case "%":
                            bwPercentSign.write(each + "\n");
                            break;
                        default:
                            if (Character.isDigit(headWord.charAt(0))) {
                                bwNum.write(each + "\n");
                            } else {
                                bwUnique.write(each + "\n");
                            }
                            break;
                    }
                }
            }
        } catch (Exception ignored) {

        } finally {
            bwA.close();
            bwB.close();
            bwC.close();
            bwD.close();
            bwE.close();
            bwF.close();
            bwG.close();
            bwH.close();
            bwI.close();
            bwK.close();
            bwL.close();
            bwM.close();
            bwN.close();
            bwO.close();
            bwP.close();
            bwQ.close();
            bwR.close();
            bwS.close();
            bwT.close();
            bwU.close();
            bwV.close();
            bwW.close();
            bwX.close();
            bwY.close();
            bwZ.close();
            bwNum.close();
            bwPercentSign.close();
            bwUnderline.close();
            bwUnique.close();
        }
    }

    private void __searchFile(String path, int searchDepth, String ignorePath) throws IOException {
        File searchResults = new File(SettingsFrame.tmp + "\\searchResults.txt");
        String command = "cmd /c fileSearcher.exe \"" + path + "\"" + " \"" + searchDepth + "\" " + "\"" + ignorePath + "\" " + "\"" + searchResults.getAbsolutePath() + "\" " + "\"" + "0" + "\"";
        String each;
        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            while (p.isAlive()) {
                Thread.sleep(1);
            }
        } catch (IOException | InterruptedException ignored) {

        }

        BufferedWriter bwA = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listA.txt", true)));
        BufferedWriter bwB = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listB.txt", true)));
        BufferedWriter bwC = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listC.txt", true)));
        BufferedWriter bwD = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listD.txt", true)));
        BufferedWriter bwE = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listE.txt", true)));
        BufferedWriter bwF = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listF.txt", true)));
        BufferedWriter bwG = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listG.txt", true)));
        BufferedWriter bwH = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listH.txt", true)));
        BufferedWriter bwI = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listI.txt", true)));
        BufferedWriter bwJ = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listJ.txt", true)));
        BufferedWriter bwK = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listK.txt", true)));
        BufferedWriter bwL = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listL.txt", true)));
        BufferedWriter bwM = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listM.txt", true)));
        BufferedWriter bwN = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listN.txt", true)));
        BufferedWriter bwO = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listO.txt", true)));
        BufferedWriter bwP = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listP.txt", true)));
        BufferedWriter bwQ = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listQ.txt", true)));
        BufferedWriter bwR = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listR.txt", true)));
        BufferedWriter bwS = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listS.txt", true)));
        BufferedWriter bwT = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listT.txt", true)));
        BufferedWriter bwU = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listU.txt", true)));
        BufferedWriter bwV = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listV.txt", true)));
        BufferedWriter bwW = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listW.txt", true)));
        BufferedWriter bwX = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listX.txt", true)));
        BufferedWriter bwY = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listY.txt", true)));
        BufferedWriter bwZ = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listZ.txt", true)));
        BufferedWriter bwNum = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listNum.txt", true)));
        BufferedWriter bwUnderline = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listUnderline.txt", true)));
        BufferedWriter bwPercentSign = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listPercentSign.txt", true)));
        BufferedWriter bwUnique = new BufferedWriter((new FileWriter(SettingsFrame.dataPath + "\\listUnique.txt", true)));

        try (BufferedReader buffr = new BufferedReader(new InputStreamReader(new FileInputStream(searchResults)))) {
            while ((each = buffr.readLine()) != null) {
                File tmp = new File(each);
                String name = tmp.getName();
                name = PinYinConverter.getPinYin(name);
                HashSet<String> set = new HashSet<>();
                boolean isNumAdded = false;
                for (char i : name.toCharArray()) {
                    if (Character.isAlphabetic(i)) {
                        set.add(String.valueOf(i));
                    }
                    if (Character.isDigit(i) && !isNumAdded) {
                        set.add(String.valueOf(i));
                        isNumAdded = true;
                    }
                    if (set.size() > 7) {
                        break;
                    }
                }
                for (String headWord : set) {
                    headWord = headWord.toUpperCase();
                    switch (headWord) {
                        case "A":
                            bwA.write(each + "\n");
                            break;
                        case "B":
                            bwB.write(each + "\n");
                            break;
                        case "C":
                            bwC.write(each + "\n");
                            break;
                        case "D":
                            bwD.write(each + "\n");
                            break;
                        case "E":
                            bwE.write(each + "\n");
                            break;
                        case "F":
                            bwF.write(each + "\n");
                            break;
                        case "G":
                            bwG.write(each + "\n");
                            break;
                        case "H":
                            bwH.write(each + "\n");
                            break;
                        case "I":
                            bwI.write(each + "\n");
                            break;
                        case "J":
                            bwJ.write(each + "\n");
                            break;
                        case "K":
                            bwK.write(each + "\n");
                            break;
                        case "L":
                            bwL.write(each + "\n");
                            break;
                        case "M":
                            bwM.write(each + "\n");
                            break;
                        case "N":
                            bwN.write(each + "\n");
                            break;
                        case "O":
                            bwO.write(each + "\n");
                            break;
                        case "P":
                            bwP.write(each + "\n");
                            break;
                        case "Q":
                            bwQ.write(each + "\n");
                            break;
                        case "R":
                            bwR.write(each + "\n");
                            break;
                        case "S":
                            bwS.write(each + "\n");
                            break;
                        case "T":
                            bwT.write(each + "\n");
                            break;
                        case "U":
                            bwU.write(each + "\n");
                            break;
                        case "V":
                            bwV.write(each + "\n");
                            break;
                        case "W":
                            bwW.write(each + "\n");
                            break;
                        case "X":
                            bwX.write(each + "\n");
                            break;
                        case "Y":
                            bwY.write(each + "\n");
                            break;
                        case "Z":
                            bwZ.write(each + "\n");
                            break;
                        case "_":
                            bwUnderline.write(each + "\n");
                            break;
                        case "%":
                            bwPercentSign.write(each + "\n");
                            break;
                        default:
                            if (Character.isDigit(headWord.charAt(0))) {
                                bwNum.write(each + "\n");
                            } else {
                                bwUnique.write(each + "\n");
                            }
                            break;
                    }
                }
            }
        } catch (Exception ignored) {

        } finally {
            bwA.close();
            bwB.close();
            bwC.close();
            bwD.close();
            bwE.close();
            bwF.close();
            bwG.close();
            bwH.close();
            bwI.close();
            bwK.close();
            bwL.close();
            bwM.close();
            bwN.close();
            bwO.close();
            bwP.close();
            bwQ.close();
            bwR.close();
            bwS.close();
            bwT.close();
            bwU.close();
            bwV.close();
            bwW.close();
            bwX.close();
            bwY.close();
            bwZ.close();
            bwNum.close();
            bwPercentSign.close();
            bwUnderline.close();
            bwUnique.close();
        }
    }


    public void updateLists(String ignorePath, int searchDepth) {
        MainClass.deleteDir(SettingsFrame.dataPath);
        searchFile(ignorePath, searchDepth);
    }

}