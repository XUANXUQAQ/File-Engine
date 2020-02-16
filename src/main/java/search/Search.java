package search;

import frame.SettingsFrame;
import main.MainClass;
import pinyin.PinYinConverter;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ConcurrentModificationException;
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
            __searchFile(path, searchDepth, ignorePath);
        }
        __searchFileIgnoreSearchDepth(getStartMenu(), ignorePath);
        __searchFileIgnoreSearchDepth("C:\\ProgramData\\Microsoft\\Windows\\Start Menu", ignorePath);
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
        MainClass.deleteDir(SettingsFrame.dataPath);
        searchFile(ignorePath, searchDepth);
    }

}