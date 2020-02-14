package search;

import frame.SettingsFrame;
import main.MainClass;
import pinyin.PinYinConverter;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CopyOnWriteArraySet;


public class Search {
    private static CopyOnWriteArraySet<String> listA = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listB = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listC = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listD = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listE = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listF = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listG = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listH = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listI = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listJ = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listK = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listL = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listM = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listN = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listO = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listP = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listQ = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listR = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listS = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listT = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listU = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listV = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listW = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listX = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listY = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listZ = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listNum = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listPercentSign = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listUnique = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<String> listUnderline = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArrayList<String> listToLoad = new CopyOnWriteArrayList<>();
    private static boolean isUsable = false;
    private static boolean isManualUpdate = false;
    private static CopyOnWriteArraySet<String> RecycleBin = new CopyOnWriteArraySet<>();

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
                    RecycleBin.clear();
                }
            } catch (ConcurrentModificationException ignored) {

            } finally {
                isUsable = true;
            }
        }
    }

    private void updateLocalList(CopyOnWriteArraySet<String> list, String ch) {
        File data = new File(SettingsFrame.dataPath);
        String path;
        Serialize listFile;
        if (!data.exists()) {
            data.mkdir();
        }
        switch (ch) {
            case "A":
                path = SettingsFrame.dataPath + "\\listA";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "B":
                path = SettingsFrame.dataPath + "\\listB";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listB.clear();

                } catch (IOException ignored) {

                }
                break;
            case "C":
                path = SettingsFrame.dataPath + "\\listC";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listC.clear();

                } catch (IOException ignored) {

                }
                break;
            case "D":

                path = SettingsFrame.dataPath + "\\listD";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listD.clear();

                } catch (IOException ignored) {

                }
                break;
            case "E":

                path = SettingsFrame.dataPath + "\\listE";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listE.clear();

                } catch (IOException ignored) {

                }
                break;
            case "F":

                path = SettingsFrame.dataPath + "\\listF";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listF.clear();

                } catch (IOException ignored) {

                }
                break;
            case "G":

                path = SettingsFrame.dataPath + "\\listG";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listG.clear();

                } catch (IOException ignored) {

                }
                break;
            case "H":

                path = SettingsFrame.dataPath + "\\listH";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listH.clear();

                } catch (IOException ignored) {

                }
                break;
            case "I":

                path = SettingsFrame.dataPath + "\\listI";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listI.clear();

                } catch (IOException ignored) {

                }
                break;
            case "J":

                path = SettingsFrame.dataPath + "\\listJ";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listJ.clear();

                } catch (IOException ignored) {

                }
                break;
            case "K":

                path = SettingsFrame.dataPath + "\\listK";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listK.clear();

                } catch (IOException ignored) {

                }
                break;
            case "L":

                path = SettingsFrame.dataPath + "\\listL";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listL.clear();

                } catch (IOException ignored) {

                }
                break;
            case "M":

                path = SettingsFrame.dataPath + "\\listM";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listM.clear();

                } catch (IOException ignored) {

                }
                break;
            case "N":

                path = SettingsFrame.dataPath + "\\listN";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listN.clear();

                } catch (IOException ignored) {

                }
                break;
            case "O":

                path = SettingsFrame.dataPath + "\\listO";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listO.clear();

                } catch (IOException ignored) {

                }
                break;
            case "P":

                path = SettingsFrame.dataPath + "\\listP";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listP.clear();

                } catch (IOException ignored) {

                }
                break;
            case "Q":

                path = SettingsFrame.dataPath + "\\listQ";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listQ.clear();

                } catch (IOException ignored) {

                }
                break;
            case "R":

                path = SettingsFrame.dataPath + "\\listR";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listR.clear();

                } catch (IOException ignored) {

                }
                break;
            case "S":

                path = SettingsFrame.dataPath + "\\listS";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listS.clear();

                } catch (IOException ignored) {

                }
                break;
            case "T":

                path = SettingsFrame.dataPath + "\\listT";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listT.clear();

                } catch (IOException ignored) {

                }
                break;
            case "U":

                path = SettingsFrame.dataPath + "\\listU";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listU.clear();

                } catch (IOException ignored) {

                }
                break;
            case "V":

                path = SettingsFrame.dataPath + "\\listV";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listV.clear();

                } catch (IOException ignored) {

                }
                break;
            case "W":

                path = SettingsFrame.dataPath + "\\listW";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listW.clear();

                } catch (IOException ignored) {

                }
                break;
            case "X":

                path = SettingsFrame.dataPath + "\\listX";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listX.clear();

                } catch (IOException ignored) {

                }
                break;
            case "Y":

                path = SettingsFrame.dataPath + "\\listY";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listY.clear();

                } catch (IOException ignored) {

                }
                break;
            case "Z":

                path = SettingsFrame.dataPath + "\\listZ";

                listFile = new Serialize();
                listFile.setList(list);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listZ.clear();

                } catch (IOException ignored) {

                }
                break;
            case "PercentSign":
                path = SettingsFrame.dataPath + "\\listPercentSign";

                listFile = new Serialize();
                listFile.setList(listPercentSign);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listPercentSign.clear();

                } catch (IOException ignored) {

                }
                break;
            case "Underline":

                path = SettingsFrame.dataPath + "\\listUnderline";

                listFile = new Serialize();
                listFile.setList(listUnderline);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listUnderline.clear();

                } catch (IOException ignored) {

                }
                break;
            case "Unique":

                path = SettingsFrame.dataPath + "\\listUnique";

                listFile = new Serialize();
                listFile.setList(listUnique);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listUnique.clear();

                } catch (IOException ignored) {

                }
                break;
            case "Num":
                path = SettingsFrame.dataPath + "\\listNum";

                listFile = new Serialize();
                listFile.setList(listNum);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);
                    listNum.clear();

                } catch (IOException ignored) {

                }
                break;
        }
    }

    private void saveAndReleaseList(String ch) {
        File data = new File(SettingsFrame.dataPath);
        String path;
        Serialize listFile;
        if (!data.exists()) {
            data.mkdir();
        }
        switch (ch) {
            case "A":
                path = SettingsFrame.dataPath + "\\listA";
            {
                try {
                    listA.addAll(Objects.requireNonNull(readFileList("A")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listA);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listA.clear();

            } catch (IOException ignored) {

            }
            break;
            case "B":
                path = SettingsFrame.dataPath + "\\listB";
            {
                try {
                    listB.addAll(Objects.requireNonNull(readFileList("B")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listB);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listB.clear();

            } catch (IOException ignored) {

            }
            break;
            case "C":
                path = SettingsFrame.dataPath + "\\listC";
            {
                try {
                    listC.addAll(Objects.requireNonNull(readFileList("C")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listC);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listC.clear();

            } catch (IOException ignored) {

            }
            break;
            case "D":

                path = SettingsFrame.dataPath + "\\listD";
            {
                try {
                    listD.addAll(Objects.requireNonNull(readFileList("D")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listD);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listD.clear();

            } catch (IOException ignored) {

            }
            break;
            case "E":

                path = SettingsFrame.dataPath + "\\listE";
            {
                try {
                    listE.addAll(Objects.requireNonNull(readFileList("E")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listE);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listE.clear();

            } catch (IOException ignored) {

            }
            break;
            case "F":

                path = SettingsFrame.dataPath + "\\listF";
            {
                try {
                    listF.addAll(Objects.requireNonNull(readFileList("F")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listF);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listF.clear();

            } catch (IOException ignored) {

            }
            break;
            case "G":

                path = SettingsFrame.dataPath + "\\listG";
            {
                try {
                    listG.addAll(Objects.requireNonNull(readFileList("G")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listG);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listG.clear();

            } catch (IOException ignored) {

            }
            break;
            case "H":

                path = SettingsFrame.dataPath + "\\listH";
            {
                try {
                    listH.addAll(Objects.requireNonNull(readFileList("H")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listH);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listH.clear();

            } catch (IOException ignored) {

            }
            break;
            case "I":

                path = SettingsFrame.dataPath + "\\listI";
            {
                try {
                    listI.addAll(Objects.requireNonNull(readFileList("I")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listI);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listI.clear();

            } catch (IOException ignored) {

            }
            break;
            case "J":

                path = SettingsFrame.dataPath + "\\listJ";
            {
                try {
                    listJ.addAll(Objects.requireNonNull(readFileList("J")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listJ);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listJ.clear();

            } catch (IOException ignored) {

            }
            break;
            case "K":

                path = SettingsFrame.dataPath + "\\listK";
            {
                try {
                    listK.addAll(Objects.requireNonNull(readFileList("K")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listK);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listK.clear();

            } catch (IOException ignored) {

            }
            break;
            case "L":

                path = SettingsFrame.dataPath + "\\listL";
            {
                try {
                    listL.addAll(Objects.requireNonNull(readFileList("L")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listL);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listL.clear();

            } catch (IOException ignored) {

            }
            break;
            case "M":

                path = SettingsFrame.dataPath + "\\listM";
            {
                try {
                    listM.addAll(Objects.requireNonNull(readFileList("M")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listM);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listM.clear();

            } catch (IOException ignored) {

            }
            break;
            case "N":

                path = SettingsFrame.dataPath + "\\listN";
            {
                try {
                    listN.addAll(Objects.requireNonNull(readFileList("N")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listN);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listN.clear();

            } catch (IOException ignored) {

            }
            break;
            case "O":

                path = SettingsFrame.dataPath + "\\listO";
            {
                try {
                    listO.addAll(Objects.requireNonNull(readFileList("O")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listO);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listO.clear();

            } catch (IOException ignored) {

            }
            break;
            case "P":

                path = SettingsFrame.dataPath + "\\listP";
            {
                try {
                    listP.addAll(Objects.requireNonNull(readFileList("P")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listP);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listP.clear();

            } catch (IOException ignored) {

            }
            break;
            case "Q":

                path = SettingsFrame.dataPath + "\\listQ";
            {
                try {
                    listQ.addAll(Objects.requireNonNull(readFileList("Q")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listQ);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listQ.clear();

            } catch (IOException ignored) {

            }
            break;
            case "R":

                path = SettingsFrame.dataPath + "\\listR";
            {
                try {
                    listR.addAll(Objects.requireNonNull(readFileList("R")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listR);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listR.clear();

            } catch (IOException ignored) {

            }
            break;
            case "S":

                path = SettingsFrame.dataPath + "\\listS";
            {
                try {
                    listS.addAll(Objects.requireNonNull(readFileList("S")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listS);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listS.clear();

            } catch (IOException ignored) {

            }
            break;
            case "T":

                path = SettingsFrame.dataPath + "\\listT";
            {
                try {
                    listT.addAll(Objects.requireNonNull(readFileList("T")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listT);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listT.clear();

            } catch (IOException ignored) {

            }
            break;
            case "U":

                path = SettingsFrame.dataPath + "\\listU";
            {
                try {
                    listU.addAll(Objects.requireNonNull(readFileList("U")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listU);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listU.clear();

            } catch (IOException ignored) {

            }
            break;
            case "V":

                path = SettingsFrame.dataPath + "\\listV";
            {
                try {
                    listV.addAll(Objects.requireNonNull(readFileList("V")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listV);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listV.clear();

            } catch (IOException ignored) {

            }
            break;
            case "W":

                path = SettingsFrame.dataPath + "\\listW";
            {
                try {
                    listW.addAll(Objects.requireNonNull(readFileList("W")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listW);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listW.clear();

            } catch (IOException ignored) {

            }
            break;
            case "X":

                path = SettingsFrame.dataPath + "\\listX";
            {
                try {
                    listX.addAll(Objects.requireNonNull(readFileList("X")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listX);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listX.clear();

            } catch (IOException ignored) {

            }
            break;
            case "Y":

                path = SettingsFrame.dataPath + "\\listY";
            {
                try {
                    listY.addAll(Objects.requireNonNull(readFileList("Y")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listY);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listY.clear();

            } catch (IOException ignored) {

            }
            break;
            case "Z":

                path = SettingsFrame.dataPath + "\\listZ";
            {
                try {
                    listZ.addAll(Objects.requireNonNull(readFileList("Z")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listZ);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listZ.clear();

            } catch (IOException ignored) {

            }
            break;
            case "PercentSign":
                path = SettingsFrame.dataPath + "\\listPercentSign";
            {
                try {
                    listPercentSign.addAll(Objects.requireNonNull(readFileList("PercentSign")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listPercentSign);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listPercentSign.clear();

            } catch (IOException ignored) {

            }
            break;
            case "Underline":

                path = SettingsFrame.dataPath + "\\listUnderline";
            {
                try {
                    listUnderline.addAll(Objects.requireNonNull(readFileList("Underline")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listUnderline);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listUnderline.clear();

            } catch (IOException ignored) {

            }
            break;
            case "Unique":

                path = SettingsFrame.dataPath + "\\listUnique";
            {
                try {
                    listUnique.addAll(Objects.requireNonNull(readFileList("Unique")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listUnique);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listUnique.clear();

            } catch (IOException ignored) {

            }
            break;
            case "Num":
                path = SettingsFrame.dataPath + "\\listNum";
            {
                try {
                    listNum.addAll(Objects.requireNonNull(readFileList("Num")));
                } catch (NullPointerException ignored) {

                }
            }
            listFile = new Serialize();
            listFile.setList(listNum);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                oos.writeObject(listFile);
                listNum.clear();

            } catch (IOException ignored) {

            }
            break;
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
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("A")));
                    list.remove(path);
                    listA.remove(path);
                    list.addAll(listA);
                    updateLocalList(list, "A");
                } catch (Exception ignored) {

                }

                break;
            case 'B':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("B")));
                    list.remove(path);
                    listB.remove(path);
                    list.addAll(listB);
                    updateLocalList(list, "B");
                } catch (Exception ignored) {

                }

                break;
            case 'C':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("C")));
                    list.remove(path);
                    listC.remove(path);
                    list.addAll(listC);
                    updateLocalList(list, "C");
                } catch (Exception ignored) {

                }

                break;
            case 'D':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("D")));
                    list.remove(path);
                    listD.remove(path);
                    list.addAll(listD);
                    updateLocalList(list, "D");
                } catch (Exception ignored) {

                }

                break;
            case 'E':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("E")));
                    list.remove(path);
                    listE.remove(path);
                    list.addAll(listE);
                    updateLocalList(list, "E");

                } catch (Exception ignored) {

                }

                break;
            case 'F':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("F")));
                    list.remove(path);
                    listF.remove(path);
                    list.addAll(listF);
                    updateLocalList(list, "F");
                } catch (Exception ignored) {

                }

                break;
            case 'G':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("G")));
                    list.remove(path);
                    listG.remove(path);
                    list.addAll(listG);
                    updateLocalList(list, "G");
                } catch (Exception ignored) {

                }

                break;
            case 'H':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("H")));
                    list.remove(path);
                    listH.remove(path);
                    list.addAll(listH);
                    updateLocalList(list, "H");
                } catch (Exception ignored) {

                }

                break;
            case 'I':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("I")));
                    list.remove(path);
                    listI.remove(path);
                    list.addAll(listI);
                    updateLocalList(list, "i");
                } catch (Exception ignored) {

                }

                break;
            case 'J':
                try {
                   CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("J")));
                   list.remove(path);
                    listJ.remove(path);
                   list.addAll(listJ);
                   updateLocalList(list, "J");
                } catch (Exception ignored) {

                }

                break;
            case 'K':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("K")));
                    list.remove(path);
                    listK.remove(path);
                    list.addAll(listK);
                    updateLocalList(list, "K");
                } catch (Exception ignored) {

                }

                break;
            case 'L':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("L")));
                    list.remove(path);
                    listL.remove(path);
                    list.addAll(listL);
                    updateLocalList(list, "L");
                } catch (Exception ignored) {

                }

                break;
            case 'M':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("M")));
                    list.remove(path);
                    listM.remove(path);
                    list.addAll(listM);
                    updateLocalList(list, "M");
                } catch (Exception ignored) {

                }

                break;
            case 'N':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("N")));
                    list.remove(path);
                    listN.remove(path);
                    list.addAll(listN);
                    updateLocalList(list, "N");
                } catch (Exception ignored) {

                }

                break;
            case 'O':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("O")));
                    list.remove(path);
                    listO.remove(path);
                    list.addAll(listO);
                    updateLocalList(list, "O");
                } catch (Exception ignored) {

                }

                break;
            case 'P':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("P")));
                    list.remove(path);
                    listP.remove(path);
                    list.addAll(listP);
                    updateLocalList(list, "P");
                } catch (Exception ignored) {

                }

                break;
            case 'Q':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Q")));
                    list.remove(path);
                    listQ.remove(path);
                    list.addAll(listQ);
                    updateLocalList(list, "Q");
                } catch (Exception ignored) {

                }

                break;
            case 'R':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("R")));
                    list.remove(path);
                    listR.remove(path);
                    list.addAll(listR);
                    updateLocalList(list, "R");
                } catch (Exception ignored) {

                }

                break;
            case 'S':
                try {
                   CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("S")));
                   list.remove(path);
                    listS.remove(path);
                   list.addAll(listS);
                   updateLocalList(list, "S");
                } catch (Exception ignored) {

                }

                break;
            case 'T':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("T")));
                    list.remove(path);
                    listT.remove(path);
                    list.addAll(listT);
                    updateLocalList(list, "T");
                } catch (Exception ignored) {

                }

                break;
            case 'U':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("U")));
                    list.remove(path);
                    listU.remove(path);
                    list.addAll(listU);
                    updateLocalList(list, "U");
                } catch (Exception ignored) {

                }

                break;
            case 'V':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("V")));
                    list.remove(path);
                    listV.remove(path);
                    list.addAll(listV);
                    updateLocalList(list, "V");
                } catch (Exception ignored) {

                }

                break;
            case 'W':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("W")));
                    list.remove(path);
                    listW.remove(path);
                    list.addAll(listW);
                    updateLocalList(list, "W");
                } catch (Exception ignored) {

                }

                break;
            case 'X':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("X")));
                    list.remove(path);
                    listX.remove(path);
                    list.addAll(listX);
                    updateLocalList(list, "X");
                } catch (Exception ignored) {

                }

                break;
            case 'Y':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Y")));
                    list.remove(path);
                    listY.remove(path);
                    list.addAll(listY);
                    updateLocalList(list, "Y");
                } catch (Exception ignored) {

                }

                break;
            case 'Z':
                try {
                    CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Z")));
                    list.remove(path);
                    listZ.remove(path);
                    list.addAll(listZ);
                    updateLocalList(list, "Z");
                } catch (Exception ignored) {

                }

                break;
            default:
                if (Character.isDigit(headWord)) {
                    try {
                        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Num")));
                        list.remove(path);
                        listNum.remove(path);
                        list.addAll(listNum);
                        updateLocalList(list, "Num");
                    } catch (Exception ignored) {

                    }
                } else if ('_' == headWord) {
                    try {
                        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Underline")));
                        list.remove(path);
                        listUnderline.remove(path);
                        list.addAll(listUnderline);
                        updateLocalList(list, "Underline");
                    } catch (Exception ignored) {

                    }
                } else if ('%' == headWord) {
                    try {
                        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("PercentSign")));
                        list.remove(path);
                        listPercentSign.remove(path);
                        list.addAll(listPercentSign);
                        updateLocalList(list, "PercentSign");
                    } catch (Exception ignored) {

                    }
                } else {
                    try {
                        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Unique")));
                        list.remove(path);
                        listUnique.remove(path);
                        list.addAll(listUnique);
                        updateLocalList(list, "Unique");
                    } catch (Exception ignored) {

                    }
                }
                break;
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
                    try {
                        listA.add(path);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'B':
                    try {
                        listB.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'C':
                    try {
                        listC.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'D':
                    try {
                        listD.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'E':
                    try {
                        listE.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'F':
                    try {
                        listF.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'G':
                    try {
                        listG.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'H':
                    try {
                        listH.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'I':
                    try {
                        listI.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'J':
                    try {
                        listJ.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'K':
                    try {
                        listK.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'L':
                    try {
                        listL.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'M':
                    try {
                        listM.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'N':
                    try {
                        listN.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'O':
                    try {
                        listO.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'P':
                    try {
                        listP.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'Q':
                    try {
                        listQ.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'R':
                    try {
                        listR.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'S':
                    try {
                        listS.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'T':
                    try {
                        listT.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'U':
                    try {
                        listU.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'V':
                    try {
                        listV.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'W':
                    try {
                        listW.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'X':
                    try {
                        listX.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'Y':
                    try {
                        listY.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                case 'Z':
                    try {
                        listZ.add(path);
                    } catch (Exception ignored) {

                    }

                    break;
                default:
                    if (Character.isDigit(headWord)) {
                        try {
                            listNum.add(path);
                        } catch (Exception ignored) {

                        }
                    } else if ('_' == headWord) {
                        try {
                            listUnderline.add(path);
                        } catch (Exception ignored) {

                        }
                    } else if ('%' == headWord) {
                        try {
                            listPercentSign.add(path);
                        } catch (Exception ignored) {

                        }
                    } else {
                        try {
                            listUnique.add(path);
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

    private CopyOnWriteArraySet<String> readFileList(String ch) {
        String path;
        Serialize listFile;
        switch (ch) {
            case "A":
                path = SettingsFrame.dataPath + "\\listA";
                File file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "B":

                path = SettingsFrame.dataPath + "\\listB";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "C":

                path = SettingsFrame.dataPath + "\\listC";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "D":

                path = SettingsFrame.dataPath + "\\listD";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "E":

                path = SettingsFrame.dataPath + "\\listE";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "F":

                path = SettingsFrame.dataPath + "\\listF";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "G":

                path = SettingsFrame.dataPath + "\\listG";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "H":

                path = SettingsFrame.dataPath + "\\listH";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "I":

                path = SettingsFrame.dataPath + "\\listI";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "J":

                path = SettingsFrame.dataPath + "\\listJ";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "K":

                path = SettingsFrame.dataPath + "\\listK";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "L":

                path = SettingsFrame.dataPath + "\\listL";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "M":

                path = SettingsFrame.dataPath + "\\listM";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "N":

                path = SettingsFrame.dataPath + "\\listN";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "O":

                path = SettingsFrame.dataPath + "\\listO";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "P":

                path = SettingsFrame.dataPath + "\\listP";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Q":

                path = SettingsFrame.dataPath + "\\listQ";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "R":

                path = SettingsFrame.dataPath + "\\listR";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "S":

                path = SettingsFrame.dataPath + "\\listS";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "T":

                path = SettingsFrame.dataPath + "\\listT";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "U":

                path = SettingsFrame.dataPath + "\\listU";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "V":

                path = SettingsFrame.dataPath + "\\listV";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "W":

                path = SettingsFrame.dataPath + "\\listW";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "X":

                path = SettingsFrame.dataPath + "\\listX";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Y":

                path = SettingsFrame.dataPath + "\\listY";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Z":

                path = SettingsFrame.dataPath + "\\listZ";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "PercentSign":

                path = SettingsFrame.dataPath + "\\listPercentSign";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Underline":

                path = SettingsFrame.dataPath + "\\listUnderline";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Unique":

                path = SettingsFrame.dataPath + "\\listUnique";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Num":

                path = SettingsFrame.dataPath + "\\listNum";
                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    return listFile.list;
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
        }
        return null;
    }

    public void saveAndReleaseLists() {
        for (int i = 65; i < 91; i++) {
            String ch = "" + (char) i;
            saveAndReleaseList(ch);
        }
        saveAndReleaseList("PercentSign");
        saveAndReleaseList("Underline");
        saveAndReleaseList("Unique");
        saveAndReleaseList("Num");
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
        saveAndReleaseLists();
        isUsable = true;
        isManualUpdate = false;
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
        try (BufferedReader buffr = new BufferedReader(new InputStreamReader(new FileInputStream(searchResults)))) {
            int count = 0;
            while ((each = buffr.readLine()) != null) {
                if (count > 20000) {
                    saveAndReleaseLists();
                    count = 0;
                }
                File tmp = new File(each);
                String name = tmp.getName();
                char headWord = '\0';
                try {
                    headWord = PinYinConverter.getPinYin(name).charAt(0);
                    headWord = Character.toUpperCase(headWord);
                } catch (Exception ignored) {

                }
                switch (headWord) {
                    case 'A':
                        try {
                            listA.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'B':
                        try {
                            listB.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'C':
                        try {
                            listC.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'D':
                        try {
                            listD.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'E':
                        try {
                            listE.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'F':
                        try {
                            listF.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'G':
                        try {
                            listG.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'H':
                        try {
                            listH.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'I':
                        try {
                            listI.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'J':
                        try {
                            listJ.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'K':
                        try {
                            listK.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'L':
                        try {
                            listL.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'M':
                        try {
                            listM.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'N':
                        try {
                            listN.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'O':
                        try {
                            listO.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'P':
                        try {
                            listP.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'Q':
                        try {
                            listQ.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'R':
                        try {
                            listR.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'S':
                        try {
                            listS.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'T':
                        try {
                            listT.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'U':
                        try {
                            listU.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'V':
                        try {
                            listV.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'W':
                        try {
                            listW.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'X':
                        try {
                            listX.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'Y':
                        try {
                            listY.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'Z':
                        try {
                            listZ.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    default:
                        if (Character.isDigit(headWord)) {
                            try {
                                listNum.add(each);
                                count++;
                            } catch (Exception ignored) {

                            }
                        } else if ('_' == headWord) {
                            try {
                                listUnderline.add(each);
                                count++;
                            } catch (Exception ignored) {

                            }
                        } else if ('%' == headWord) {
                            try {
                                listPercentSign.add(each);
                                count++;
                            } catch (Exception ignored) {

                            }
                        } else {
                            try {
                                listUnique.add(each);
                                count++;
                            } catch (Exception ignored) {

                            }
                        }
                        break;
                }
            }
        } catch (Exception ignored) {

        }
    }

    private void __searchFile(String path, int searchDepth, String ignorePath) {
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
        try (BufferedReader buffr = new BufferedReader(new InputStreamReader(new FileInputStream(searchResults)))) {
            int count = 0;
            while ((each = buffr.readLine()) != null) {
                if (count > 20000) {
                    saveAndReleaseLists();
                    count = 0;
                }
                File tmp = new File(each);
                String name = tmp.getName();
                char headWord = '\0';
                try {
                    headWord = PinYinConverter.getPinYin(name).charAt(0);
                    headWord = Character.toUpperCase(headWord);
                } catch (Exception ignored) {

                }
                switch (headWord) {
                    case 'A':
                        try {
                            listA.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'B':
                        try {
                            listB.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'C':
                        try {
                            listC.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'D':
                        try {
                            listD.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'E':
                        try {
                            listE.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'F':
                        try {
                            listF.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'G':
                        try {
                            listG.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'H':
                        try {
                            listH.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'I':
                        try {
                            listI.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'J':
                        try {
                            listJ.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'K':
                        try {
                            listK.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'L':
                        try {
                            listL.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'M':
                        try {
                            listM.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'N':
                        try {
                            listN.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'O':
                        try {
                            listO.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'P':
                        try {
                            listP.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'Q':
                        try {
                            listQ.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'R':
                        try {
                            listR.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'S':
                        try {
                            listS.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'T':
                        try {
                            listT.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'U':
                        try {
                            listU.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'V':
                        try {
                            listV.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'W':
                        try {
                            listW.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'X':
                        try {
                            listX.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'Y':
                        try {
                            listY.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    case 'Z':
                        try {
                            listZ.add(each);
                            count++;
                        } catch (Exception ignored) {

                        }
                        break;
                    default:
                        if (Character.isDigit(headWord)) {
                            try {
                                listNum.add(each);
                                count++;
                            } catch (Exception ignored) {

                            }
                        } else if ('_' == headWord) {
                            try {
                                listUnderline.add(each);
                                count++;
                            } catch (Exception ignored) {

                            }
                        } else if ('%' == headWord) {
                            try {
                                listPercentSign.add(each);
                                count++;
                            } catch (Exception ignored) {

                            }
                        } else {
                            try {
                                listUnique.add(each);
                                count++;
                            } catch (Exception ignored) {

                            }
                        }
                        break;
                }
            }
        } catch (Exception ignored) {

        }
    }


    public CopyOnWriteArraySet<String> getListA() {
        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("A")));
        list.addAll(listA);
        return list;
    }

    public CopyOnWriteArraySet<String> getListB() {

        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("B")));
        list.addAll(listB);
        return list;
    }

    public CopyOnWriteArraySet<String> getListC() {

        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("C")));
        list.addAll(listC);
        return list;

    }

    public CopyOnWriteArraySet<String> getListD() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("D")));
        list.addAll(listD);
        return list;

    }

    public CopyOnWriteArraySet<String> getListE() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("E")));
        list.addAll(listE);
        return list;

    }

    public CopyOnWriteArraySet<String> getListF() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("F")));
        list.addAll(listF);
        return list;

    }

    public CopyOnWriteArraySet<String> getListG() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("G")));
        list.addAll(listG);
        return list;

    }

    public CopyOnWriteArraySet<String> getListH() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("H")));
        list.addAll(listH);
        return list;

    }

    public CopyOnWriteArraySet<String> getListI() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("I")));
        list.addAll(listI);
        return list;

    }

    public CopyOnWriteArraySet<String> getListJ() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("J")));
        list.addAll(listJ);
        return list;

    }

    public CopyOnWriteArraySet<String> getListK() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("K")));
        list.addAll(listK);
        return list;

    }

    public CopyOnWriteArraySet<String> getListL() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("L")));
        list.addAll(listL);
        return list;

    }

    public CopyOnWriteArraySet<String> getListM() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("M")));
        list.addAll(listM);
        return list;

    }

    public CopyOnWriteArraySet<String> getListN() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("N")));
        list.addAll(listN);
        return list;

    }

    public CopyOnWriteArraySet<String> getListO() {

        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("O")));
        list.addAll(listO);
        return list;

    }

    public CopyOnWriteArraySet<String> getListP() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("P")));
        list.addAll(listP);
        return list;

    }

    public CopyOnWriteArraySet<String> getListQ() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Q")));
        list.addAll(listQ);
        return list;

    }

    public CopyOnWriteArraySet<String> getListR() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("R")));
        list.addAll(listR);
        return list;

    }

    public CopyOnWriteArraySet<String> getListNum() {

        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Num")));
        list.addAll(listT);
        return list;

    }

    public CopyOnWriteArraySet<String> getListPercentSign() {

        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("PercentSign")));
        list.addAll(listT);
        return list;

    }

    public CopyOnWriteArraySet<String> getListS() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("S")));
        list.addAll(listS);
        return list;

    }

    public CopyOnWriteArraySet<String> getListT() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("T")));
        list.addAll(listT);
        return list;

    }

    public CopyOnWriteArraySet<String> getListUnique() {

        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Unique")));
        list.addAll(listT);
        return list;
    }

    public CopyOnWriteArraySet<String> getListU() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("U")));
        list.addAll(listU);
        return list;

    }

    public CopyOnWriteArraySet<String> getListUnderline() {

        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Underline")));
        list.addAll(listT);
        return list;

    }

    public CopyOnWriteArraySet<String> getListV() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("V")));
        list.addAll(listV);
        return list;

    }

    public CopyOnWriteArraySet<String> getListW() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("W")));
        list.addAll(listW);
        return list;

    }

    public CopyOnWriteArraySet<String> getListY() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("X")));
        list.addAll(listX);
        return list;

    }

    public CopyOnWriteArraySet<String> getListX() {


        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Y")));
        list.addAll(listY);
        return list;

    }

    public CopyOnWriteArraySet<String> getListZ() {
        CopyOnWriteArraySet<String> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Z")));
        list.addAll(listZ);
        return list;

    }

    public void updateLists(String ignorePath, int searchDepth) {
        listA.clear();
        listB.clear();
        listC.clear();
        listD.clear();
        listE.clear();
        listF.clear();
        listG.clear();
        listH.clear();
        listI.clear();
        listJ.clear();
        listK.clear();
        listL.clear();
        listM.clear();
        listN.clear();
        listO.clear();
        listP.clear();
        listQ.clear();
        listR.clear();
        listS.clear();
        listT.clear();
        listU.clear();
        listV.clear();
        listW.clear();
        listX.clear();
        listY.clear();
        listZ.clear();
        listNum.clear();
        listPercentSign.clear();
        listUnique.clear();
        listUnderline.clear();
        MainClass.deleteDir(SettingsFrame.dataPath);
        searchFile(ignorePath, searchDepth);
    }

    static class Serialize implements Serializable {
        private static final long serialVersionUID = 1L;
        private CopyOnWriteArraySet<String> list;

        public void setList(CopyOnWriteArraySet<String> listOut) {
            this.list = listOut;
        }
    }
}