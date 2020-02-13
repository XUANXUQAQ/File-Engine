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
    private static CopyOnWriteArraySet<byte[]> listA = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listB = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listC = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listD = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listE = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listF = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listG = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listH = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listI = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listJ = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listK = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listL = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listM = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listN = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listO = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listP = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listQ = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listR = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listS = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listT = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listU = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listV = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listW = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listX = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listY = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listZ = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listNum = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listPercentSign = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listUnique = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArraySet<byte[]> listUnderline = new CopyOnWriteArraySet<>();
    private static CopyOnWriteArrayList<byte[]> listToLoad = new CopyOnWriteArrayList<>();
    private static boolean isUsable = false;
    private static boolean isManualUpdate = false;
    private static List<byte[]> RecycleBin = Collections.synchronizedList(new LinkedList<>());

    public static String byteArrayToStr(byte[] byteArray) {
        if (byteArray == null) {
            return null;
        }
        return new String(byteArray).intern();
    }

    public static byte[] strToByteArray(String str) {
        if (str == null) {
            return null;
        }
        return str.getBytes();
    }

    private static void clearInfoForFile(String fileName) {
        File file = new File(fileName);
        try {
            if (!file.exists()) {
                file.createNewFile();
            }
            FileWriter fileWriter = new FileWriter(file);
            fileWriter.write("");
            fileWriter.flush();
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int getRecycleBinSize() {
        return RecycleBin.size();
    }

    public int getLoadListSize() {
        return listToLoad.size();
    }

    public void addToRecycleBin(String path) {
        RecycleBin.add(strToByteArray(path));
    }

    public void mergeAndClearRecycleBin() {
        if (!isManualUpdate) {
            isUsable = false;
            try {
                for (byte[] i : RecycleBin) {
                    String path = byteArrayToStr(i);
                    deletePathInList(path);
                    RecycleBin.clear();
                }
            } catch (ConcurrentModificationException ignored) {

            } finally {
                isUsable = true;
            }
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
                if (!isManualUpdate) {
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
        listToLoad.add(strToByteArray(path));
    }

    public void mergeFileToList() {
        if (!isManualUpdate) {
            isUsable = false;
            for (byte[] i : listToLoad) {
                String each = byteArrayToStr(i);
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
            firstWord = file.getName().charAt(0);
        } catch (Exception ignored) {

        }
        char headWord = Character.toUpperCase(firstWord);
        switch (headWord) {
            case 'A':
                try {
                    listA.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'B':
                try {
                    listB.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'C':
                try {
                    listC.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'D':
                try {
                    listD.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'E':
                try {
                    listE.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'F':
                try {
                    listF.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'G':
                try {
                    listG.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'H':
                try {
                    listH.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'I':
                try {
                    listI.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'J':
                try {
                    listJ.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'K':
                try {
                    listK.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'L':
                try {
                    listL.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'M':
                try {
                    listM.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'N':
                try {
                    listN.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'O':
                try {
                    listO.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'P':
                try {
                    listP.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'Q':
                try {
                    listQ.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'R':
                try {
                    listR.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'S':
                try {
                    listS.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'T':
                try {
                    listT.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'U':
                try {
                    listU.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'V':
                try {
                    listV.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'W':
                try {
                    listW.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'X':
                try {
                    listX.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'Y':
                try {
                    listY.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            case 'Z':
                try {
                    listZ.remove(strToByteArray(path));
                } catch (Exception ignored) {

                }

                break;
            default:
                if (Character.isDigit(headWord)) {
                    try {
                        listNum.remove(strToByteArray(path));
                    } catch (Exception ignored) {

                    }
                } else if ('_' == headWord) {
                    try {
                        listUnderline.remove(strToByteArray(path));
                    } catch (Exception ignored) {

                    }
                } else if ('%' == headWord) {
                    try {
                        listPercentSign.remove(strToByteArray(path));
                    } catch (Exception ignored) {

                    }
                } else {
                    try {
                        listUnique.remove(strToByteArray(path));
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
                        listA.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }
                    break;
                case 'B':
                    try {
                        listB.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'C':
                    try {
                        listC.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'D':
                    try {
                        listD.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'E':
                    try {
                        listE.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'F':
                    try {
                        listF.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'G':
                    try {
                        listG.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'H':
                    try {
                        listH.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'I':
                    try {
                        listI.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'J':
                    try {
                        listJ.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'K':
                    try {
                        listK.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'L':
                    try {
                        listL.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'M':
                    try {
                        listM.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'N':
                    try {
                        listN.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'O':
                    try {
                        listO.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'P':
                    try {
                        listP.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'Q':
                    try {
                        listQ.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'R':
                    try {
                        listR.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'S':
                    try {
                        listS.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'T':
                    try {
                        listT.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'U':
                    try {
                        listU.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'V':
                    try {
                        listV.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'W':
                    try {
                        listW.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'X':
                    try {
                        listX.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'Y':
                    try {
                        listY.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                case 'Z':
                    try {
                        listZ.add(strToByteArray(path));
                    } catch (Exception ignored) {

                    }

                    break;
                default:
                    if (Character.isDigit(headWord)) {
                        try {
                            listNum.add(strToByteArray(path));
                        } catch (Exception ignored) {

                        }
                    } else if ('_' == headWord) {
                        try {
                            listUnderline.add(strToByteArray(path));
                        } catch (Exception ignored) {

                        }
                    } else if ('%' == headWord) {
                        try {
                            listPercentSign.add(strToByteArray(path));
                        } catch (Exception ignored) {

                        }
                    } else {
                        try {
                            listUnique.add(strToByteArray(path));
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

    private CopyOnWriteArraySet<byte[]> readFileList(String ch) {
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
        clearInfoForFile(SettingsFrame.tmp + "\\searchResults.txt");
        String command = "cmd /c fileSearcher.exe \"" + path + "\"" + " \"6\" " + "\"" + ignorePath + "\" " + "\"" + searchResults.getAbsolutePath() + "\" " + "\"" + "1" + "\"";
        String each;
        BufferedReader buffr;
        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            buffr = new BufferedReader(new InputStreamReader(new FileInputStream(searchResults)));
            while (p.isAlive()) {
                try {
                    while ((each = buffr.readLine()) != null) {
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
                                    listA.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'B':
                                try {
                                    listB.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'C':
                                try {
                                    listC.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'D':
                                try {
                                    listD.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'E':
                                try {
                                    listE.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'F':
                                try {
                                    listF.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'G':
                                try {
                                    listG.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'H':
                                try {
                                    listH.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'I':
                                try {
                                    listI.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'J':
                                try {
                                    listJ.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'K':
                                try {
                                    listK.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'L':
                                try {
                                    listL.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'M':
                                try {
                                    listM.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'N':
                                try {
                                    listN.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'O':
                                try {
                                    listO.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'P':
                                try {
                                    listP.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'Q':
                                try {
                                    listQ.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'R':
                                try {
                                    listR.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'S':
                                try {
                                    listS.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'T':
                                try {
                                    listT.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'U':
                                try {
                                    listU.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'V':
                                try {
                                    listV.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'W':
                                try {
                                    listW.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'X':
                                try {
                                    listX.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'Y':
                                try {
                                    listY.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'Z':
                                try {
                                    listZ.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            default:
                                if (Character.isDigit(headWord)) {
                                    try {
                                        listNum.add(strToByteArray(each));
                                    } catch (Exception ignored) {

                                    }
                                } else if ('_' == headWord) {
                                    try {
                                        listUnderline.add(strToByteArray(each));
                                    } catch (Exception ignored) {

                                    }
                                } else if ('%' == headWord) {
                                    try {
                                        listPercentSign.add(strToByteArray(each));
                                    } catch (Exception ignored) {

                                    }
                                } else {
                                    try {
                                        listUnique.add(strToByteArray(each));
                                    } catch (Exception ignored) {

                                    }
                                }
                                break;
                        }
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void __searchFile(String path, int searchDepth, String ignorePath) {
        File searchResults = new File(SettingsFrame.tmp + "\\searchResults.txt");
        clearInfoForFile(SettingsFrame.tmp + "\\searchResults.txt");
        String command = "cmd /c fileSearcher.exe \"" + path + "\"" + " \"" + searchDepth + "\" " + "\"" + ignorePath + "\" " + "\"" + searchResults.getAbsolutePath() + "\" " + "\"" + "0" + "\"";
        String each;
        BufferedReader buffr;
        Process p;
        try {
            p = Runtime.getRuntime().exec(command);
            buffr = new BufferedReader(new InputStreamReader(new FileInputStream(searchResults)));
            try {
                while (p.isAlive()) {
                    while ((each = buffr.readLine()) != null) {
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
                                    listA.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'B':
                                try {
                                    listB.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'C':
                                try {
                                    listC.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'D':
                                try {
                                    listD.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'E':
                                try {
                                    listE.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'F':
                                try {
                                    listF.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'G':
                                try {
                                    listG.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'H':
                                try {
                                    listH.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'I':
                                try {
                                    listI.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'J':
                                try {
                                    listJ.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'K':
                                try {
                                    listK.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'L':
                                try {
                                    listL.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'M':
                                try {
                                    listM.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'N':
                                try {
                                    listN.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'O':
                                try {
                                    listO.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'P':
                                try {
                                    listP.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'Q':
                                try {
                                    listQ.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'R':
                                try {
                                    listR.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'S':
                                try {
                                    listS.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'T':
                                try {
                                    listT.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'U':
                                try {
                                    listU.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'V':
                                try {
                                    listV.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'W':
                                try {
                                    listW.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'X':
                                try {
                                    listX.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'Y':
                                try {
                                    listY.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            case 'Z':
                                try {
                                    listZ.add(strToByteArray(each));
                                } catch (Exception ignored) {

                                }
                                break;
                            default:
                                if (Character.isDigit(headWord)) {
                                    try {
                                        listNum.add(strToByteArray(each));
                                    } catch (Exception ignored) {

                                    }
                                } else if ('_' == headWord) {
                                    try {
                                        listUnderline.add(strToByteArray(each));
                                    } catch (Exception ignored) {

                                    }
                                } else if ('%' == headWord) {
                                    try {
                                        listPercentSign.add(strToByteArray(each));
                                    } catch (Exception ignored) {

                                    }
                                } else {
                                    try {
                                        listUnique.add(strToByteArray(each));
                                    } catch (Exception ignored) {

                                    }
                                }
                                break;
                        }
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public CopyOnWriteArraySet<byte[]> getListA() {
        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("A")));
        list.addAll(listA);
        return list;
    }

    public CopyOnWriteArraySet<byte[]> getListB() {

        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("B")));
        list.addAll(listB);
        return list;
    }

    public CopyOnWriteArraySet<byte[]> getListC() {

        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("C")));
        list.addAll(listC);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListD() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("D")));
        list.addAll(listD);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListE() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("E")));
        list.addAll(listE);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListF() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("F")));
        list.addAll(listF);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListG() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("G")));
        list.addAll(listG);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListH() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("H")));
        list.addAll(listH);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListI() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("I")));
        list.addAll(listI);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListJ() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("J")));
        list.addAll(listJ);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListK() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("K")));
        list.addAll(listK);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListL() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("L")));
        list.addAll(listL);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListM() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("M")));
        list.addAll(listM);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListN() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("N")));
        list.addAll(listN);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListO() {

        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("O")));
        list.addAll(listO);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListP() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("P")));
        list.addAll(listP);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListQ() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Q")));
        list.addAll(listQ);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListR() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("R")));
        list.addAll(listR);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListNum() {

        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Num")));
        list.addAll(listT);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListPercentSign() {

        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("PercentSign")));
        list.addAll(listT);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListS() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("S")));
        list.addAll(listS);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListT() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("T")));
        list.addAll(listT);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListUnique() {

        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Unique")));
        list.addAll(listT);
        return list;
    }

    public CopyOnWriteArraySet<byte[]> getListU() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("U")));
        list.addAll(listU);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListUnderline() {

        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Underline")));
        list.addAll(listT);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListV() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("V")));
        list.addAll(listV);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListW() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("W")));
        list.addAll(listW);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListY() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("X")));
        list.addAll(listX);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListX() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Y")));
        list.addAll(listY);
        return list;

    }

    public CopyOnWriteArraySet<byte[]> getListZ() {


        CopyOnWriteArraySet<byte[]> list = new CopyOnWriteArraySet<>(Objects.requireNonNull(readFileList("Z")));
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
        searchFile(ignorePath, searchDepth);
    }

    static class Serialize implements Serializable {
        private static final long serialVersionUID = 1L;
        private CopyOnWriteArraySet<byte[]> list;

        public void setList(CopyOnWriteArraySet<byte[]> listOut) {
            this.list = listOut;
        }
    }
}