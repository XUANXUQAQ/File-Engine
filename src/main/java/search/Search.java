package search;

import frame.SettingsFrame;
import main.MainClass;

import javax.swing.filechooser.FileSystemView;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.LinkedList;


public class Search {
    private static LinkedHashSet<String> listA = new LinkedHashSet<>();
    private static LinkedHashSet<String> listB = new LinkedHashSet<>();
    private static LinkedHashSet<String> listC = new LinkedHashSet<>();
    private static LinkedHashSet<String> listD = new LinkedHashSet<>();
    private static LinkedHashSet<String> listE = new LinkedHashSet<>();
    private static LinkedHashSet<String> listF = new LinkedHashSet<>();
    private static LinkedHashSet<String> listG = new LinkedHashSet<>();
    private static LinkedHashSet<String> listH = new LinkedHashSet<>();
    private static LinkedHashSet<String> listI = new LinkedHashSet<>();
    private static LinkedHashSet<String> listJ = new LinkedHashSet<>();
    private static LinkedHashSet<String> listK = new LinkedHashSet<>();
    private static LinkedHashSet<String> listL = new LinkedHashSet<>();
    private static LinkedHashSet<String> listM = new LinkedHashSet<>();
    private static LinkedHashSet<String> listN = new LinkedHashSet<>();
    private static LinkedHashSet<String> listO = new LinkedHashSet<>();
    private static LinkedHashSet<String> listP = new LinkedHashSet<>();
    private static LinkedHashSet<String> listQ = new LinkedHashSet<>();
    private static LinkedHashSet<String> listR = new LinkedHashSet<>();
    private static LinkedHashSet<String> listS = new LinkedHashSet<>();
    private static LinkedHashSet<String> listT = new LinkedHashSet<>();
    private static LinkedHashSet<String> listU = new LinkedHashSet<>();
    private static LinkedHashSet<String> listV = new LinkedHashSet<>();
    private static LinkedHashSet<String> listW = new LinkedHashSet<>();
    private static LinkedHashSet<String> listX = new LinkedHashSet<>();
    private static LinkedHashSet<String> listY = new LinkedHashSet<>();
    private static LinkedHashSet<String> listZ = new LinkedHashSet<>();
    private static LinkedHashSet<String> listNum = new LinkedHashSet<>();
    private static LinkedHashSet<String> listPercentSign = new LinkedHashSet<>();
    private static LinkedHashSet<String> listUnique = new LinkedHashSet<>();
    private static LinkedHashSet<String> listUnderline = new LinkedHashSet<>();
    private static LinkedList<String> listToAdd = new LinkedList<>();
    private static LinkedList<String> listRemain = new LinkedList<>();
    private static boolean isUsable = false;
    private static boolean isFocusLost = true;
    private static boolean isManualUpdate = false;
    private static LinkedList<String> RecycleBin = new LinkedList<>();
    private String startMenu = getStartMenu();
    private final String desktop = FileSystemView.getFileSystemView().getHomeDirectory().getAbsolutePath();
    private int num;



    public void addToRecycleBin(String path) {
        RecycleBin.add(path);
    }

    public void clearRecycleBin() {
        if (!isManualUpdate) {
            isUsable = false;
            for (String path : RecycleBin) {
                deletePathInList(path);
            }
            isUsable = true;
        }
    }

    public void addFileToLoadBin(String path){
        listToAdd.add(path);
    }

    public void mergeFileToList(){
        while (!listToAdd.isEmpty()){
            String each = listToAdd.pop();
            File add = new File(each);
            if (add.exists()) {
                addFileToList(each);
            }
        }
    }

    private void deletePathInList(String path) {
        File file = new File(path);
        char firstWord = file.getName().charAt(0);
        char headWord = Character.toUpperCase(firstWord);
        switch (headWord) {
            case 'A':
                try {
                    listA.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'B':
                try {
                    listB.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'C':
                try {
                    listC.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'D':
                try {
                    listD.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'E':
                try {
                    listE.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'F':
                try {
                    listF.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'G':
                try {
                    listG.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'H':
                try {
                    listH.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'I':
                try {
                    listI.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'J':
                try {
                    listJ.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'K':
                try {
                    listK.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'L':
                try {
                    listL.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'M':
                try {
                    listM.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'N':
                try {
                    listN.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'O':
                try {
                    listO.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'P':
                try {
                    listP.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'Q':
                try {
                    listQ.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'R':
                try {
                    listR.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'S':
                try {
                    listS.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'T':
                try {
                    listT.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'U':
                try {
                    listU.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'V':
                try {
                    listV.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'W':
                try {
                    listW.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'X':
                try {
                    listX.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'Y':
                try {
                    listY.remove(path);
                } catch (Exception ignored) {

                }

                break;
            case 'Z':
                try {
                    listZ.remove(path);
                } catch (Exception ignored) {

                }

                break;
            default:
                if (Character.isDigit(headWord)) {
                    try {
                        listNum.remove(path);
                    } catch (Exception ignored) {

                    }
                } else if ('_' == headWord) {
                    try {
                        listUnderline.remove(path);
                    } catch (Exception ignored) {

                    }
                } else if ('%' == headWord) {
                    try {
                        listPercentSign.remove(path);
                    } catch (Exception ignored) {

                    }
                } else {
                    try {
                        listUnique.remove(path);
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
        }else{
            isUsable = false;
        }
    }

    public boolean isIsFocusLost() {
        return isFocusLost;
    }

    public void setFocusLostStatus(boolean b) {
        isFocusLost = b;
    }

    private void addFileToList(String path) {
        File file = new File(path);
        char firstWord = file.getName().charAt(0);
        if (firstWord != '$' && firstWord != '.') {
            char headWord = Character.toUpperCase(firstWord);
            switch (headWord) {
                case 'A':
                    listA.add(path);
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

    public void loadAllLists() {
        for (int i = 65; i < 91; i++) {
            String ch = "" + (char) i;
            readFileList(ch);
        }
        readFileList("PercentSign");
        readFileList("Underline");
        readFileList("Unique");
        readFileList("Num");
    }


    public void saveLists(){
        for (int i = 65; i < 91; i++) {
            String ch = "" + (char) i;
            saveList(ch);
        }
        saveList("PercentSign");
        saveList("Underline");
        saveList("Unique");
        saveList("Num");
    }

    private void saveList(String ch){
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
                listFile.setList(listA);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "B":
                path = SettingsFrame.dataPath + "\\listB";
                listFile = new Serialize();
                listFile.setList(listB);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "C":
                path = SettingsFrame.dataPath + "\\listC";
                listFile = new Serialize();
                listFile.setList(listC);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "D":

                path = SettingsFrame.dataPath + "\\listD";
                listFile = new Serialize();
                listFile.setList(listD);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "E":

                path = SettingsFrame.dataPath + "\\listE";
                listFile = new Serialize();
                listFile.setList(listE);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "F":

                path = SettingsFrame.dataPath + "\\listF";
                listFile = new Serialize();
                listFile.setList(listF);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "G":

                path = SettingsFrame.dataPath + "\\listG";
                listFile = new Serialize();
                listFile.setList(listG);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "H":

                path = SettingsFrame.dataPath + "\\listH";
                listFile = new Serialize();
                listFile.setList(listH);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "I":

                path = SettingsFrame.dataPath + "\\listI";
                listFile = new Serialize();
                listFile.setList(listI);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "J":

                path = SettingsFrame.dataPath + "\\listJ";
                listFile = new Serialize();
                listFile.setList(listJ);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "K":

                path = SettingsFrame.dataPath + "\\listK";
                listFile = new Serialize();
                listFile.setList(listK);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "L":

                path = SettingsFrame.dataPath + "\\listL";
                listFile = new Serialize();
                listFile.setList(listL);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "M":

                path = SettingsFrame.dataPath + "\\listM";
                listFile = new Serialize();
                listFile.setList(listM);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "N":

                path = SettingsFrame.dataPath + "\\listN";
                listFile = new Serialize();
                listFile.setList(listN);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "O":

                path = SettingsFrame.dataPath + "\\listO";
                listFile = new Serialize();
                listFile.setList(listO);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "P":

                path = SettingsFrame.dataPath + "\\listP";
                listFile = new Serialize();
                listFile.setList(listP);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "Q":

                path = SettingsFrame.dataPath + "\\listQ";
                listFile = new Serialize();
                listFile.setList(listQ);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "R":

                path = SettingsFrame.dataPath + "\\listR";
                listFile = new Serialize();
                listFile.setList(listR);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "S":

                path = SettingsFrame.dataPath + "\\listS";
                listFile = new Serialize();
                listFile.setList(listS);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "T":

                path = SettingsFrame.dataPath + "\\listT";
                listFile = new Serialize();
                listFile.setList(listT);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "U":

                path = SettingsFrame.dataPath + "\\listU";
                listFile = new Serialize();
                listFile.setList(listU);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "V":

                path = SettingsFrame.dataPath + "\\listV";
                listFile = new Serialize();
                listFile.setList(listV);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "W":

                path = SettingsFrame.dataPath + "\\listW";
                listFile = new Serialize();
                listFile.setList(listW);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "X":

                path = SettingsFrame.dataPath + "\\listX";
                listFile = new Serialize();
                listFile.setList(listX);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "Y":

                path = SettingsFrame.dataPath + "\\listY";
                listFile = new Serialize();
                listFile.setList(listY);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "Z":

                path = SettingsFrame.dataPath + "\\listZ";
                listFile = new Serialize();
                listFile.setList(listZ);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "PercentSign":
                path = SettingsFrame.dataPath + "\\listPercentSign";
                listFile = new Serialize();
                listFile.setList(listPercentSign);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "Underline":

                path = SettingsFrame.dataPath + "\\listUnderline";
                listFile = new Serialize();
                listFile.setList(listUnderline);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "Unique":

                path = SettingsFrame.dataPath + "\\listUnique";
                listFile = new Serialize();
                listFile.setList(listUnique);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
            case "Num":
                path = SettingsFrame.dataPath + "\\listNum";
                listFile = new Serialize();
                listFile.setList(listNum);
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(path).getAbsolutePath()))) {
                    oos.writeObject(listFile);

                } catch (IOException ignored) {

                }
                break;
        }
    }

    private void readFileList(String ch) {
        String path;
        Serialize listFile;
        switch (ch) {
            case "A":
                path = SettingsFrame.dataPath + "\\listA";
                File file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listA.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "B":

                path = SettingsFrame.dataPath + "\\listB";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listB.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "C":

                path = SettingsFrame.dataPath + "\\listC";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listC.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "D":

                path = SettingsFrame.dataPath + "\\listD";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listD.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "E":

                path = SettingsFrame.dataPath + "\\listE";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listE.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "F":

                path = SettingsFrame.dataPath + "\\listF";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listF.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "G":

                path = SettingsFrame.dataPath + "\\listG";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listG.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "H":

                path = SettingsFrame.dataPath + "\\listH";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listH.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "I":

                path = SettingsFrame.dataPath + "\\listI";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listI.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "J":

                path = SettingsFrame.dataPath + "\\listJ";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listJ.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "K":

                path = SettingsFrame.dataPath + "\\listK";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listK.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "L":

                path = SettingsFrame.dataPath + "\\listL";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listL.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "M":

                path = SettingsFrame.dataPath + "\\listM";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listM.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "N":

                path = SettingsFrame.dataPath + "\\listN";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listN.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "O":

                path = SettingsFrame.dataPath + "\\listO";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listO.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "P":

                path = SettingsFrame.dataPath + "\\listP";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listP.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Q":

                path = SettingsFrame.dataPath + "\\listQ";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listQ.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "R":

                path = SettingsFrame.dataPath + "\\listR";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listR.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "S":

                path = SettingsFrame.dataPath + "\\listS";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listS.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "T":

                path = SettingsFrame.dataPath + "\\listT";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listT.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "U":

                path = SettingsFrame.dataPath + "\\listU";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listU.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "V":

                path = SettingsFrame.dataPath + "\\listV";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listV.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "W":

                path = SettingsFrame.dataPath + "\\listW";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listW.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "X":

                path = SettingsFrame.dataPath + "\\listX";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listX.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Y":

                path = SettingsFrame.dataPath + "\\listY";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listY.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Z":

                path = SettingsFrame.dataPath + "\\listZ";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listZ.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "PercentSign":

                path = SettingsFrame.dataPath + "\\listPercentSign";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listPercentSign.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "UnderLine":

                path = SettingsFrame.dataPath + "\\listUnderline";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listUnderline.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Unique":

                path = SettingsFrame.dataPath + "\\listUnique";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listUnique.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
            case "Num":

                path = SettingsFrame.dataPath + "\\listNum";

                file = new File(path);
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                    listFile = (Serialize) ois.readObject();
                    listNum.addAll(listFile.list);
                } catch (IOException | ClassNotFoundException ignored) {

                }
                break;
        }
    }


    public void searchFile(String ignorePath, int searchDepth) {
        num = 0;
        File[] roots = File.listRoots();
        for (File root : roots) {
            listRemain.add(root.getAbsolutePath());
        }


        while (!listRemain.isEmpty()) {
            String tmp = listRemain.pop();
            __searchFile(ignorePath, new File(tmp), searchDepth);
        }
        System.out.println("搜索完成，总数据数量：");
        System.out.println(num);
        MainClass.showMessage("提示","索引已经更新");
        isUsable = true;
        isManualUpdate = false;
    }


    private void __searchFile(String ignorePath, File path, int searchDepth) {
        ignorePath = ignorePath.toUpperCase();
        boolean exist = path.exists();
        if (exist && !isIgnore(path.getAbsolutePath().toUpperCase(), ignorePath)) {
            File[] files = path.listFiles();
            if (null == files || files.length == 0) {
            } else if (searchDepth >= count(path.getAbsolutePath(), "\\") || (path.getAbsolutePath().equals(startMenu)) || path.getAbsolutePath().equals(desktop)) {
                for (File file2 : files) {
                    String fileName = file2.getName();

                    char firstWord = fileName.charAt(0);
                    if (fileName.length() >= 2) {
                        char secondWord = fileName.charAt(1);

                        if (firstWord != '$' && (firstWord != '.' && secondWord != '.')) {
                            char headWord = Character.toUpperCase(firstWord);
                            switch (headWord) {
                                case 'A':
                                    try {
                                        listA.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'B':
                                    try {
                                        listB.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'C':
                                    try {
                                        listC.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'D':
                                    try {
                                        listD.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'E':
                                    try {
                                        listE.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'F':
                                    try {
                                        listF.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'G':
                                    try {
                                        listG.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'H':
                                    try {
                                        listH.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'I':
                                    try {
                                        listI.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'J':
                                    try {
                                        listJ.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'K':
                                    try {
                                        listK.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'L':
                                    try {
                                        listL.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'M':
                                    try {
                                        listM.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'N':
                                    try {
                                        listN.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'O':
                                    try {
                                        listO.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'P':
                                    try {
                                        listP.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'Q':
                                    try {
                                        listQ.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'R':
                                    try {
                                        listR.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'S':
                                    try {
                                        listS.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'T':
                                    try {
                                        listT.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'U':
                                    try {
                                        listU.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'V':
                                    try {
                                        listV.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'W':
                                    try {
                                        listW.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'X':
                                    try {
                                        listX.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'Y':
                                    try {
                                        listY.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'Z':
                                    try {
                                        listZ.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                default:
                                    if (Character.isDigit(headWord)) {
                                        try {
                                            listNum.add(file2.getAbsolutePath());
                                        } catch (Exception ignored) {

                                        }
                                    } else if ('_' == headWord) {
                                        try {
                                            listUnderline.add(file2.getAbsolutePath());
                                        } catch (Exception ignored) {

                                        }
                                    } else if ('%' == headWord) {
                                        try {
                                            listPercentSign.add(file2.getAbsolutePath());
                                        } catch (Exception ignored) {

                                        }
                                    } else {
                                        try {
                                            listUnique.add(file2.getAbsolutePath());
                                        } catch (Exception ignored) {

                                        }
                                    }
                                    num += 1;
                                    break;
                            }
                            if (file2.isDirectory()) {
                                listRemain.add(file2.getAbsolutePath());
                            }
                        }


                    } else {
                        if (firstWord != '$') {
                            char headWord = Character.toUpperCase(firstWord);
                            switch (headWord) {
                                case 'A':
                                    try {
                                        listA.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'B':
                                    try {
                                        listB.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'C':
                                    try {
                                        listC.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'D':
                                    try {
                                        listD.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'E':
                                    try {
                                        listE.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'F':
                                    try {
                                        listF.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'G':
                                    try {
                                        listG.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'H':
                                    try {
                                        listH.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'I':
                                    try {
                                        listI.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'J':
                                    try {
                                        listJ.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'K':
                                    try {
                                        listK.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'L':
                                    try {
                                        listL.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'M':
                                    try {
                                        listM.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'N':
                                    try {
                                        listN.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'O':
                                    try {
                                        listO.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'P':
                                    try {
                                        listP.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'Q':
                                    try {
                                        listQ.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'R':
                                    try {
                                        listR.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'S':
                                    try {
                                        listS.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'T':
                                    try {
                                        listT.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'U':
                                    try {
                                        listU.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'V':
                                    try {
                                        listV.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'W':
                                    try {
                                        listW.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'X':
                                    try {
                                        listX.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'Y':
                                    try {
                                        listY.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                case 'Z':
                                    try {
                                        listZ.add(file2.getAbsolutePath());
                                    } catch (Exception ignored) {

                                    }
                                    num += 1;
                                    break;
                                default:
                                    if (Character.isDigit(headWord)) {
                                        try {
                                            listNum.add(file2.getAbsolutePath());
                                        } catch (Exception ignored) {

                                        }
                                    } else if ('_' == headWord) {
                                        try {
                                            listUnderline.add(file2.getAbsolutePath());
                                        } catch (Exception ignored) {

                                        }
                                    } else if ('%' == headWord) {
                                        try {
                                            listPercentSign.add(file2.getAbsolutePath());
                                        } catch (Exception ignored) {

                                        }
                                    } else {
                                        try {
                                            listUnique.add(file2.getAbsolutePath());
                                        } catch (Exception ignored) {

                                        }
                                    }
                                    num += 1;
                                    break;
                            }
                            if (file2.isDirectory()) {
                                try {
                                    listRemain.add(file2.getAbsolutePath());
                                } catch (Exception ignored) {

                                }
                            }
                        }
                    }
                }
            }
        }

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

    /**
     * @param srcText  源字符串
     * @param findText 需要计数的字符串
     * @return count
     */
    public int count(String srcText, String findText) {
        int count = 0;
        int start = 0;
        while (srcText.indexOf(findText, start) >= 0 && start < srcText.length()) {
            count++;
            start = srcText.indexOf(findText, start) + findText.length();
        }
        return count;
    }

    /**
     * @param target txt
     * txt是否属于target的子字符串
     * @return true false
     */
    private boolean isIgnore(String txt, String target) {
        int each;
        txt = txt.toLowerCase();
        target = target.toLowerCase();
        ArrayList<String> list = new ArrayList<>();
        while ((each = target.indexOf(",")) != -1){
            list.add(target.substring(0, each));
            target = target.substring(each + 1);
        }
        for (String i : list) {
            if (txt.contains(i)) {
                return true;
            }
        }
        return false;
    }

    public LinkedHashSet<String> getListA() {

        return listA;
    }

    public LinkedHashSet<String> getListB() {

        return listB;
    }

    public LinkedHashSet<String> getListC() {

        return listC;

    }

    public LinkedHashSet<String> getListD() {


        return listD;

    }

    public LinkedHashSet<String> getListE() {


        return listE;

    }

    public LinkedHashSet<String> getListF() {


        return listF;

    }

    public LinkedHashSet<String> getListG() {


        return listG;

    }

    public LinkedHashSet<String> getListH() {


        return listH;

    }

    public LinkedHashSet<String> getListI() {


        return listI;

    }

    public LinkedHashSet<String> getListJ() {


        return listJ;

    }

    public LinkedHashSet<String> getListK() {


        return listK;

    }

    public LinkedHashSet<String> getListL() {


        return listL;

    }

    public LinkedHashSet<String> getListM() {


        return listM;

    }

    public LinkedHashSet<String> getListN() {


        return listN;

    }

    public LinkedHashSet<String> getListO() {

        return listO;

    }

    public LinkedHashSet<String> getListP() {


        return listP;

    }

    public LinkedHashSet<String> getListQ() {


        return listQ;

    }

    public LinkedHashSet<String> getListR() {


        return listR;

    }

    public LinkedHashSet<String> getListNum() {


        return listNum;

    }

    public LinkedHashSet<String> getListPercentSign() {


        return listPercentSign;

    }

    public LinkedHashSet<String> getListS() {


        return listS;

    }

    public LinkedHashSet<String> getListT() {


        return listT;

    }

    public LinkedHashSet<String> getListUnique() {

        return listUnique;
    }

    public LinkedHashSet<String> getListU() {


        return listU;

    }

    public LinkedHashSet<String> getListUnderline() {

        return listUnderline;

    }

    public LinkedHashSet<String> getListV() {


        return listV;

    }

    public LinkedHashSet<String> getListW() {


        return listW;

    }

    public LinkedHashSet<String> getListY() {


        return listY;

    }

    public LinkedHashSet<String> getListX() {


        return listX;

    }

    public LinkedHashSet<String> getListZ() {


        return listZ;

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
        private LinkedHashSet<String> list;

        public void setList(LinkedHashSet<String> listOut) {
            this.list = listOut;
        }
    }
}