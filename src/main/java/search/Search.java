package search;

import fileSearcher.FileSearcher;
import frame.SettingsFrame;
import main.MainClass;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;


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
    private static CopyOnWriteArrayList<String> listToLoad = new CopyOnWriteArrayList<>();
    private static boolean isUsable = false;
    private static boolean isManualUpdate = false;
    private static boolean isFileSearcherDefined = false;
    private static List<String> RecycleBin = Collections.synchronizedList(new LinkedList<>());

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
            for (String path : RecycleBin) {
                deletePathInList(path);
            }
            isUsable = true;
            RecycleBin.clear();
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
            firstWord = file.getName().charAt(0);
        } catch (Exception ignored) {

        }
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
        } else {
            isUsable = false;
        }
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

    public void loadAllLists() throws Exception{
        Serialize lists;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(SettingsFrame.dataPath + "\\data.dat"))) {
            lists = (Serialize) ois.readObject();
            listA.addAll(lists._listA);
            listB.addAll(lists._listB);
            listC.addAll(lists._listC);
            listD.addAll(lists._listD);
            listE.addAll(lists._listE);
            listF.addAll(lists._listF);
            listG.addAll(lists._listG);
            listH.addAll(lists._listH);
            listI.addAll(lists._listI);
            listJ.addAll(lists._listJ);
            listK.addAll(lists._listK);
            listL.addAll(lists._listL);
            listM.addAll(lists._listM);
            listN.addAll(lists._listN);
            listO.addAll(lists._listO);
            listP.addAll(lists._listP);
            listQ.addAll(lists._listQ);
            listR.addAll(lists._listR);
            listS.addAll(lists._listS);
            listT.addAll(lists._listT);
            listU.addAll(lists._listU);
            listV.addAll(lists._listV);
            listW.addAll(lists._listW);
            listX.addAll(lists._listX);
            listY.addAll(lists._listY);
            listZ.addAll(lists._listZ);
            listNum.addAll(lists._listNum);
            listPercentSign.addAll(lists._listPercentSign);
            listUnderline.addAll(lists._listUnderline);
            listUnique.addAll(lists._listUnique);
        } catch (Exception e) {
            throw new Exception("无法读取");
        }
    }


    public void saveLists() {
        Serialize lists = new Serialize();
        lists.setList();
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(SettingsFrame.dataPath + "\\data.dat"))) {
            oos.writeObject(lists);
        } catch (IOException | ConcurrentModificationException ignored) {

        }
    }



    private void searchFile(String ignorePath, int searchDepth) {
        if (!isFileSearcherDefined) {
            isFileSearcherDefined = true;
            String[] ignorePaths = getIgnorePaths(ignorePath);
            for (String each : ignorePaths) {
                FileSearcher.INSTANCE.addIgnorePath(each);
            }
            FileSearcher.INSTANCE.setSearchDepth(searchDepth);
        }
        File[] roots = File.listRoots();
        for (File root : roots) {
            String path = root.getAbsolutePath();
            path = path.substring(0, 2);
            __searchFile(path);
        }
        __searchFileIgnoreSearchDepth(getStartMenu());
        __searchFileIgnoreSearchDepth("C:\\ProgramData\\Microsoft\\Windows\\Start Menu");
        MainClass.showMessage("提示", "搜索完成");
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

    private void __searchFileIgnoreSearchDepth(String path) {
        FileSearcher.INSTANCE.clearResults();
        FileSearcher.INSTANCE.searchFilesIgnoreSearchDepth(path, "*");
        while (!FileSearcher.INSTANCE.ResultReady()) {
            try {
                Thread.sleep(1);
            } catch (InterruptedException ignored) {

            }
        }
        String results = FileSearcher.INSTANCE.getResult();
        FileSearcher.INSTANCE.deleteResult();
        String[] resultList = results.split("\n");
        for (String each : resultList) {
            File tmp = new File(each);
            String name = tmp.getName();
            char headWord = '\0';
            try {
                headWord = name.charAt(0);
                headWord = Character.toUpperCase(headWord);
            } catch (Exception ignored) {

            }
            switch (headWord) {
                case 'A':
                    try {
                        listA.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'B':
                    try {
                        listB.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'C':
                    try {
                        listC.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'D':
                    try {
                        listD.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'E':
                    try {
                        listE.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'F':
                    try {
                        listF.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'G':
                    try {
                        listG.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'H':
                    try {
                        listH.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'I':
                    try {
                        listI.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'J':
                    try {
                        listJ.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'K':
                    try {
                        listK.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'L':
                    try {
                        listL.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'M':
                    try {
                        listM.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'N':
                    try {
                        listN.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'O':
                    try {
                        listO.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'P':
                    try {
                        listP.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'Q':
                    try {
                        listQ.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'R':
                    try {
                        listR.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'S':
                    try {
                        listS.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'T':
                    try {
                        listT.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'U':
                    try {
                        listU.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'V':
                    try {
                        listV.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'W':
                    try {
                        listW.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'X':
                    try {
                        listX.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'Y':
                    try {
                        listY.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'Z':
                    try {
                        listZ.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                default:
                    if (Character.isDigit(headWord)) {
                        try {
                            listNum.add(each);
                        } catch (Exception ignored) {

                        }
                    } else if ('_' == headWord) {
                        try {
                            listUnderline.add(each);
                        } catch (Exception ignored) {

                        }
                    } else if ('%' == headWord) {
                        try {
                            listPercentSign.add(each);
                        } catch (Exception ignored) {

                        }
                    } else {
                        try {
                            listUnique.add(each);
                        } catch (Exception ignored) {

                        }
                    }
                    break;
            }
        }
    }

    private void __searchFile(String path) {
        FileSearcher.INSTANCE.clearResults();
        FileSearcher.INSTANCE.searchFiles(path, "*");
        while (!FileSearcher.INSTANCE.ResultReady()) {
            try {
                Thread.sleep(1);
            } catch (InterruptedException ignored) {

            }
        }
        String results = FileSearcher.INSTANCE.getResult();
        FileSearcher.INSTANCE.deleteResult();
        String[] resultList = results.split("\n");
        for (String each : resultList) {
            File tmp = new File(each);
            String name = tmp.getName();
            char headWord = '\0';
            try {
                headWord = name.charAt(0);
                headWord = Character.toUpperCase(headWord);
            } catch (Exception ignored) {

            }
            switch (headWord) {
                case 'A':
                    try {
                        listA.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'B':
                    try {
                        listB.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'C':
                    try {
                        listC.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'D':
                    try {
                        listD.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'E':
                    try {
                        listE.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'F':
                    try {
                        listF.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'G':
                    try {
                        listG.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'H':
                    try {
                        listH.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'I':
                    try {
                        listI.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'J':
                    try {
                        listJ.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'K':
                    try {
                        listK.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'L':
                    try {
                        listL.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'M':
                    try {
                        listM.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'N':
                    try {
                        listN.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'O':
                    try {
                        listO.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'P':
                    try {
                        listP.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'Q':
                    try {
                        listQ.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'R':
                    try {
                        listR.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'S':
                    try {
                        listS.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'T':
                    try {
                        listT.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'U':
                    try {
                        listU.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'V':
                    try {
                        listV.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'W':
                    try {
                        listW.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'X':
                    try {
                        listX.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'Y':
                    try {
                        listY.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                case 'Z':
                    try {
                        listZ.add(each);
                    } catch (Exception ignored) {

                    }
                    break;
                default:
                    if (Character.isDigit(headWord)) {
                        try {
                            listNum.add(each);
                        } catch (Exception ignored) {

                        }
                    } else if ('_' == headWord) {
                        try {
                            listUnderline.add(each);
                        } catch (Exception ignored) {

                        }
                    } else if ('%' == headWord) {
                        try {
                            listPercentSign.add(each);
                        } catch (Exception ignored) {

                        }
                    } else {
                        try {
                            listUnique.add(each);
                        } catch (Exception ignored) {

                        }
                    }
                    break;
            }
        }
        isManualUpdate = false;
        isUsable = true;
    }


    private String[] getIgnorePaths(String target) {
        int each;
        target = target.toLowerCase();
        ArrayList<String> list = new ArrayList<>();
        while ((each = target.indexOf(",")) != -1) {
            list.add(target.substring(0, each));
            target = target.substring(each + 1);
        }
        return list.toArray(new String[0]);
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
        public LinkedHashSet<String> _listA;
        public LinkedHashSet<String> _listB;
        public LinkedHashSet<String> _listC;
        public LinkedHashSet<String> _listD;
        public LinkedHashSet<String> _listE;
        public LinkedHashSet<String> _listF;
        public LinkedHashSet<String> _listG;
        public LinkedHashSet<String> _listH;
        public LinkedHashSet<String> _listI;
        public LinkedHashSet<String> _listJ;
        public LinkedHashSet<String> _listK;
        public LinkedHashSet<String> _listL;
        public LinkedHashSet<String> _listM;
        public LinkedHashSet<String> _listN;
        public LinkedHashSet<String> _listO;
        public LinkedHashSet<String> _listP;
        public LinkedHashSet<String> _listQ;
        public LinkedHashSet<String> _listR;
        public LinkedHashSet<String> _listS;
        public LinkedHashSet<String> _listT;
        public LinkedHashSet<String> _listU;
        public LinkedHashSet<String> _listV;
        public LinkedHashSet<String> _listW;
        public LinkedHashSet<String> _listX;
        public LinkedHashSet<String> _listY;
        public LinkedHashSet<String> _listZ;
        public LinkedHashSet<String> _listNum;
        public LinkedHashSet<String> _listPercentSign;
        public LinkedHashSet<String> _listUnderline;
        public LinkedHashSet<String> _listUnique;

        public void setList() {
            _listA = listA;
            _listB = listB;
            _listC = listC;
            _listD = listD;
            _listE = listE;
            _listF = listF;
            _listG = listG;
            _listH = listH;
            _listI = listI;
            _listJ = listJ;
            _listK = listK;
            _listL = listL;
            _listM = listM;
            _listN = listN;
            _listO = listO;
            _listP = listP;
            _listQ = listQ;
            _listR = listR;
            _listS = listS;
            _listT = listT;
            _listU = listU;
            _listV = listV;
            _listW = listW;
            _listX = listX;
            _listY = listY;
            _listZ = listZ;
            _listNum = listNum;
            _listPercentSign = listPercentSign;
            _listUnderline = listUnderline;
            _listUnique = listUnique;
        }
    }
}