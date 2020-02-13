package search;

import fileSearcher.FileSearcher;
import frame.SettingsFrame;
import main.MainClass;
import pinyin.PinYinConverter;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.regex.Pattern;


public class Search {
    private static HashSet<byte[]> listA = new HashSet<>();
    private static HashSet<byte[]> listB = new HashSet<>();
    private static HashSet<byte[]> listC = new HashSet<>();
    private static HashSet<byte[]> listD = new HashSet<>();
    private static HashSet<byte[]> listE = new HashSet<>();
    private static HashSet<byte[]> listF = new HashSet<>();
    private static HashSet<byte[]> listG = new HashSet<>();
    private static HashSet<byte[]> listH = new HashSet<>();
    private static HashSet<byte[]> listI = new HashSet<>();
    private static HashSet<byte[]> listJ = new HashSet<>();
    private static HashSet<byte[]> listK = new HashSet<>();
    private static HashSet<byte[]> listL = new HashSet<>();
    private static HashSet<byte[]> listM = new HashSet<>();
    private static HashSet<byte[]> listN = new HashSet<>();
    private static HashSet<byte[]> listO = new HashSet<>();
    private static HashSet<byte[]> listP = new HashSet<>();
    private static HashSet<byte[]> listQ = new HashSet<>();
    private static HashSet<byte[]> listR = new HashSet<>();
    private static HashSet<byte[]> listS = new HashSet<>();
    private static HashSet<byte[]> listT = new HashSet<>();
    private static HashSet<byte[]> listU = new HashSet<>();
    private static HashSet<byte[]> listV = new HashSet<>();
    private static HashSet<byte[]> listW = new HashSet<>();
    private static HashSet<byte[]> listX = new HashSet<>();
    private static HashSet<byte[]> listY = new HashSet<>();
    private static HashSet<byte[]> listZ = new HashSet<>();
    private static HashSet<byte[]> listNum = new HashSet<>();
    private static HashSet<byte[]> listPercentSign = new HashSet<>();
    private static HashSet<byte[]> listUnique = new HashSet<>();
    private static HashSet<byte[]> listUnderline = new HashSet<>();
    private static CopyOnWriteArrayList<byte[]> listToLoad = new CopyOnWriteArrayList<>();
    private static boolean isUsable = false;
    private static boolean isManualUpdate = false;
    private static boolean isFileSearcherDefined = false;
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

    public void loadAllLists() throws Exception {
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
        Pattern pattern = Pattern.compile("\n");
        String[] resultList = pattern.split(results);
        for (String each : resultList) {
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
        Pattern pattern = Pattern.compile("\n");
        String[] resultList = pattern.split(results);
        for (String each : resultList) {
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

    public HashSet<byte[]> getListA() {

        return listA;
    }

    public HashSet<byte[]> getListB() {

        return listB;
    }

    public HashSet<byte[]> getListC() {

        return listC;

    }

    public HashSet<byte[]> getListD() {


        return listD;

    }

    public HashSet<byte[]> getListE() {


        return listE;

    }

    public HashSet<byte[]> getListF() {


        return listF;

    }

    public HashSet<byte[]> getListG() {


        return listG;

    }

    public HashSet<byte[]> getListH() {


        return listH;

    }

    public HashSet<byte[]> getListI() {


        return listI;

    }

    public HashSet<byte[]> getListJ() {


        return listJ;

    }

    public HashSet<byte[]> getListK() {


        return listK;

    }

    public HashSet<byte[]> getListL() {


        return listL;

    }

    public HashSet<byte[]> getListM() {


        return listM;

    }

    public HashSet<byte[]> getListN() {


        return listN;

    }

    public HashSet<byte[]> getListO() {

        return listO;

    }

    public HashSet<byte[]> getListP() {


        return listP;

    }

    public HashSet<byte[]> getListQ() {


        return listQ;

    }

    public HashSet<byte[]> getListR() {


        return listR;

    }

    public HashSet<byte[]> getListNum() {


        return listNum;

    }

    public HashSet<byte[]> getListPercentSign() {


        return listPercentSign;

    }

    public HashSet<byte[]> getListS() {


        return listS;

    }

    public HashSet<byte[]> getListT() {


        return listT;

    }

    public HashSet<byte[]> getListUnique() {

        return listUnique;
    }

    public HashSet<byte[]> getListU() {


        return listU;

    }

    public HashSet<byte[]> getListUnderline() {

        return listUnderline;

    }

    public HashSet<byte[]> getListV() {


        return listV;

    }

    public HashSet<byte[]> getListW() {


        return listW;

    }

    public HashSet<byte[]> getListY() {


        return listY;

    }

    public HashSet<byte[]> getListX() {


        return listX;

    }

    public HashSet<byte[]> getListZ() {


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
        public HashSet<byte[]> _listA;
        public HashSet<byte[]> _listB;
        public HashSet<byte[]> _listC;
        public HashSet<byte[]> _listD;
        public HashSet<byte[]> _listE;
        public HashSet<byte[]> _listF;
        public HashSet<byte[]> _listG;
        public HashSet<byte[]> _listH;
        public HashSet<byte[]> _listI;
        public HashSet<byte[]> _listJ;
        public HashSet<byte[]> _listK;
        public HashSet<byte[]> _listL;
        public HashSet<byte[]> _listM;
        public HashSet<byte[]> _listN;
        public HashSet<byte[]> _listO;
        public HashSet<byte[]> _listP;
        public HashSet<byte[]> _listQ;
        public HashSet<byte[]> _listR;
        public HashSet<byte[]> _listS;
        public HashSet<byte[]> _listT;
        public HashSet<byte[]> _listU;
        public HashSet<byte[]> _listV;
        public HashSet<byte[]> _listW;
        public HashSet<byte[]> _listX;
        public HashSet<byte[]> _listY;
        public HashSet<byte[]> _listZ;
        public HashSet<byte[]> _listNum;
        public HashSet<byte[]> _listPercentSign;
        public HashSet<byte[]> _listUnderline;
        public HashSet<byte[]> _listUnique;

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