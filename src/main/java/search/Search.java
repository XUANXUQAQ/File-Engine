package search;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

import javax.swing.filechooser.FileSystemView;


public class Search {
    private static LinkedList<String> listA = new LinkedList<>();
    private static LinkedList<String> listB = new LinkedList<>();
    private static LinkedList<String> listC = new LinkedList<>();
    private static LinkedList<String> listD = new LinkedList<>();
    private static LinkedList<String> listE = new LinkedList<>();
    private static LinkedList<String> listF = new LinkedList<>();
    private static LinkedList<String> listG = new LinkedList<>();
    private static LinkedList<String> listH = new LinkedList<>();
    private static LinkedList<String> listI = new LinkedList<>();
    private static LinkedList<String> listJ = new LinkedList<>();
    private static LinkedList<String> listK = new LinkedList<>();
    private static LinkedList<String> listL = new LinkedList<>();
    private static LinkedList<String> listM = new LinkedList<>();
    private static LinkedList<String> listN = new LinkedList<>();
    private static LinkedList<String> listO = new LinkedList<>();
    private static LinkedList<String> listP = new LinkedList<>();
    private static LinkedList<String> listQ = new LinkedList<>();
    private static LinkedList<String> listR = new LinkedList<>();
    private static LinkedList<String> listS = new LinkedList<>();
    private static LinkedList<String> listT = new LinkedList<>();
    private static LinkedList<String> listU = new LinkedList<>();
    private static LinkedList<String> listV = new LinkedList<>();
    private static LinkedList<String> listW = new LinkedList<>();
    private static LinkedList<String> listX = new LinkedList<>();
    private static LinkedList<String> listY = new LinkedList<>();
    private static LinkedList<String> listZ = new LinkedList<>();
    private static LinkedList<String> listNum = new LinkedList<>();
    private static LinkedList<String> listPercentSign = new LinkedList<>();
    private static LinkedList<String> listUnique = new LinkedList<>();
    private static LinkedList<String> listUnderline = new LinkedList<>();
    private static boolean isSearch = false;
    private static boolean isFocusLost = true;
    private final String desktop = FileSystemView.getFileSystemView().getHomeDirectory().getAbsolutePath();
    LinkedList<String> listRemain = new LinkedList<>();
    private String startMenu = getStartMenu();
    private boolean isFirstRun = true;
    private int num = 0;


    public boolean isSearch() {
        return isSearch;
    }

    public void setSearch(boolean b) {
        isSearch = b;
    }

    public boolean isIsFocusLost() {
        return isFocusLost;
    }

    public void setFocusLostStatus(boolean b) {
        isFocusLost = b;
    }

    public void searchFile(String ignorePath, int searchDepth) {
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
        isSearch = false;
    }


    private void __searchFile(String ignorePath, File path, int searchDepth) {
        ignorePath = ignorePath.toUpperCase();
        boolean exist = path.exists();
        if (exist && !isIgnore(path.getAbsolutePath().toUpperCase(), ignorePath)) {
            File[] files = path.listFiles();
            if (null == files || files.length == 0) {
                //System.out.println("空白目录");
            } else if (searchDepth >= count(path.getAbsolutePath(), "\\") || (path.getAbsolutePath().equals(startMenu)) || path.getAbsolutePath().equals(desktop)) {
                for (File file2 : files) {
                    String fileFullName = file2.getAbsolutePath();
                    String[] nameList = fileFullName.split("\\\\");
                    String fileName = nameList[nameList.length - 1];


                    char firstWord = fileName.charAt(0);
                    if (fileName.length() >= 2) {
                        char secondWord = fileName.charAt(1);

                        if (firstWord != '$' && (firstWord != '.' && secondWord != '.')) {
                            char headWord = Character.toUpperCase(firstWord);
                            switch (headWord) {
                                case 'A':
                                    listA.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'B':
                                    listB.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'C':
                                    listC.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'D':
                                    listD.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'E':
                                    listE.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'F':
                                    listF.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'G':
                                    listG.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'H':
                                    listH.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'I':
                                    listI.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'J':
                                    listJ.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'K':
                                    listK.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'L':
                                    listL.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'M':
                                    listM.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'N':
                                    listN.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'O':
                                    listO.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'P':
                                    listP.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'Q':
                                    listQ.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'R':
                                    listR.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'S':
                                    listS.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'T':
                                    listT.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'U':
                                    listU.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'V':
                                    listV.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'W':
                                    listW.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'X':
                                    listX.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'Y':
                                    listY.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'Z':
                                    listZ.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                default:
                                    if (Character.isDigit(headWord)) {
                                        listNum.add(file2.getAbsolutePath());
                                    } else if ('_' == headWord) {
                                        listUnderline.add(file2.getAbsolutePath());
                                    } else if ('%' == headWord) {
                                        listPercentSign.add(file2.getAbsolutePath());
                                    } else {
                                        listUnique.add(file2.getAbsolutePath());
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
                                    listA.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'B':
                                    listB.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'C':
                                    listC.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'D':
                                    listD.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'E':
                                    listE.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'F':
                                    listF.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'G':
                                    listG.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'H':
                                    listH.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'I':
                                    listI.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'J':
                                    listJ.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'K':
                                    listK.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'L':
                                    listL.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'M':
                                    listM.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'N':
                                    listN.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'O':
                                    listO.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'P':
                                    listP.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'Q':
                                    listQ.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'R':
                                    listR.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'S':
                                    listS.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'T':
                                    listT.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'U':
                                    listU.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'V':
                                    listV.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'W':
                                    listW.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'X':
                                    listX.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'Y':
                                    listY.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                case 'Z':
                                    listZ.add(file2.getAbsolutePath());
                                    num += 1;
                                    break;
                                default:
                                    if (Character.isDigit(headWord)) {
                                        listNum.add(file2.getAbsolutePath());
                                    } else if ('_' == headWord) {
                                        listUnderline.add(file2.getAbsolutePath());
                                    } else if ('%' == headWord) {
                                        listPercentSign.add(file2.getAbsolutePath());
                                    } else {
                                        listUnique.add(file2.getAbsolutePath());
                                    }
                                    num += 1;
                                    break;
                            }
                            if (file2.isDirectory()) {
                                listRemain.add(file2.getAbsolutePath());
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
     *               txt是否属于target的子字符串
     * @return true false
     */
    private boolean isIgnore(String txt, String target) {
        String[] list = target.split(",");
        for (String each : list) {
            if (txt.contains(each)) {
                return true;
            }
        }
        return false;
    }

    public LinkedList<String> getListA() {
        return listA;
    }

    public LinkedList<String> getListB() {
        return listB;
    }

    public LinkedList<String> getListC() {
        return listC;
    }

    public LinkedList<String> getListD() {
        return listD;
    }

    public LinkedList<String> getListE() {
        return listE;
    }

    public LinkedList<String> getListF() {
        return listF;
    }

    public LinkedList<String> getListG() {
        return listG;
    }

    public LinkedList<String> getListH() {
        return listH;
    }

    public LinkedList<String> getListI() {
        return listI;
    }

    public LinkedList<String> getListJ() {
        return listJ;
    }

    public LinkedList<String> getListK() {
        return listK;
    }

    public LinkedList<String> getListL() {
        return listL;
    }

    public LinkedList<String> getListM() {
        return listM;
    }

    public LinkedList<String> getListN() {
        return listN;
    }

    public LinkedList<String> getListO() {
        return listO;
    }

    public LinkedList<String> getListP() {
        return listP;
    }

    public LinkedList<String> getListQ() {
        return listQ;
    }

    public LinkedList<String> getListR() {
        return listR;
    }

    public LinkedList<String> getListNum() {
        return listNum;
    }

    public LinkedList<String> getListPercentSign() {
        return listPercentSign;
    }

    public LinkedList<String> getListS() {
        return listS;
    }

    public LinkedList<String> getListT() {
        return listT;
    }

    public LinkedList<String> getListUnique() {
        return listUnique;
    }

    public LinkedList<String> getListU() {
        return listU;
    }

    public LinkedList<String> getListUnderline() {
        return listUnderline;
    }

    public LinkedList<String> getListV() {
        return listV;
    }

    public LinkedList<String> getListW() {
        return listW;
    }

    public LinkedList<String> getListY() {
        return listY;
    }

    public LinkedList<String> getListX() {
        return listX;
    }

    public LinkedList<String> getListZ() {
        return listZ;
    }

    public void updateLists(String ignorePath, int searchDepth) {
        if (!isFirstRun) {
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
        } else {
            isFirstRun = false;
        }
        searchFile(ignorePath, searchDepth);
    }

    public void clearAll() {
        listA = null;
        listB = null;
        listC = null;
        listD = null;
        listE = null;
        listF = null;
        listG = null;
        listH = null;
        listI = null;
        listJ = null;
        listK = null;
        listL = null;
        listM = null;
        listN = null;
        listO = null;
        listP = null;
        listQ = null;
        listR = null;
        listS = null;
        listT = null;
        listU = null;
        listV = null;
        listW = null;
        listX = null;
        listY = null;
        listZ = null;
        listNum = null;
        listPercentSign = null;
        listUnique = null;
        listUnderline = null;
    }
}