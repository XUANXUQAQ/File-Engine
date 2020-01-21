package search;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

import javax.swing.filechooser.FileSystemView;


public class Search {
	static LinkedList<String> listRemain = new LinkedList<>();
	public static ArrayList<String> listA = new ArrayList<>();
	public static ArrayList<String> listB = new ArrayList<>();
	public static ArrayList<String> listC = new ArrayList<>();
	public static ArrayList<String> listD = new ArrayList<>();
	public static ArrayList<String> listE = new ArrayList<>();
	public static ArrayList<String> listF = new ArrayList<>();
	public static ArrayList<String> listG = new ArrayList<>();
	public static ArrayList<String> listH = new ArrayList<>();
	public static ArrayList<String> listI = new ArrayList<>();
	public static ArrayList<String> listJ = new ArrayList<>();
	public static ArrayList<String> listK = new ArrayList<>();
	public static ArrayList<String> listL = new ArrayList<>();
	public static ArrayList<String> listM = new ArrayList<>();
	public static ArrayList<String> listN = new ArrayList<>();
	public static ArrayList<String> listO = new ArrayList<>();
	public static ArrayList<String> listP = new ArrayList<>();
	public static ArrayList<String> listQ = new ArrayList<>();
	public static ArrayList<String> listR = new ArrayList<>();
	public static ArrayList<String> listS = new ArrayList<>();
	public static ArrayList<String> listT = new ArrayList<>();
	public static ArrayList<String> listU = new ArrayList<>();
	public static ArrayList<String> listV = new ArrayList<>();
	public static ArrayList<String> listW = new ArrayList<>();
	public static ArrayList<String> listX = new ArrayList<>();
	public static ArrayList<String> listY = new ArrayList<>();
	public static ArrayList<String> listZ = new ArrayList<>();
	public static ArrayList<String> listNum = new ArrayList<>();
	public static ArrayList<String> listPercentSign = new ArrayList<>();
	public static ArrayList<String> listUnique = new ArrayList<>();
	public static ArrayList<String> listUnderline = new ArrayList<>();
	public static String startMenu = getStartMenu();
	public static String desktop = FileSystemView.getFileSystemView().getHomeDirectory().getAbsolutePath();
	private static boolean isSearch = false;
	private static boolean isFirstRun = true;


	public boolean isSearch(){
		return isSearch;
	}
	public void setSearch(boolean b){
		isSearch = b;
	}

	public static void searchFile(String ignorePath, int searchDepth) {
		File[] roots = File.listRoots();
		for (File root: roots) {
			listRemain.add(root.getAbsolutePath());
		}


		while (!listRemain.isEmpty()) {
			String tmp = listRemain.pop();
			__searchFile(ignorePath, new File(tmp), searchDepth);
		}
		System.out.println("Search Done");
		isSearch = false;
	}


	private static void __searchFile(String ignorePath, File path, int searchDepth) {
		ignorePath = ignorePath.toUpperCase();
		boolean exist = path.exists();
		if ( exist && !isIgnore(path.getAbsolutePath().toUpperCase(), ignorePath)) {
			File[] files = path.listFiles();
			if (null == files || files.length == 0) {
				//System.out.println("�հ�Ŀ¼");
			} else if (searchDepth >= count(path.getAbsolutePath(), "\\")||(path.getAbsolutePath().equals(startMenu))||path.getAbsolutePath().equals(desktop)) {
				for (File file2 : files) {
					String fileFullName = file2.getAbsolutePath();
					String[] nameList = fileFullName.split("\\\\");
					String fileName = nameList[ nameList.length -1 ];


					char firstWord = fileName.charAt(0);
					if (fileName.length() >= 2) {
						char secondWord = fileName.charAt(1);

						if (firstWord != '$' && (firstWord != '.' && secondWord != '.')) {
							char headWord = Character.toUpperCase(firstWord);
							switch (headWord){
								case 'A':
									listA.add(file2.getAbsolutePath());
									break;
								case 'B':
									listB.add(file2.getAbsolutePath());
									break;
								case 'C':
									listC.add(file2.getAbsolutePath());
									break;
								case 'D':
									listD.add(file2.getAbsolutePath());
									break;
								case 'E':
									listE.add(file2.getAbsolutePath());
									break;
								case 'F':
									listF.add(file2.getAbsolutePath());
									break;
								case 'G':
									listG.add(file2.getAbsolutePath());
									break;
								case 'H':
									listH.add(file2.getAbsolutePath());
									break;
								case 'I':
									listI.add(file2.getAbsolutePath());
									break;
								case 'J':
									listJ.add(file2.getAbsolutePath());
									break;
								case 'K':
									listK.add(file2.getAbsolutePath());
									break;
								case 'L':
									listL.add(file2.getAbsolutePath());
									break;
								case 'M':
									listM.add(file2.getAbsolutePath());
									break;
								case 'N':
									listN.add(file2.getAbsolutePath());
									break;
								case 'O':
									listO.add(file2.getAbsolutePath());
									break;
								case 'P':
									listP.add(file2.getAbsolutePath());
									break;
								case 'Q':
									listQ.add(file2.getAbsolutePath());
									break;
								case 'R':
									listR.add(file2.getAbsolutePath());
									break;
								case 'S':
									listS.add(file2.getAbsolutePath());
									break;
								case 'T':
									listT.add(file2.getAbsolutePath());
									break;
								case 'U':
									listU.add(file2.getAbsolutePath());
									break;
								case 'V':
									listV.add(file2.getAbsolutePath());
									break;
								case 'W':
									listW.add(file2.getAbsolutePath());
									break;
								case 'X':
									listX.add(file2.getAbsolutePath());
									break;
								case 'Y':
									listY.add(file2.getAbsolutePath());
									break;
								case 'Z':
									listZ.add(file2.getAbsolutePath());
									break;
								default:
									if (Character.isDigit(headWord)) {listNum.add(file2.getAbsolutePath());}
									else if ('_' == headWord) {listUnderline.add(file2.getAbsolutePath());}
									else if ('%' == headWord) {listPercentSign.add(file2.getAbsolutePath());}
									else {listUnique.add(file2.getAbsolutePath());}
									break;
							}
							if (file2.isDirectory()) {
								listRemain.add(file2.getAbsolutePath());
							}
						}


					}else {
						if (firstWord != '$') {
							char headWord = Character.toUpperCase(firstWord);
							switch (headWord){
								case 'A':
									listA.add(file2.getAbsolutePath());
									break;
								case 'B':
									listB.add(file2.getAbsolutePath());
									break;
								case 'C':
									listC.add(file2.getAbsolutePath());
									break;
								case 'D':
									listD.add(file2.getAbsolutePath());
									break;
								case 'E':
									listE.add(file2.getAbsolutePath());
									break;
								case 'F':
									listF.add(file2.getAbsolutePath());
									break;
								case 'G':
									listG.add(file2.getAbsolutePath());
									break;
								case 'H':
									listH.add(file2.getAbsolutePath());
									break;
								case 'I':
									listI.add(file2.getAbsolutePath());
									break;
								case 'J':
									listJ.add(file2.getAbsolutePath());
									break;
								case 'K':
									listK.add(file2.getAbsolutePath());
									break;
								case 'L':
									listL.add(file2.getAbsolutePath());
									break;
								case 'M':
									listM.add(file2.getAbsolutePath());
									break;
								case 'N':
									listN.add(file2.getAbsolutePath());
									break;
								case 'O':
									listO.add(file2.getAbsolutePath());
									break;
								case 'P':
									listP.add(file2.getAbsolutePath());
									break;
								case 'Q':
									listQ.add(file2.getAbsolutePath());
									break;
								case 'R':
									listR.add(file2.getAbsolutePath());
									break;
								case 'S':
									listS.add(file2.getAbsolutePath());
									break;
								case 'T':
									listT.add(file2.getAbsolutePath());
									break;
								case 'U':
									listU.add(file2.getAbsolutePath());
									break;
								case 'V':
									listV.add(file2.getAbsolutePath());
									break;
								case 'W':
									listW.add(file2.getAbsolutePath());
									break;
								case 'X':
									listX.add(file2.getAbsolutePath());
									break;
								case 'Y':
									listY.add(file2.getAbsolutePath());
									break;
								case 'Z':
									listZ.add(file2.getAbsolutePath());
									break;
								default:
									if (Character.isDigit(headWord)) {listNum.add(file2.getAbsolutePath());}
									else if ('_' == headWord) {listUnderline.add(file2.getAbsolutePath());}
									else if ('%' == headWord) {listPercentSign.add(file2.getAbsolutePath());}
									else {listUnique.add(file2.getAbsolutePath());}
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

	private static String getStartMenu() {
		String startMenu;
		BufferedReader bufrIn;
		try {
			Process getStartMenu = Runtime.getRuntime().exec("reg query \"HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders\" "+"/v "+"\"Start Menu\"");
			bufrIn = new BufferedReader(new InputStreamReader(getStartMenu.getInputStream(), StandardCharsets.UTF_8));
			while((startMenu = bufrIn.readLine()) != null) {
				if (startMenu.contains("REG_SZ")){
					startMenu = startMenu.substring(startMenu.indexOf("REG_SZ")+10);
					return startMenu;
				}
			}
		} catch (IOException e) {

			e.printStackTrace();
		}
		return null;
	}
	/**
	 *
	 * @param srcText Դ�ַ���
	 * @param findText ��Ҫ�������ַ���
	 * @return count
	 */
	public static int count(String srcText, String findText){
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
	 * txt�Ƿ�����target�����ַ���
	 * @return true false
	 */
	private static boolean isIgnore(String txt, String target) {
		String[] list = target.split(",");
		for (String each : list) {
			if (txt.contains(each)) {
				return true;
			}
		}
		return false;
	}


	public static void updateLists(String ignorePath, int searchDepth) {
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
		}else{
			isFirstRun = false;
		}
		searchFile(ignorePath, searchDepth);
	}
}