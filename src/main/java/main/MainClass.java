package main;

import com.alibaba.fastjson.JSONObject;
import fileMonitor.FileMonitor;
import frame.SearchBar;
import frame.SettingsFrame;
import frame.TaskBar;
import search.Search;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.util.Objects;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;



public class MainClass {
	public static boolean mainExit = false;
	private static SearchBar searchBar = new SearchBar();
	private static Search search = new Search();
	public static String name;

	public static void setMainExit(boolean b){
		mainExit = b;
	}
	private static TaskBar taskBar = null;
	public static void showMessage(String caption, String message){
		if (taskBar != null){
			taskBar.showMessage(caption, message);
		}
	}
	public static boolean isAdmin() {
		try {
			ProcessBuilder processBuilder = new ProcessBuilder("cmd.exe");
			Process process = processBuilder.start();
			PrintStream printStream = new PrintStream(process.getOutputStream(), true);
			Scanner scanner = new Scanner(process.getInputStream());
			printStream.println("@echo off");
			printStream.println(">nul 2>&1 \"%SYSTEMROOT%\\system32\\cacls.exe\" \"%SYSTEMROOT%\\system32\\config\\system\"");
			printStream.println("echo %errorlevel%");

			boolean printedErrorlevel = false;
			while (true) {
				String nextLine = scanner.nextLine();
				if (printedErrorlevel) {
					int errorlevel = Integer.parseInt(nextLine);
					return errorlevel == 0;
				} else if (nextLine.equals("echo %errorlevel%")) {
					printedErrorlevel = true;
				}
			}
		} catch (IOException e) {
			return false;
		}
	}

	private static void readMonitorFile(){

	}
	private static void copyFile(InputStream source, File dest) {
		try(OutputStream os = new FileOutputStream(dest);BufferedInputStream bis = new BufferedInputStream(source);BufferedOutputStream bos = new BufferedOutputStream(os)) {
			byte[]buffer = new byte[8192];
			int count = bis.read(buffer);
			while(count != -1){
				//使用缓冲流写数据
				bos.write(buffer,0,count);
				//刷新
				bos.flush();
				count = bis.read(buffer);
			}
		} catch (IOException ignored) {

		}
	}

	public static void deleteDir(String path){
		File file = new File(path);
		if(!file.exists()){//判断是否待删除目录是否存在
			return;
		}

		String[] content = file.list();//取得当前目录下所有文件和文件夹
		if (content != null) {
			for (String name : content) {
				File temp = new File(path, name);
				if (temp.isDirectory()) {//判断是否是目录
					deleteDir(temp.getAbsolutePath());//递归调用，删除目录里的内容
					temp.delete();//删除空目录
				} else {
					if (!temp.delete()) {//直接删除文件
						System.err.println("Failed to delete " + name);
					}
				}
			}
		}
	}

	public static void main(String[] args) {
		String osArch =System.getProperty("os.arch");
		if (osArch.contains("64")){
			name = "search_x64.exe";
		}else{
			name = "search_x86.exe";
		}
		File settings = new File(System.getenv("Appdata") + "/settings.json");
		File caches = new File("cache.dat");
		File data = new File("data");
		//清空tmp
		deleteDir(SettingsFrame.tmp.getAbsolutePath());
		if (!settings.exists()){
			String ignorePath = "";
			JSONObject json = new JSONObject();
			json.put("hotkey", "Ctrl + Alt + J");
			json.put("ignorePath", ignorePath);
			json.put("isStartup", false);
			json.put("updateTimeLimit", 300);
			json.put("cacheNumLimit", 1000);
			json.put("searchDepth", 6);
			json.put("priorityFolder", "");
			json.put("dataPath", data.getAbsolutePath());
			try(BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
				buffW.write(json.toJSONString());
			} catch (IOException ignored) {

			}
		}
		File target;
		InputStream fileMonitorDll64 = MainClass.class.getResourceAsStream("/fileMonitor64.dll");
		InputStream fileMonitorDll32 = MainClass.class.getResourceAsStream("/fileMonitor32.dll");

		target = new File("fileMonitor.dll");
		if (!target.exists()) {
			File dll;
			if (name.contains("x64")) {
				copyFile(fileMonitorDll64, target);
				System.out.println("已加载64位dll");
				dll = new File("fileMonitor64.dll");
			}else{
				copyFile(fileMonitorDll32, target);
				System.out.println("已加载32位dll");
				dll = new File("fileMonitor32.dll");
			}
			dll.renameTo(new File("fileMonitor.dll"));
		}
		if (!caches.exists()){
			try {
				caches.createNewFile();
			} catch (IOException e) {
				JOptionPane.showMessageDialog(null, "创建缓存文件失败，程序正在退出");
				mainExit = true;
			}
		}
		if (!SettingsFrame.tmp.exists()){
			SettingsFrame.tmp.mkdir();
		}
		taskBar = new TaskBar();
		taskBar.showTaskBar();

		File[] roots = File.listRoots();
		ExecutorService fixedThreadPool = Executors.newFixedThreadPool(roots.length+4);

		SettingsFrame.initSettings();

		data = new File(SettingsFrame.dataPath);
		if (data.isDirectory() && data.exists()){
			if (Objects.requireNonNull(data.list()).length == 30){
				System.out.println("检测到data文件，正在读取");
				showMessage("提示", "检测到data文件，正在读取");
				search.setUsable(false);
				search.loadAllLists();
				search.setUsable(true);
				System.out.println("读取完成");
				showMessage("提示", "读取完成");
			}else{
				System.out.println("检测到data文件损坏，开始搜索并创建data文件");
				showMessage("提示", "检检测到data文件损坏，开始搜索并创建data文件");
				search.setManualUpdate(true);
			}
		}else{
			System.out.println("未检测到data文件，开始搜索并创建data文件");
			search.setManualUpdate(true);
		}


		for(File root:roots){
			if (!isAdmin()){
				System.out.println("无管理员权限，文件监控功能已关闭");
				taskBar.showMessage("警告","无管理员权限，文件监控功能已关闭");
			}else {
				fixedThreadPool.execute(() -> FileMonitor.INSTANCE.monitor(root.getAbsolutePath(), SettingsFrame.tmp.getAbsolutePath(), SettingsFrame.tmp.getAbsolutePath() + "\\CLOSE"));
			}
		}


		fixedThreadPool.execute(()->{
			//检测文件改动线程
			int countAdd = 0;
			int countRemove = 0;
			String filesToAdd;
			String filesToRemove;
			//分割字符串
			while (!mainExit) {
				if (!search.isManualUpdate()) {
					int loopAdd = 0;
					int loopRemove = 0;
					try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(SettingsFrame.tmp.getAbsolutePath() + "\\fileAdded.txt"))));
						 BufferedReader br2 = new BufferedReader(new InputStreamReader(new FileInputStream(new File(SettingsFrame.tmp.getAbsolutePath() + "\\fileRemoved.txt"))))) {
						while ((filesToAdd = br.readLine()) != null) {
							loopAdd++;
							if (loopAdd > countAdd) {
								countAdd++;
								search.addFileToLoadBin(filesToAdd);
								break;
							}
						}
						while ((filesToRemove = br2.readLine()) != null) {
							loopRemove++;
							if (loopRemove > countRemove) {
								countRemove++;
								search.addToRecycleBin(filesToRemove);
								break;
							}
						}
					} catch (IOException ignored) {

					} finally {
						try {
							Thread.sleep(50);
						} catch (InterruptedException ignored) {

						}
					}
				}
			}
		});


		fixedThreadPool.execute(() -> {
			// 时间检测线程
			long count = 0;
			while (!mainExit) {
				boolean isUsing = searchBar.getUsing();
				count++;
				if (count >= SettingsFrame.updateTimeLimit << 10 && !isUsing && !search.isManualUpdate()) {
					count = 0;
					System.out.println("正在更新本地索引data文件");
					search.saveLists();
				}

				try {
					Thread.sleep(1);
				} catch (InterruptedException ignore) {

				}
			}
		});


		//刷新屏幕线程
		fixedThreadPool.execute(() -> {
			Container panel;
			while (!mainExit) {
				try {
					panel = searchBar.getPanel();
					panel.repaint();
				} catch (Exception ignored) {

				}finally {
					try {
						Thread.sleep(50);
					} catch (InterruptedException ignored) {

					}
				}
			}
		});


		//搜索线程
		fixedThreadPool.execute(() ->{
			while (!mainExit){
				if (search.isManualUpdate()){
					search.setUsable(false);
					System.out.println("已收到更新请求");
					search.updateLists(SettingsFrame.ignorePath, SettingsFrame.searchDepth);
				}
				try {
					Thread.sleep(16);
				} catch (InterruptedException ignored) {

				}
			}
		});


		while(true) {
			// 主循环开始
			if (!search.isIsFocusLost()){
				searchBar.showSearchbar();
			}
			try {
				Thread.sleep(100);
			} catch (InterruptedException ignored) {

			}
			if (mainExit){
				if (search.isUsable()) {
					File CLOSEDLL = new File(SettingsFrame.tmp.getAbsolutePath() + "\\CLOSE");
					try {
						CLOSEDLL.createNewFile();
					} catch (IOException ignored) {

					}
					System.out.println("即将退出，保存最新文件列表到data");
					search.mergeFileToList();
					search.saveLists();
				}
				fixedThreadPool.shutdown();
				System.exit(0);
			}
		}
	}
}
