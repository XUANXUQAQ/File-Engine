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
	private static File fileWatcherTXT = new File(SettingsFrame.tmp.getAbsolutePath()+ "\\fileMonitor.txt");
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

	private static void copyFile(InputStream source, File dest) {
		try(OutputStream os = new FileOutputStream(dest);BufferedInputStream bis = new BufferedInputStream(source);BufferedOutputStream bos = new BufferedOutputStream(os)) {
			byte[]buffer = new byte[8192];
			int count = bis.read(buffer);
			while(count != -1){
				//ʹ�û�����д����
				bos.write(buffer,0,count);
				//ˢ��
				bos.flush();
				count = bis.read(buffer);
			}
		} catch (IOException ignored) {

		}
	}

	public static void deleteDir(String path){
		File file = new File(path);
		if(!file.exists()){//�ж��Ƿ��ɾ��Ŀ¼�Ƿ����
			return;
		}

		String[] content = file.list();//ȡ�õ�ǰĿ¼�������ļ����ļ���
		if (content != null) {
			for (String name : content) {
				File temp = new File(path, name);
				if (temp.isDirectory()) {//�ж��Ƿ���Ŀ¼
					deleteDir(temp.getAbsolutePath());//�ݹ���ã�ɾ��Ŀ¼�������
					temp.delete();//ɾ����Ŀ¼
				} else {
					if (!temp.delete()) {//ֱ��ɾ���ļ�
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
		//���tmp
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
				System.out.println("�Ѽ���64λdll");
				dll = new File("fileMonitor64.dll");
			}else{
				copyFile(fileMonitorDll32, target);
				System.out.println("�Ѽ���32λdll");
				dll = new File("fileMonitor32.dll");
			}
			dll.renameTo(new File("fileMonitor.dll"));
		}
		if (!caches.exists()){
			try {
				caches.createNewFile();
			} catch (IOException e) {
				JOptionPane.showMessageDialog(null, "���������ļ�ʧ�ܣ����������˳�");
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
				System.out.println("��⵽data�ļ������ڶ�ȡ");
				showMessage("��ʾ", "��⵽data�ļ������ڶ�ȡ");
				search.setUsable(false);
				search.loadAllLists();
				search.setUsable(true);
				System.out.println("��ȡ���");
				showMessage("��ʾ", "��ȡ���");
			}else{
				System.out.println("��⵽data�ļ��𻵣���ʼ����������data�ļ�");
				showMessage("��ʾ", "���⵽data�ļ��𻵣���ʼ����������data�ļ�");
				search.setManualUpdate(true);
			}
		}else{
			System.out.println("δ��⵽data�ļ�����ʼ����������data�ļ�");
			search.setManualUpdate(true);
		}


		for(File root:roots){
			fixedThreadPool.execute(()-> {
				FileMonitor.INSTANCE.fileWatcher(root.getAbsolutePath(), SettingsFrame.tmp.getAbsolutePath() + "\\fileMonitor.txt", SettingsFrame.tmp.getAbsolutePath() + "\\CLOSE");
			});
		}

		if (!isAdmin()){
			System.out.println("�޹���ԱȨ�ޣ��ļ���ع�������");
			taskBar.showMessage("����","δ��ȡ����ԱȨ�ޣ��ļ���ع�������");
		}

		fixedThreadPool.execute(() -> {
			// ʱ�����߳�
			long count = 0;
			while (!mainExit) {
				boolean isUsing = searchBar.getUsing();
				boolean isSleep = searchBar.getSleep();
				count++;
				if (count >= SettingsFrame.updateTimeLimit << 10 && !isUsing && !isSleep && !search.isManualUpdate()) {
					count = 0;
					System.out.println("���ڸ��±�������data�ļ�");
					search.saveLists();
				}

				try {
					Thread.sleep(1);
				} catch (InterruptedException ignore) {

				}
			}
		});


		//ˢ����Ļ�߳�
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


		//�����߳�
		fixedThreadPool.execute(() ->{
			while (!mainExit){
				if (search.isManualUpdate()){
					search.setUsable(false);
					System.out.println("���յ���������");
					search.updateLists(SettingsFrame.ignorePath, SettingsFrame.searchDepth);
				}
				try {
					Thread.sleep(16);
				} catch (InterruptedException ignored) {

				}
			}
		});

		//����ļ��Ķ��߳�
		fixedThreadPool.execute(() -> {
			long count = 0;
			while (!mainExit) {
				try (BufferedReader bw = new BufferedReader(new FileReader(fileWatcherTXT))) {
					long loop = 0;
					String line;
					while ((line = bw.readLine()) != null) {
						loop += 1;
						if (loop > count) {
							String[] strings = line.split(" : ");
							switch (strings[0]) {
								case "file add":
									search.FilesToAdd(strings[1]);
									break;
								case "file renamed":
									String[] add = strings[1].split("->");
									search.addToRecycleBin(add[0]);
									search.FilesToAdd(add[1]);
									break;
								case "file removed":
									search.addToRecycleBin(strings[1]);
									break;
							}
							count += 1;
						}
					}
					Thread.sleep(50);
				} catch (IOException | InterruptedException ignored) {

				}
			}
		});



		do {
			// ��ѭ����ʼ
			if (!search.isIsFocusLost()){
				searchBar.showSearchbar();
			}
			try {
				Thread.sleep(100);
			} catch (InterruptedException ignored) {

			}
			if (mainExit){
				File close = new File(SettingsFrame.tmp.getAbsolutePath() + "\\" + "CLOSE");
				try {
					close.createNewFile();
					if (search.isUsable()) {
						System.out.println("�����˳������������ļ��б�data");
						search.mergeFileToList();
						search.saveLists();
					}
				} catch (IOException ignored) {

				}
				fixedThreadPool.shutdown();
				System.exit(0);
			}
		}while(true);
	}
}
