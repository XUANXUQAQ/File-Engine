package main;
import search.*;

import java.awt.*;
import java.io.*;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import fileMonitor.FileMonitor;
import frame.*;
import com.alibaba.fastjson.*;
import javax.swing.*;


public class Main {
	private static int updateTimeLimit = 600;
	public static boolean mainExit = false;
	private static String ignorePath;
	private static int searchDepth;
	private static SearchBar searchBar = new SearchBar();
	private static Search search = new Search();
	private static File fileWatcherTXT = new File("tmp\\fileMonitor.txt");

	public static void setMainExit(boolean b){
		mainExit = b;
	}
	public static void setIgnorePath(String paths){
		ignorePath = paths + "C:\\Config.Msi,C:\\Windows";
	}
	public static void setSearchDepth(int searchDepth1){
		searchDepth = searchDepth1;
	}
	public static void setUpdateTimeLimit(int updateTimeLimit1){
		updateTimeLimit = updateTimeLimit1;
	}


	public static void main(String[] args) {
		File settings = new File("settings.json");
		File caches = new File("cache.dat");
		File files = new File("Files");
		File tmp = new File("tmp");
		if (!settings.exists()){
			String ignorePath = "";
			JSONObject json = new JSONObject();
			json.put("hotkey", "Ctrl + Alt + J");
			json.put("ignorePath", ignorePath);
			json.put("isStartup", false);
			json.put("updateTimeLimit", 300);
			json.put("cacheNumLimit", 1000);
			json.put("searchDepth", 6);
			try(BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
				buffW.write(json.toJSONString());
			} catch (IOException ignored) {

			}
		}
		if (!caches.exists()){
			try {
				caches.createNewFile();
			} catch (IOException e) {
				JOptionPane.showMessageDialog(null, "���������ļ�ʧ�ܣ����������˳�");
				mainExit = true;
			}
		}
		if (!files.exists()){
			files.mkdir();
		}
		if (!tmp.exists()){
			tmp.mkdir();
		}
		TaskBar taskBar = new TaskBar();
		taskBar.showTaskBar();
		ignorePath = "";
		searchDepth = 0;
		try(BufferedReader settingReader = new BufferedReader(new FileReader(settings))) {
			String line;
			StringBuilder result = new StringBuilder();
			while ((line = settingReader.readLine()) != null){
				result.append(line);
			}
			JSONObject allSettings = JSON.parseObject(result.toString());
			searchDepth = allSettings.getInteger("searchDepth");
			ignorePath = allSettings.getString("ignorePath");
			updateTimeLimit = allSettings.getInteger("updateTimeLimit");
		} catch (IOException ignored) {

		}
		ignorePath = ignorePath + "C:\\Config.Msi,C:\\Windows";
		SettingsFrame.initSettings();

		File data = new File("data");
		if (data.isDirectory() && data.exists()){
			if (Objects.requireNonNull(data.list()).length == 30){
				System.out.println("��⵽data�ļ������ڶ�ȡ");
				search.setUsable(false);
				search.loadAllLists();
				search.setUsable(true);
				System.out.println("��ȡ���");
			}else{
				System.out.println("��⵽data�ļ��𻵣���ʼ����������data�ļ�");
				search.setManualUpdate(true);
			}
		}else{
			System.out.println("δ��⵽data�ļ�����ʼ����������data�ļ�");
			search.setManualUpdate(true);
		}

		File[] roots = File.listRoots();
		ExecutorService fixedThreadPool = Executors.newFixedThreadPool(roots.length+4);

		for(File root:roots) {
			fixedThreadPool.execute(() -> FileMonitor.INSTANCE.fileWatcher(root.getAbsolutePath(), tmp.getAbsolutePath() + "\\" + "fileMonitor.txt", tmp.getAbsolutePath() + "\\"+"CLOSE"));
		}

		fixedThreadPool.execute(() -> {
			// ʱ�����߳�
			long count = 0;
			long usingCount = 0;
			updateTimeLimit = updateTimeLimit * 1000;
			while (!mainExit) {
				boolean isUsing = searchBar.getUsing();
				boolean isSleep = searchBar.getSleep();
				count++;
				if (count >= updateTimeLimit && !isUsing && !isSleep && !search.isManualUpdate()) {
					count = 0;
					System.out.println("���ڸ��±�������data�ļ�");
					search.saveLists();
				}

				if (!isUsing){
					usingCount++;
					if (usingCount > 900000 && search.isUsable()) {
						System.out.println("��⵽��ʱ��δʹ�ã��Զ��ͷ��ڴ�ռ䣬��������");
						searchBar.setSleep(true);
						search.setUsable(false);
						search.saveAndReleaseLists();
					}
				}else{
					usingCount = 0;
					if (!search.isUsable() && !search.isManualUpdate()) {
						System.out.println("��⵽��ʼʹ�ã������б�");
						search.setUsable(false);
						searchBar.setSleep(false);
						search.loadAllLists();
						search.setUsable(true);
					}
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
					search.updateLists(ignorePath, searchDepth);
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
				System.out.println("�����˳������������ļ��б�data");
				search.mergeListToadd();
				search.saveLists();
				File close = new File(tmp.getAbsolutePath() + "\\" + "CLOSE");
				try {
					close.createNewFile();
					Thread.sleep(100);
					fileWatcherTXT.delete();
					close.delete();
				} catch (IOException | InterruptedException ignored) {

				}
				fixedThreadPool.shutdownNow();
				search.clearAll();
				System.exit(0);
			}
		}while(true);
	}
}
