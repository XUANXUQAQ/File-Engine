package main;
import search.*;

import java.awt.*;
import java.io.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import frame.*;
import com.alibaba.fastjson.*;



public class Main {
	private static BufferedReader settingReader;
	private static int updateTimeLimit = 600;
	private static boolean mainExit = false;
	private static String ignorePath;
	private static int searchDepth;
	
	
	public static void main(String[] args) {
		File settings = new File("settings.json");
		File caches = new File("cache.dat");
		File files = new File("Files");
		if (!settings.exists()){
			String ignorePath = "";
			JSONObject json = new JSONObject();
			json.put("ignorePath", ignorePath);
			json.put("isStartup", false);
			json.put("updateTimeLimit", 300);
			json.put("cacheNumLimit", 1000);
			json.put("searchDepth", 6);
			try(BufferedWriter buffW = new BufferedWriter(new FileWriter(settings))) {
				buffW.write(json.toJSONString());
			} catch (IOException e) {
				//e.printStackTrace();
			}
		}
		if (!caches.exists()){
			try {
				caches.createNewFile();
			} catch (IOException ignored) {

			}
		}
		if (!files.exists()){
			files.mkdir();
		}
		TaskBar taskBar = new TaskBar();
		ignorePath = "";
		searchDepth = 0;
		try {
			settingReader = new BufferedReader(new FileReader(settings));
			String line;
			StringBuilder result = new StringBuilder();
			while ((line = settingReader.readLine()) != null){
				result.append(line);
			}
			JSONObject allSettings = JSON.parseObject(result.toString());
			searchDepth = allSettings.getInteger("searchDepth");
			ignorePath = allSettings.getString("ignorePath");
			updateTimeLimit = allSettings.getInteger("updateTimeLimit");
		} catch (FileNotFoundException e1) {
			//e1.printStackTrace();
		} catch (IOException e) {
			//e.printStackTrace();
		}finally {
			try {
				settingReader.close();
			} catch (IOException e) {
				//e.printStackTrace();
			}
		}
		ignorePath = ignorePath + "C:\\Config.Msi,C:\\Windows";

		SearchBar searchBar = new SearchBar();
		CheckHotKey HotKeyListener = new CheckHotKey();
		SettingsFrame.initCacheLimit();

		new Search().setSearch(true);



		ExecutorService fixedThreadPool = Executors.newFixedThreadPool(3);
		fixedThreadPool.execute(() -> {
			// 时间检测线程
			int count = 0;
			updateTimeLimit = updateTimeLimit * 1000;
			while (!mainExit) {
				count++;
				if (count >= updateTimeLimit) {
					count = 0;
					new Search().setSearch(true);
				}
				try {
					Thread.sleep(1);
				} catch (InterruptedException ignore) {

				}
			}
		});


		//刷新屏幕线程
		fixedThreadPool.execute(() ->{
			Container panel = searchBar.getPanel();
			while (!mainExit) {
				panel.repaint();
				try {
					Thread.sleep(8);
				} catch (InterruptedException e) {
					//e.printStackTrace();
				}
			}
		});


		//搜索线程
		fixedThreadPool.execute(() ->{
			Search search = new Search();
			while (!mainExit){
				if (search.isSearch()){
					System.out.println("已收到更新请求");
					Search.updateLists(ignorePath, searchDepth);
				}
				try {
					Thread.sleep(16);
				} catch (InterruptedException e) {
					//e.printStackTrace();
				}
			}
		});


		while (true) {
			// 主循环开始
			//System.out.println("isShowSearchBar:"+CheckHotKey.isShowSearachBar);
			if (CheckHotKey.isShowSearachBar){
				CheckHotKey.isShowSearachBar = false;
				searchBar.showSearchbar();
			}
			try {
			Thread.sleep(100);
			} catch (InterruptedException e) {
				//e.printStackTrace();
			}
			if (mainExit){
				fixedThreadPool.shutdown();
				System.exit(0);
			}
		}
	}
}
