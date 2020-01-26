package main;
import search.*;

import java.awt.*;
import java.io.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import frame.*;
import com.alibaba.fastjson.*;

import javax.swing.*;


public class Main {
	private static int updateTimeLimit = 600;
	private static boolean mainExit = false;
	private static String ignorePath;
	private static int searchDepth;
	private static SearchBar searchBar;
	private static Search search = new Search();
	
	
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
			} catch (IOException e) {
				JOptionPane.showMessageDialog(null, "创建缓存文件失败，程序正在退出");
				mainExit = true;
			}

		}
		if (!files.exists()){
			files.mkdir();
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

		CheckHotKey HotKeyListener = new CheckHotKey();
		SettingsFrame.initCacheLimit();

		search.setSearch(true);



		ExecutorService fixedThreadPool = Executors.newFixedThreadPool(4);
		fixedThreadPool.execute(() -> {
			// 时间检测线程 兼 内存释放线程
			int count = 0;
			updateTimeLimit = updateTimeLimit * 1000;
			while (!mainExit) {
				count++;
				if (count >= updateTimeLimit) {
					count = 0;
					System.out.println("正在发送更新请求并清理内存空间");
					System.gc();
					search.setSearch(true);
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
				if (search.isSearch()){
					System.out.println("已收到更新请求");
					search.updateLists(ignorePath, searchDepth);
				}
				try {
					Thread.sleep(16);
				} catch (InterruptedException e) {
					//e.printStackTrace();
				}
			}
		});

		fixedThreadPool.execute(()->{
			//垃圾回收线程
			int count = 1;
			while (!mainExit){
				if (search.isIsFocusLost() && count == 0){
					count+=1;
					System.out.println("检测到窗口失去焦点，进行垃圾回收");
					System.gc();
				}else if (!search.isIsFocusLost()){
					count = 0;
				}
				try {
					Thread.sleep(20);
				} catch (InterruptedException ignored) {

				}
			}
		});



		do {
			// 主循环开始
			if (!search.isIsFocusLost()){
				if (searchBar == null) {
					searchBar = new SearchBar();
					searchBar.showSearchbar();
				}
			}else{
				if (searchBar != null) {
					searchBar = null;
				}
			}
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				//e.printStackTrace();
			}
			if (mainExit){
				fixedThreadPool.shutdownNow();
				search.clearAll();
			}
		}while(!mainExit);
	}
}
