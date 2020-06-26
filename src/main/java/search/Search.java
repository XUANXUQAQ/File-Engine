package search;

import DllInterface.IsLocalDisk;
import DllInterface.isNTFS;
import frames.SearchBar;
import frames.SettingsFrame;
import frames.TaskBar;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;


public class Search {
    private volatile static boolean isUsable = true;
    private static volatile boolean isManualUpdate = false;
    private Set<String> commandSet = ConcurrentHashMap.newKeySet();
    private ConcurrentLinkedQueue<String> commandQueue = new ConcurrentLinkedQueue<>();

    private static class SearchBuilder {
        private static Search instance = new Search();
    }

    private Search() {
    }

    public static Search getInstance() {
        return SearchBuilder.instance;
    }

    public void addToRecycleBin(String path) {
        SearchBar searchBar = SearchBar.getInstance();
        int ascII = searchBar.getAscIISum(searchBar.getFileName(path));
        String command;
        if (0 <= ascII && ascII <= 100) {
            command = "DELETE from list0 where PATH=" + "\"" + path + "\";";
        } else if (100 < ascII && ascII <= 200) {
            command = "DELETE from list1 where PATH=" + "\"" + path + "\";";
        } else if (200 < ascII && ascII <= 300) {
            command = "DELETE from list2 where PATH=" + "\"" + path + "\";";
        } else if (300 < ascII && ascII <= 400) {
            command = "DELETE from list3 where PATH=" + "\"" + path + "\";";
        } else if (400 < ascII && ascII <= 500) {
            command = "DELETE from list4 where PATH=" + "\"" + path + "\";";
        } else if (500 < ascII && ascII <= 600) {
            command = "DELETE from list5 where PATH=" + "\"" + path + "\";";
        } else if (600 < ascII && ascII <= 700) {
            command = "DELETE from list6 where PATH=" + "\"" + path + "\";";
        } else if (700 < ascII && ascII <= 800) {
            command = "DELETE from list7 where PATH=" + "\"" + path + "\";";
        } else if (800 < ascII && ascII <= 900) {
            command = "DELETE from list8 where PATH=" + "\"" + path + "\";";
        } else if (900 < ascII && ascII <= 1000) {
            command = "DELETE from list9 where PATH=" + "\"" + path + "\";";
        } else if (1000 < ascII && ascII <= 1100) {
            command = "DELETE from list10 where PATH=" + "\"" + path + "\";";
        } else if (1100 < ascII && ascII <= 1200) {
            command = "DELETE from list11 where PATH=" + "\"" + path + "\";";
        } else if (1200 < ascII && ascII <= 1300) {
            command = "DELETE from list12 where PATH=" + "\"" + path + "\";";
        } else if (1300 < ascII && ascII <= 1400) {
            command = "DELETE from list13 where PATH=" + "\"" + path + "\";";
        } else if (1400 < ascII && ascII <= 1500) {
            command = "DELETE from list14 where PATH=" + "\"" + path + "\";";
        } else if (1500 < ascII && ascII <= 1600) {
            command = "DELETE from list15 where PATH=" + "\"" + path + "\";";
        } else if (1600 < ascII && ascII <= 1700) {
            command = "DELETE from list16 where PATH=" + "\"" + path + "\";";
        } else if (1700 < ascII && ascII <= 1800) {
            command = "DELETE from list17 where PATH=" + "\"" + path + "\";";
        } else if (1800 < ascII && ascII <= 1900) {
            command = "DELETE from list18 where PATH=" + "\"" + path + "\";";
        } else if (1900 < ascII && ascII <= 2000) {
            command = "DELETE from list19 where PATH=" + "\"" + path + "\";";
        } else if (2000 < ascII && ascII <= 2100) {
            command = "DELETE from list20 where PATH=" + "\"" + path + "\";";
        } else if (2100 < ascII && ascII <= 2200) {
            command = "DELETE from list21 where PATH=" + "\"" + path + "\";";
        } else if (2200 < ascII && ascII <= 2300) {
            command = "DELETE from list22 where PATH=" + "\"" + path + "\";";
        } else if (2300 < ascII && ascII <= 2400) {
            command = "DELETE from list23 where PATH=" + "\"" + path + "\";";
        } else if (2400 < ascII && ascII <= 2500) {
            command = "DELETE from list24 where PATH=" + "\"" + path + "\";";
        } else {
            command = "DELETE from list25 where PATH=" + "\"" + path + "\";";
        }
        commandSet.add(command);
    }

    public void addFileToLoadBin(String path) {
        File file = new File(path);
        int ascII = SearchBar.getInstance().getAscIISum(file.getName());
        String command;
        if (0 < ascII && ascII <= 100) {
            command = "INSERT OR IGNORE INTO list0(PATH) VALUES(\"" + path + "\");";
        } else if (100 < ascII && ascII <= 200) {
            command = "INSERT OR IGNORE INTO list1(PATH) VALUES(\"" + path + "\");";
        } else if (200 < ascII && ascII <= 300) {
            command = "INSERT OR IGNORE INTO list2(PATH) VALUES(\"" + path + "\");";
        } else if (300 < ascII && ascII <= 400) {
            command = "INSERT OR IGNORE INTO list3(PATH) VALUES(\"" + path + "\");";
        } else if (400 < ascII && ascII <= 500) {
            command = "INSERT OR IGNORE INTO list4(PATH) VALUES(\"" + path + "\");";
        } else if (500 < ascII && ascII <= 600) {
            command = "INSERT OR IGNORE INTO list5(PATH) VALUES(\"" + path + "\");";
        } else if (600 < ascII && ascII <= 700) {
            command = "INSERT OR IGNORE INTO list6(PATH) VALUES(\"" + path + "\");";
        } else if (700 < ascII && ascII <= 800) {
            command = "INSERT OR IGNORE INTO list7(PATH) VALUES(\"" + path + "\");";
        } else if (800 < ascII && ascII <= 900) {
            command = "INSERT OR IGNORE INTO list8(PATH) VALUES(\"" + path + "\");";
        } else if (900 < ascII && ascII <= 1000) {
            command = "INSERT OR IGNORE INTO list9(PATH) VALUES(\"" + path + "\");";
        } else if (1000 < ascII && ascII <= 1100) {
            command = "INSERT OR IGNORE INTO list10(PATH) VALUES(\"" + path + "\");";
        } else if (1100 < ascII && ascII <= 1200) {
            command = "INSERT OR IGNORE INTO list11(PATH) VALUES(\"" + path + "\");";
        } else if (1200 < ascII && ascII <= 1300) {
            command = "INSERT OR IGNORE INTO list12(PATH) VALUES(\"" + path + "\");";
        } else if (1300 < ascII && ascII <= 1400) {
            command = "INSERT OR IGNORE INTO list13(PATH) VALUES(\"" + path + "\");";
        } else if (1400 < ascII && ascII <= 1500) {
            command = "INSERT OR IGNORE INTO list14(PATH) VALUES(\"" + path + "\");";
        } else if (1500 < ascII && ascII <= 1600) {
            command = "INSERT OR IGNORE INTO list15(PATH) VALUES(\"" + path + "\");";
        } else if (1600 < ascII && ascII <= 1700) {
            command = "INSERT OR IGNORE INTO list16(PATH) VALUES(\"" + path + "\");";
        } else if (1700 < ascII && ascII <= 1800) {
            command = "INSERT OR IGNORE INTO list17(PATH) VALUES(\"" + path + "\");";
        } else if (1800 < ascII && ascII <= 1900) {
            command = "INSERT OR IGNORE INTO list18(PATH) VALUES(\"" + path + "\");";
        } else if (1900 < ascII && ascII <= 2000) {
            command = "INSERT OR IGNORE INTO list19(PATH) VALUES(\"" + path + "\");";
        } else if (2000 < ascII && ascII <= 2100) {
            command = "INSERT OR IGNORE INTO list20(PATH) VALUES(\"" + path + "\");";
        } else if (2100 < ascII && ascII <= 2200) {
            command = "INSERT OR IGNORE INTO list21(PATH) VALUES(\"" + path + "\");";
        } else if (2200 < ascII && ascII <= 2300) {
            command = "INSERT OR IGNORE INTO list22(PATH) VALUES(\"" + path + "\");";
        } else if (2300 < ascII && ascII <= 2400) {
            command = "INSERT OR IGNORE INTO list23(PATH) VALUES(\"" + path + "\");";
        } else if (2400 < ascII && ascII <= 2500) {
            command = "INSERT OR IGNORE INTO list24(PATH) VALUES(\"" + path + "\");";
        } else {
            command = "INSERT OR IGNORE INTO list25(PATH) VALUES(\"" + path + "\");";
        }
        commandSet.add(command);
    }

    public void executeAllCommands(Statement stmt) {
        try {
            if (!commandSet.isEmpty()) {
                isUsable = false;
                commandQueue.addAll(commandSet);
                commandSet.clear();
                stmt.execute("BEGIN;");
                for (String each : commandQueue) {
                    stmt.executeUpdate(each);
                }
                stmt.execute("COMMIT;");
                commandQueue.clear();
                isUsable = true;
            }
        } catch (SQLException ignored) {

        }
    }

    public static void initDatabase() {
        Statement stmt;
        String sql = "CREATE TABLE list";
        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:data.db")) {
            if (SettingsFrame.isDebug()) {
                System.out.println("open database successfully");
            }
            stmt = conn.createStatement();
            stmt.execute("BEGIN;");
            for (int i = 0; i < 26; i++) {
                String command = sql + i + " " + "(PATH text unique)" + ";";
                stmt.executeUpdate(command);
            }
            stmt.execute("COMMIT;");
            stmt.execute("PRAGMA SQLITE_THREADSAFE=2;");
            stmt.execute("PRAGMA SQLITE_TEMP_STORE=2;");
            stmt.execute("PRAGMA journal_mode=WAL;");
            stmt.execute("PRAGMA synchronous=OFF;");
            stmt.execute("PRAGMA page_size=4096;");
            stmt.execute("PRAGMA cache_size=8000;");
            stmt.execute("PRAGMA auto_vacuum=0;");
            stmt.execute("PRAGMA mmap_size=4096;");
        } catch (Exception e) {
            if (SettingsFrame.isDebug()) {
                e.printStackTrace();
            }
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


    private void searchFile(String ignorePath, int searchDepth) throws IOException, InterruptedException {
        boolean needSearchIgnoreSearchDepth = true;
        File[] roots = File.listRoots();
        StringBuilder strb = new StringBuilder(26);
        for (File root : roots) {
            if (IsLocalDisk.INSTANCE.isLocalDisk(root.getAbsolutePath())) {
                if (isNTFS.INSTANCE.isDiskNTFS(root.getAbsolutePath())) {
                    needSearchIgnoreSearchDepth = false;
                    strb.append(root.getAbsolutePath()).append(",");
                } else {
                    String path = root.getAbsolutePath();
                    path = path.substring(0, 2);
                    __searchFile(path, searchDepth, ignorePath);
                }
            }
        }
        if (needSearchIgnoreSearchDepth) {
            __searchFileIgnoreSearchDepth(getStartMenu(), ignorePath);
            __searchFileIgnoreSearchDepth("C:\\ProgramData\\Microsoft\\Windows\\Start Menu", ignorePath);
        } else {
            searchByUSN(strb.toString());
        }
        TaskBar.getInstance().showMessage(SettingsFrame.getTranslation("Info"), SettingsFrame.getTranslation("Search Done"));
        isManualUpdate = false;
        isUsable = true;
    }

    private void searchByUSN(String paths) throws IOException, InterruptedException {
        File fileSearcherUSN = new File("user/fileSearcherUSN.exe");
        String absPath = fileSearcherUSN.getAbsolutePath();
        String start = absPath.substring(0, 2);
        String end = "\"" + absPath.substring(2) + "\"";
        File database = new File("data.db");
        try (BufferedWriter buffW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("user/MFTSearchInfo.dat"), StandardCharsets.UTF_8))) {
            buffW.write(paths + "\r\n");
            buffW.write(database.getAbsolutePath());
        }
        String command = "cmd.exe /c " + start + end;
        Process p = Runtime.getRuntime().exec(command, null, new File("user"));
        p.waitFor();
    }

    private String getStartMenu() {
        try {
            String startMenu;
            BufferedReader bufrIn;
            Process getStartMenu = Runtime.getRuntime().exec("reg query \"HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders\" " + "/v " + "\"Start Menu\"");
            bufrIn = new BufferedReader(new InputStreamReader(getStartMenu.getInputStream(), StandardCharsets.UTF_8));
            while ((startMenu = bufrIn.readLine()) != null) {
                if (startMenu.contains("REG_SZ")) {
                    startMenu = startMenu.substring(startMenu.indexOf("REG_SZ") + 10);
                    return startMenu;
                }
            }
        } catch (IOException ignored) {

        }
        return null;
    }

    private void __searchFileIgnoreSearchDepth(String path, String ignorePath) {
        File fileSearcher = new File("user/fileSearcher.exe");
        String absPath = fileSearcher.getAbsolutePath();
        String start = absPath.substring(0, 2);
        String end = "\"" + absPath.substring(2) + "\"";
        File database = new File("data.db");
        String command = "cmd.exe /c " + start + end + " \"" + path + "\"" + " \"1\" " + "\"" + ignorePath + "\" " + "\"" + database.getAbsolutePath() + "\" " + "\"" + "1" + "\"";
        try {
            Process p = Runtime.getRuntime().exec(command);
            p.waitFor();
        } catch (IOException | InterruptedException ignored) {

        }
    }

    private void __searchFile(String path, int searchDepth, String ignorePath) throws InterruptedException, IOException {
        File fileSearcher = new File("user/fileSearcher.exe");
        String absPath = fileSearcher.getAbsolutePath();
        String start = absPath.substring(0, 2);
        String end = "\"" + absPath.substring(2) + "\"";
        File database = new File("data.db");
        String command = "cmd.exe /c " + start + end + " \"" + path + "\"" + " \"" + searchDepth + "\" " + "\"" + ignorePath + "\" " + "\"" + database.getAbsolutePath() + "\" " + "\"" + "0" + "\"";
        Process p = Runtime.getRuntime().exec(command);
        p.waitFor();
    }

    public void updateLists(String ignorePath, int searchDepth, Statement stmt) {
        try {
            TaskBar.getInstance().showMessage(SettingsFrame.getTranslation("Info"), SettingsFrame.getTranslation("Updating file index"));
            clearAllTables(stmt);
            searchFile(ignorePath, searchDepth);
        } catch (IOException | InterruptedException e) {
            if (SettingsFrame.isDebug()) {
                e.printStackTrace();
            }
        }
    }

    private void clearAllTables(Statement stmt) {
        commandSet.clear();
        //删除所有表
        for (int i = 0; i < 26; i++) {
            commandSet.add("DROP TABLE IF EXISTS list" + i + ";");
        }
        executeAllCommands(stmt);
        initDatabase();
    }
}