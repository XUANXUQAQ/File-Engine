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

    public void removeFileFromDatabase(String path) {
        SearchBar searchBar = SearchBar.getInstance();
        int ascII = searchBar.getAscIISum(searchBar.getFileName(path));
        int asciiGroup = ascII / 100;
        String command;
        switch (asciiGroup) {
            case 0:
                command = "DELETE from list0 where PATH=" + "\"" + path + "\";";
                break;
            case 1:
                command = "DELETE from list1 where PATH=" + "\"" + path + "\";";
                break;
            case 2:
                command = "DELETE from list2 where PATH=" + "\"" + path + "\";";
                break;
            case 3:
                command = "DELETE from list3 where PATH=" + "\"" + path + "\";";
                break;
            case 4:
                command = "DELETE from list4 where PATH=" + "\"" + path + "\";";
                break;
            case 5:
                command = "DELETE from list5 where PATH=" + "\"" + path + "\";";
                break;
            case 6:
                command = "DELETE from list6 where PATH=" + "\"" + path + "\";";
                break;
            case 7:
                command = "DELETE from list7 where PATH=" + "\"" + path + "\";";
                break;
            case 8:
                command = "DELETE from list8 where PATH=" + "\"" + path + "\";";
                break;
            case 9:
                command = "DELETE from list9 where PATH=" + "\"" + path + "\";";
                break;
            case 10:
                command = "DELETE from list10 where PATH=" + "\"" + path + "\";";
                break;
            case 11:
                command = "DELETE from list11 where PATH=" + "\"" + path + "\";";
                break;
            case 12:
                command = "DELETE from list12 where PATH=" + "\"" + path + "\";";
                break;
            case 13:
                command = "DELETE from list13 where PATH=" + "\"" + path + "\";";
                break;

            case 14:
                command = "DELETE from list14 where PATH=" + "\"" + path + "\";";
                break;

            case 15:
                command = "DELETE from list15 where PATH=" + "\"" + path + "\";";
                break;

            case 16:
                command = "DELETE from list16 where PATH=" + "\"" + path + "\";";
                break;

            case 17:
                command = "DELETE from list17 where PATH=" + "\"" + path + "\";";
                break;
            case 18:
                command = "DELETE from list18 where PATH=" + "\"" + path + "\";";
                break;
            case 19:
                command = "DELETE from list19 where PATH=" + "\"" + path + "\";";
                break;
            case 20:
                command = "DELETE from list20 where PATH=" + "\"" + path + "\";";
                break;
            case 21:
                command = "DELETE from list21 where PATH=" + "\"" + path + "\";";
                break;

            case 22:
                command = "DELETE from list22 where PATH=" + "\"" + path + "\";";
                break;

            case 23:
                command = "DELETE from list23 where PATH=" + "\"" + path + "\";";
                break;

            case 24:
                command = "DELETE from list24 where PATH=" + "\"" + path + "\";";
                break;

            case 25:
                command = "DELETE from list25 where PATH=" + "\"" + path + "\";";
                break;
            case 26:
                command = "DELETE from list26 where PATH=" + "\"" + path + "\";";
                break;
            case 27:
                command = "DELETE from list27 where PATH=" + "\"" + path + "\";";
                break;
            case 28:
                command = "DELETE from list28 where PATH=" + "\"" + path + "\";";
                break;
            case 29:
                command = "DELETE from list29 where PATH=" + "\"" + path + "\";";
                break;
            case 30:
                command = "DELETE from list30 where PATH=" + "\"" + path + "\";";
                break;
            case 31:
                command = "DELETE from list31 where PATH=" + "\"" + path + "\";";
                break;
            case 32:
                command = "DELETE from list32 where PATH=" + "\"" + path + "\";";
                break;
            case 33:
                command = "DELETE from list33 where PATH=" + "\"" + path + "\";";
                break;
            case 34:
                command = "DELETE from list34 where PATH=" + "\"" + path + "\";";
                break;
            case 35:
                command = "DELETE from list35 where PATH=" + "\"" + path + "\";";
                break;
            case 36:
                command = "DELETE from list36 where PATH=" + "\"" + path + "\";";
                break;
            case 37:
                command = "DELETE from list37 where PATH=" + "\"" + path + "\";";
                break;
            case 38:
                command = "DELETE from list38 where PATH=" + "\"" + path + "\";";
                break;
            case 39:
                command = "DELETE from list39 where PATH=" + "\"" + path + "\";";
                break;
            case 40:
                command = "DELETE from list40 where PATH=" + "\"" + path + "\";";
                break;
            default:
                command = null;
                break;
        }
        if (command != null) {
            commandSet.add(command);
        }
    }

    public void addFileToDatabase(String path) {
        File file = new File(path);
        int ascII = SearchBar.getInstance().getAscIISum(file.getName());
        int asciiGroup = ascII / 100;
        String command;
        switch (asciiGroup) {
            case 0:
                command = "INSERT OR IGNORE INTO list0(PATH) VALUES(\"" + path + "\");";
                break;
            case 1:
                command = "INSERT OR IGNORE INTO list1(PATH) VALUES(\"" + path + "\");";
                break;
            case 2:
                command = "INSERT OR IGNORE INTO list2(PATH) VALUES(\"" + path + "\");";
                break;
            case 3:
                command = "INSERT OR IGNORE INTO list3(PATH) VALUES(\"" + path + "\");";
                break;
            case 4:
                command = "INSERT OR IGNORE INTO list4(PATH) VALUES(\"" + path + "\");";
                break;
            case 5:
                command = "INSERT OR IGNORE INTO list5(PATH) VALUES(\"" + path + "\");";
                break;
            case 6:
                command = "INSERT OR IGNORE INTO list6(PATH) VALUES(\"" + path + "\");";
                break;
            case 7:
                command = "INSERT OR IGNORE INTO list7(PATH) VALUES(\"" + path + "\");";
                break;
            case 8:
                command = "INSERT OR IGNORE INTO list8(PATH) VALUES(\"" + path + "\");";
                break;
            case 9:
                command = "INSERT OR IGNORE INTO list9(PATH) VALUES(\"" + path + "\");";
                break;
            case 10:
                command = "INSERT OR IGNORE INTO list10(PATH) VALUES(\"" + path + "\");";
                break;
            case 11:
                command = "INSERT OR IGNORE INTO list11(PATH) VALUES(\"" + path + "\");";
                break;
            case 12:
                command = "INSERT OR IGNORE INTO list12(PATH) VALUES(\"" + path + "\");";
                break;
            case 13:
                command = "INSERT OR IGNORE INTO list13(PATH) VALUES(\"" + path + "\");";
                break;

            case 14:
                command = "INSERT OR IGNORE INTO list14(PATH) VALUES(\"" + path + "\");";
                break;

            case 15:
                command = "INSERT OR IGNORE INTO list15(PATH) VALUES(\"" + path + "\");";
                break;

            case 16:
                command = "INSERT OR IGNORE INTO list16(PATH) VALUES(\"" + path + "\");";
                break;

            case 17:
                command = "INSERT OR IGNORE INTO list17(PATH) VALUES(\"" + path + "\");";
                break;

            case 18:
                command = "INSERT OR IGNORE INTO list18(PATH) VALUES(\"" + path + "\");";
                break;

            case 19:
                command = "INSERT OR IGNORE INTO list19(PATH) VALUES(\"" + path + "\");";
                break;
            case 20:
                command = "INSERT OR IGNORE INTO list20(PATH) VALUES(\"" + path + "\");";
                break;
            case 21:
                command = "INSERT OR IGNORE INTO list21(PATH) VALUES(\"" + path + "\");";
                break;
            case 22:
                command = "INSERT OR IGNORE INTO list22(PATH) VALUES(\"" + path + "\");";
                break;

            case 23:
                command = "INSERT OR IGNORE INTO list23(PATH) VALUES(\"" + path + "\");";
                break;

            case 24:
                command = "INSERT OR IGNORE INTO list24(PATH) VALUES(\"" + path + "\");";
                break;

            case 25:
                command = "INSERT OR IGNORE INTO list25(PATH) VALUES(\"" + path + "\");";
                break;
            case 26:
                command = "INSERT OR IGNORE INTO list26(PATH) VALUES(\"" + path + "\");";
                break;
            case 27:
                command = "INSERT OR IGNORE INTO list27(PATH) VALUES(\"" + path + "\");";
                break;
            case 28:
                command = "INSERT OR IGNORE INTO list28(PATH) VALUES(\"" + path + "\");";
                break;
            case 29:
                command = "INSERT OR IGNORE INTO list29(PATH) VALUES(\"" + path + "\");";
                break;
            case 30:
                command = "INSERT OR IGNORE INTO list30(PATH) VALUES(\"" + path + "\");";
                break;
            case 31:
                command = "INSERT OR IGNORE INTO list31(PATH) VALUES(\"" + path + "\");";
                break;
            case 32:
                command = "INSERT OR IGNORE INTO list32(PATH) VALUES(\"" + path + "\");";
                break;
            case 33:
                command = "INSERT OR IGNORE INTO list33(PATH) VALUES(\"" + path + "\");";
                break;
            case 34:
                command = "INSERT OR IGNORE INTO list34(PATH) VALUES(\"" + path + "\");";
                break;
            case 35:
                command = "INSERT OR IGNORE INTO list35(PATH) VALUES(\"" + path + "\");";
                break;
            case 36:
                command = "INSERT OR IGNORE INTO list36(PATH) VALUES(\"" + path + "\");";
                break;
            case 37:
                command = "INSERT OR IGNORE INTO list37(PATH) VALUES(\"" + path + "\");";
                break;
            case 38:
                command = "INSERT OR IGNORE INTO list38(PATH) VALUES(\"" + path + "\");";
                break;
            case 39:
                command = "INSERT OR IGNORE INTO list39(PATH) VALUES(\"" + path + "\");";
                break;
            case 40:
                command = "INSERT OR IGNORE INTO list40(PATH) VALUES(\"" + path + "\");";
                break;
            default:
                command = null;
                break;
        }
        if (command != null) {
            commandSet.add(command);
        }
    }

    public void executeAllCommands(Statement stmt) {
        try {
            if (!commandSet.isEmpty()) {
                isUsable = false;
                commandQueue.addAll(commandSet);
                commandSet.clear();
                stmt.execute("BEGIN;");
                for (String each : commandQueue) {
                    stmt.execute(each);
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
            for (int i = 0; i <= 40; i++) {
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
            buffW.write(paths);
            buffW.newLine();
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
            clearAllTablesAndIndex(stmt);
            searchFile(ignorePath, searchDepth);
        } catch (IOException | InterruptedException e) {
            if (SettingsFrame.isDebug()) {
                e.printStackTrace();
            }
        }
    }

    private void clearAllTablesAndIndex(Statement stmt) {
        commandSet.clear();
        //删除所有表和索引
        for (int i = 0; i <= 40; i++) {
            commandSet.add("DROP TABLE IF EXISTS list" + i + ";");
            commandSet.add("DROP INDEX IF EXISTS list" + i + "_index;");
        }
        executeAllCommands(stmt);
        initDatabase();
    }
}