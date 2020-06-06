package hotkeyListener;

import DllInterface.HotkeyListener;
import frames.SearchBar;
import frames.SettingsFrame;

import java.awt.event.KeyEvent;
import java.util.HashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CheckHotKey {

    private HashMap<String, Integer> map;
    private ExecutorService threadPool;

    private static class CheckHotKeyBuilder {
        private static CheckHotKey instance = new CheckHotKey();
    }

    public static CheckHotKey getInstance() {
        return CheckHotKeyBuilder.instance;
    }


    public void stopListen() {
        HotkeyListener.INSTANCE.stopListen();
    }

    public void registerHotkey(String hotkey) {
        int hotkey1 = -1, hotkey2 = -1, hotkey3 = -1;
        int count = 0;
        String[] hotkeys = hotkey.split(" \\+ ");
        for (String each : hotkeys) {
            if (each.length() != 1) {
                if (count == 0) {
                    hotkey1 = map.get(each);
                } else if (count == 1) {
                    hotkey2 = map.get(each);
                }
                count++;
            } else {
                hotkey3 = each.charAt(0);
            }
        }
        int finalHotkey = hotkey1;
        int finalHotkey1 = hotkey2;
        int finalHotkey2 = hotkey3;
        threadPool.execute(() -> {
            HotkeyListener.INSTANCE.registerHotKey(finalHotkey, finalHotkey1, finalHotkey2);
            HotkeyListener.INSTANCE.startListen();
        });
    }

    public void changeHotKey(String hotkey) {
        int hotkey1 = -1, hotkey2 = -1, hotkey3 = -1;
        int count = 0;
        String[] hotkeys = hotkey.split(" \\+ ");
        for (String each : hotkeys) {
            if (each.length() != 1) {
                if (count == 0) {
                    hotkey1 = map.get(each);
                } else if (count == 1) {
                    hotkey2 = map.get(each);
                }
                count++;
            } else {
                hotkey3 = each.charAt(0);
            }
        }
        HotkeyListener.INSTANCE.registerHotKey(hotkey1, hotkey2, hotkey3);
    }

    private CheckHotKey() {
        map = new HashMap<>();

        threadPool = Executors.newFixedThreadPool(2);
        map.put("Ctrl", KeyEvent.VK_CONTROL);
        map.put("Alt", KeyEvent.VK_ALT);
        map.put("Shift", KeyEvent.VK_SHIFT);
        map.put("Win", 0x5B);

        threadPool.execute(() -> {
            boolean isExecuted = false;
            SearchBar searchBar = SearchBar.getInstance();
            HotkeyListener instance = HotkeyListener.INSTANCE;
            try {
                while (!SettingsFrame.mainExit) {
                    if (!isExecuted && instance.getKeyStatus()) {
                        isExecuted = true;
                        if (!searchBar.isVisible()) {
                            searchBar.showSearchbar();
                        } else {
                            searchBar.closedTodo();
                        }
                    }
                    if (!instance.getKeyStatus()) {
                        isExecuted = false;
                    }
                    Thread.sleep(10);
                }
            } catch (InterruptedException ignored) {

            }
        });
    }
}