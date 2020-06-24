package hotkeyListener;

import DllInterface.HotkeyListener;
import frames.SearchBar;
import frames.SettingsFrame;

import java.awt.event.KeyEvent;
import java.util.HashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

public class CheckHotKey {

    private HashMap<String, Integer> map;
    private ExecutorService threadPool;
    private Pattern plus;

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
        int hotkey1 = -1, hotkey2 = -1, hotkey3 = -1, hotkey4 = -1, hotkey5 = -1;
        String[] hotkeys = plus.split(hotkey);
        int length = hotkeys.length;
        for (int i = 0; i < length - 1; i++) {
            if (i == 0) {
                hotkey1 = map.get(hotkeys[i]);
            } else if (i == 1) {
                hotkey2 = map.get(hotkeys[i]);
            } else if (i == 2) {
                hotkey3 = map.get(hotkeys[i]);
            } else if (i == 4) {
                hotkey4 = map.get(hotkeys[i]);
            }
        }
        hotkey5 = hotkeys[length - 1].charAt(0);
        int finalHotkey = hotkey1;
        int finalHotkey1 = hotkey2;
        int finalHotkey2 = hotkey3;
        int finalHotkey3 = hotkey4;
        int finalHotkey4 = hotkey5;
        threadPool.execute(() -> {
            HotkeyListener.INSTANCE.registerHotKey(finalHotkey, finalHotkey1, finalHotkey2, finalHotkey3, finalHotkey4);
            HotkeyListener.INSTANCE.startListen();
        });
    }

    public boolean isHotkeyAvailable(String hotkey) {
        String[] hotkeys = plus.split(hotkey);
        int length = hotkeys.length;
        for (int i = 0; i < length - 1; i++) {
            String each = hotkeys[i];
            if (!map.containsKey(each)) {
                return false;
            }
        }
        return 64 < hotkey.charAt(hotkey.length() - 1) && hotkey.charAt(hotkey.length() - 1) < 91;
    }

    public void changeHotKey(String hotkey) {
        int hotkey1 = -1, hotkey2 = -1, hotkey3 = -1, hotkey4 = -1, hotkey5 = -1;
        String[] hotkeys = plus.split(hotkey);
        int length = hotkeys.length;
        for (int i = 0; i < length - 1; i++) {
            if (i == 0) {
                hotkey1 = map.get(hotkeys[i]);
            } else if (i == 1) {
                hotkey2 = map.get(hotkeys[i]);
            } else if (i == 2) {
                hotkey3 = map.get(hotkeys[i]);
            } else if (i == 4) {
                hotkey4 = map.get(hotkeys[i]);
            }
        }
        hotkey5 = hotkeys[length - 1].charAt(0);
        HotkeyListener.INSTANCE.registerHotKey(hotkey1, hotkey2, hotkey3, hotkey4, hotkey5);
    }

    private CheckHotKey() {
        plus = Pattern.compile(" \\+ ");
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
                while (SettingsFrame.isNotMainExit()) {
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