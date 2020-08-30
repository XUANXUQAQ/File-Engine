package FileEngine.checkHotkey;

import FileEngine.configs.AllConfigs;
import FileEngine.dllInterface.HotkeyListener;
import FileEngine.frames.SearchBar;
import FileEngine.threadPool.CachedThreadPool;

import java.awt.event.KeyEvent;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

public class CheckHotKeyUtil {

    private final HashMap<String, Integer> map;
    private final Pattern plus;
    private boolean isRegistered = false;

    private static class CheckHotKeyBuilder {
        private static final CheckHotKeyUtil INSTANCE = new CheckHotKeyUtil();
    }

    public static CheckHotKeyUtil getInstance() {
        return CheckHotKeyBuilder.INSTANCE;
    }

    //关闭对热键的检测，在程序彻底关闭时调用
    public void stopListen() {
        HotkeyListener.INSTANCE.stopListen();
    }

    //注册快捷键
    public void registerHotkey(String hotkey) {
        isRegistered = true;
        int hotkey1 = -1, hotkey2 = -1, hotkey3 = -1, hotkey4 = -1, hotkey5;
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
        CachedThreadPool.getInstance().executeTask(() -> {
            HotkeyListener.INSTANCE.registerHotKey(finalHotkey, finalHotkey1, finalHotkey2, finalHotkey3, finalHotkey4);
            HotkeyListener.INSTANCE.startListen();
        });
    }

    //检查快捷键是否有效
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

    //更改快捷键,必须在register后才可用
    public void changeHotKey(String hotkey) {
        if (!isRegistered) {
            throw new NullPointerException();
        }
        int hotkey1 = -1, hotkey2 = -1, hotkey3 = -1, hotkey4 = -1, hotkey5;
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

    private CheckHotKeyUtil() {
        plus = Pattern.compile(" \\+ ");
        map = new HashMap<>();
        map.put("Ctrl", KeyEvent.VK_CONTROL);
        map.put("Alt", KeyEvent.VK_ALT);
        map.put("Shift", KeyEvent.VK_SHIFT);
        map.put("Win", 0x5B);

        CachedThreadPool.getInstance().executeTask(() -> {
            boolean isExecuted = false;
            long startVisibleTime = 0;
            long endVisibleTime = 0;
            SearchBar searchBar = SearchBar.getInstance();
            HotkeyListener instance = HotkeyListener.INSTANCE;
            try {
                //获取快捷键状态，检测是否被按下线程
                while (AllConfigs.isNotMainExit()) {
                    if (!isExecuted && instance.getKeyStatus()) {
                        isExecuted = true;
                        if (!searchBar.isVisible()) {
                            if (System.currentTimeMillis() - endVisibleTime > 200) {
                                searchBar.showSearchbar(true);
                                startVisibleTime = System.currentTimeMillis();
                            }
                        } else {
                            if (System.currentTimeMillis() - startVisibleTime > 200) {
                                if (searchBar.getShowingMode() == AllConfigs.ShowingSearchBarMode.NORMAL_SHOWING) {
                                    searchBar.closeSearchBar();
                                    endVisibleTime = System.currentTimeMillis();
                                }
                            }
                        }
                    }
                    if (!instance.getKeyStatus()) {
                        isExecuted = false;
                    }
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }
}