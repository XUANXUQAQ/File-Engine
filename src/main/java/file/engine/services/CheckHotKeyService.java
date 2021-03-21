package file.engine.services;

import file.engine.annotation.EventRegister;
import file.engine.configs.Enums;
import file.engine.dllInterface.HotkeyListener;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.frame.searchBar.GetShowingModeEvent;
import file.engine.event.handler.impl.frame.searchBar.HideSearchBarEvent;
import file.engine.event.handler.impl.frame.searchBar.IsSearchBarVisibleEvent;
import file.engine.event.handler.impl.frame.searchBar.ShowSearchBarEvent;
import file.engine.event.handler.impl.hotkey.RegisterHotKeyEvent;
import file.engine.event.handler.impl.hotkey.ResponseCtrlEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.utils.RegexUtil;

import java.awt.event.KeyEvent;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

public class CheckHotKeyService {

    private final HashMap<String, Integer> map;
    private boolean isRegistered = false;
    private static volatile CheckHotKeyService INSTANCE = null;
    private final Pattern plus = RegexUtil.plus;

    public static CheckHotKeyService getInstance() {
        if (INSTANCE == null) {
            synchronized (CheckHotKeyService.class) {
                if (INSTANCE == null) {
                    INSTANCE = new CheckHotKeyService();
                }
            }
        }
        return INSTANCE;
    }

    //关闭对热键的检测，在程序彻底关闭时调用
    private void stopListen() {
        HotkeyListener.INSTANCE.stopListen();
    }

    //注册快捷键
    private void registerHotkey(String hotkey) {
        if (!isRegistered) {
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
            CachedThreadPoolUtil.getInstance().executeTask(() -> {
                HotkeyListener.INSTANCE.registerHotKey(finalHotkey, finalHotkey1, finalHotkey2, finalHotkey3, finalHotkey4);
                HotkeyListener.INSTANCE.startListen();
            });
        } else {
            changeHotKey(hotkey);
        }
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
    private void changeHotKey(String hotkey) {
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

    private void startListenHotkeyThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            boolean isExecuted = true;
            long startVisibleTime = 0;
            long endVisibleTime;
            HotkeyListener instance = HotkeyListener.INSTANCE;
            EventManagement eventManagement = EventManagement.getInstance();
            try {
                endVisibleTime = System.currentTimeMillis();
                //获取快捷键状态，检测是否被按下线程
                while (eventManagement.isNotMainExit()) {
                    if (!isExecuted && instance.getKeyStatus()) {
                        IsSearchBarVisibleEvent isSearchBarVisibleEvent = new IsSearchBarVisibleEvent();
                        eventManagement.putEvent(isSearchBarVisibleEvent);
                        //等待任务执行
                        if (!eventManagement.waitForEvent(isSearchBarVisibleEvent)) {
                            //是否搜索框可见
                            if (isSearchBarVisibleEvent.getReturnValue()) {
                                //搜索框最小可见时间为200ms，必须显示超过200ms后才响应关闭事件，防止闪屏
                                if (System.currentTimeMillis() - startVisibleTime > 200) {
                                    //获取当前显示模式
                                    GetShowingModeEvent getShowingModeEvent = new GetShowingModeEvent();
                                    eventManagement.putEvent(getShowingModeEvent);
                                    if (!eventManagement.waitForEvent(getShowingModeEvent)) {
                                        if (getShowingModeEvent.getReturnValue() ==
                                                Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                                            eventManagement.putEvent(new HideSearchBarEvent());
                                            endVisibleTime = System.currentTimeMillis();
                                        }
                                    }
                                }
                            } else {
                                if (System.currentTimeMillis() - endVisibleTime > 200) {
                                    eventManagement.putEvent(new ShowSearchBarEvent(true));
                                    startVisibleTime = System.currentTimeMillis();
                                }
                            }
                        }
                    }
                    isExecuted = instance.getKeyStatus();
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    @EventRegister
    @SuppressWarnings("unused")
    public static void registerEventHandler() {
        EventManagement eventManagement = EventManagement.getInstance();

        eventManagement.register(RegisterHotKeyEvent.class, event -> getInstance().registerHotkey(((RegisterHotKeyEvent) event).hotkey));

        eventManagement.registerListener(RestartEvent.class, () -> getInstance().stopListen());

        eventManagement.register(ResponseCtrlEvent.class, event -> {
            ResponseCtrlEvent responseCtrlEvent = (ResponseCtrlEvent) event;
            HotkeyListener.INSTANCE.setCtrlDoubleClick(responseCtrlEvent.isResponse);
        });
    }

    private void initThreadPool() {
        startListenHotkeyThread();
    }

    private CheckHotKeyService() {
        map = new HashMap<>();
        map.put("Ctrl", KeyEvent.VK_CONTROL);
        map.put("Alt", KeyEvent.VK_ALT);
        map.put("Shift", KeyEvent.VK_SHIFT);
        map.put("Win", 0x5B);

        initThreadPool();
    }
}

