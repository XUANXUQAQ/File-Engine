package file.engine.services;

import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.Constants;
import file.engine.dllInterface.HotkeyListener;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.frame.searchBar.GetShowingModeEvent;
import file.engine.event.handler.impl.frame.searchBar.HideSearchBarEvent;
import file.engine.event.handler.impl.frame.searchBar.ShowSearchBarEvent;
import file.engine.event.handler.impl.hotkey.CheckHotKeyAvailableEvent;
import file.engine.event.handler.impl.hotkey.RegisterHotKeyEvent;
import file.engine.event.handler.impl.hotkey.ResponseCtrlEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.frames.SearchBar;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.utils.RegexUtil;

import java.awt.event.KeyEvent;
import java.util.HashMap;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

public class CheckHotKeyService {

    private final HashMap<String, Integer> map;
    private boolean isRegistered = false;
    private static volatile CheckHotKeyService INSTANCE = null;

    private static CheckHotKeyService getInstance() {
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
            //noinspection DuplicatedCode
            int hotkey1 = -1, hotkey2 = -1, hotkey3 = -1, hotkey4 = -1, hotkey5;
            String[] hotkeys = RegexUtil.plus.split(hotkey);
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
    private boolean isHotkeyAvailable(String hotkey) {
        String[] hotkeys = RegexUtil.plus.split(hotkey);
        int length = hotkeys.length;
        for (int i = 0; i < length - 1; i++) {
            String each = hotkeys[i];
            if (!map.containsKey(each)) {
                return false;
            }
        }
        return 'A' <= hotkey.charAt(hotkey.length() - 1) && hotkey.charAt(hotkey.length() - 1) <= 'Z';
    }

    //更改快捷键,必须在register后才可用
    private void changeHotKey(String hotkey) {
        if (!isRegistered) {
            throw new NullPointerException();
        }
        //noinspection DuplicatedCode
        int hotkey1 = -1, hotkey2 = -1, hotkey3 = -1, hotkey4 = -1, hotkey5;
        String[] hotkeys = RegexUtil.plus.split(hotkey);
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
            var ref = new Object() {
                long startVisibleTime = 0;
                long endVisibleTime;
            };
            HotkeyListener instance = HotkeyListener.INSTANCE;
            EventManagement eventManagement = EventManagement.getInstance();
            try {
                ref.endVisibleTime = System.currentTimeMillis();
                //获取快捷键状态，检测是否被按下线程
                SearchBar searchBar = SearchBar.getInstance();
                while (eventManagement.notMainExit()) {
                    if (!isExecuted && instance.getKeyStatus()) {
                        //是否搜索框可见
                        if (searchBar.isVisible()) {
                            //搜索框最小可见时间为200ms，必须显示超过200ms后才响应关闭事件，防止闪屏
                            if (System.currentTimeMillis() - ref.startVisibleTime > 200) {
                                //获取当前显示模式
                                eventManagement.putEvent(new GetShowingModeEvent(), getShowingModeEvent -> {
                                    Optional<Constants.Enums.ShowingSearchBarMode> returnValue = getShowingModeEvent.getReturnValue();
                                    //noinspection OptionalGetWithoutIsPresent
                                    if (returnValue.get() == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                                        eventManagement.putEvent(new HideSearchBarEvent());
                                        ref.endVisibleTime = System.currentTimeMillis();
                                    } else {
                                        eventManagement.putEvent(new ShowSearchBarEvent(true, true));
                                        ref.startVisibleTime = System.currentTimeMillis();
                                    }
                                }, event1 -> System.err.println("获取当前显示模式任务执行失败"));
                            }
                        } else {
                            if (System.currentTimeMillis() - ref.endVisibleTime > 200) {
                                eventManagement.putEvent(new ShowSearchBarEvent(true));
                                ref.startVisibleTime = System.currentTimeMillis();
                            }
                        }
                    }
                    isExecuted = instance.getKeyStatus();
                    TimeUnit.MILLISECONDS.sleep(10);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    @EventRegister(registerClass = CheckHotKeyAvailableEvent.class)
    private static void checkHotKeyAvailableEvent(Event event) {
        CheckHotKeyAvailableEvent event1 = (CheckHotKeyAvailableEvent) event;
        event1.setReturnValue(getInstance().isHotkeyAvailable(event1.hotkey));
    }

    @EventRegister(registerClass = RegisterHotKeyEvent.class)
    private static void registerHotKeyEvent(Event event) {
        getInstance().registerHotkey(((RegisterHotKeyEvent) event).hotkey);
    }

    @EventRegister(registerClass = ResponseCtrlEvent.class)
    private static void responseCtrlEvent(Event event) {
        ResponseCtrlEvent responseCtrlEvent = (ResponseCtrlEvent) event;
        HotkeyListener.INSTANCE.setCtrlDoubleClick(responseCtrlEvent.isResponse);
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void restartEvent(Event event) {
        getInstance().stopListen();
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

