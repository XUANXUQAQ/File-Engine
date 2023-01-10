package file.engine.services;

import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.dllInterface.HotkeyListener;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.frame.searchBar.*;
import file.engine.event.handler.impl.hotkey.CheckHotKeyAvailableEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.utils.RegexUtil;
import lombok.Getter;

import java.awt.event.KeyEvent;
import java.util.HashMap;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

public class CheckHotKeyService {

    private final HashMap<String, Integer> map;
    private boolean isRegistered = false;
    private static volatile CheckHotKeyService INSTANCE = null;
    @Getter
    private static volatile String currentHotkey;
    private static boolean isSearchBarVisible = false;

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

    private int[] parseHotkey(String hotkey) {
        int[] hotkeys = new int[]{
                -1, -1, -1, -1, -1
        };
        String[] hotkeyString = RegexUtil.plus.split(hotkey);
        for (int i = 0; i < hotkeyString.length; ++i) {
            String eachHotkey = hotkeyString[i].trim();
            hotkeys[i] = map.get(eachHotkey);
        }
        return hotkeys;
    }

    //注册快捷键
    private void registerHotkey(String hotkey) {
        currentHotkey = hotkey;
        if (!isRegistered) {
            isRegistered = true;
            int[] hotkeyIds = parseHotkey(hotkey);
            CachedThreadPoolUtil.getInstance().executeTask(() -> {
                HotkeyListener.INSTANCE.registerHotKey(hotkeyIds[0], hotkeyIds[1], hotkeyIds[2], hotkeyIds[3], hotkeyIds[4]);
                HotkeyListener.INSTANCE.startListen();
            });
        } else {
            changeHotKey(hotkey);
        }
    }

    //检查快捷键是否有效
    private boolean isHotkeyAvailable(String hotkey) {
        String[] hotkeys = RegexUtil.plus.split(hotkey);
        final int length = hotkeys.length;
        if (length > 5) {
            return false;
        }
        for (int i = 0; i < length - 1; i++) {
            String each = hotkeys[i].trim();
            if (!map.containsKey(each)) {
                return false;
            }
        }
        return true;
    }

    //更改快捷键,必须在register后才可用
    private void changeHotKey(String hotkey) {
        if (!isRegistered) {
            throw new RuntimeException("should not reach here. Call registerHotkey() first");
        }
        int[] hotkeyIds = parseHotkey(hotkey);
        HotkeyListener.INSTANCE.registerHotKey(hotkeyIds[0], hotkeyIds[1], hotkeyIds[2], hotkeyIds[3], hotkeyIds[4]);
    }

    private void startListenHotkeyThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            AtomicLong startVisibleTime = new AtomicLong();
            AtomicLong endVisibleTime = new AtomicLong();
            boolean isExecuted = true;
            endVisibleTime.set(System.currentTimeMillis());
            //获取快捷键状态，检测是否被按下线程
            long grabFocusOnAttachModeTime = 0;
            while (eventManagement.notMainExit()) {
                if (!isExecuted && shouldOpenSearchBar()) {
                    //是否搜索框可见
                    if (isSearchBarVisible) {
                        //搜索框最小可见时间为200ms，必须显示超过200ms后才响应关闭事件，防止闪屏
                        if (System.currentTimeMillis() - startVisibleTime.get() > 200) {
                            //获取当前显示模式
                            eventManagement.putEvent(new GetShowingModeEvent(), getShowingModeEvent -> {
                                Optional<Constants.Enums.ShowingSearchBarMode> returnValue = getShowingModeEvent.getReturnValue();
                                //noinspection OptionalGetWithoutIsPresent
                                if (returnValue.get() == Constants.Enums.ShowingSearchBarMode.NORMAL_SHOWING) {
                                    eventManagement.putEvent(new HideSearchBarEvent());
                                    endVisibleTime.set(System.currentTimeMillis());
                                } else {
                                    eventManagement.putEvent(new ShowSearchBarEvent(true, true));
                                    startVisibleTime.set(System.currentTimeMillis());
                                }
                            }, event1 -> System.err.println("获取当前显示模式任务执行失败"));
                        }
                    } else {
                        if (System.currentTimeMillis() - endVisibleTime.get() > 200) {
                            eventManagement.putEvent(new ShowSearchBarEvent(true));
                            startVisibleTime.set(System.currentTimeMillis());
                        }
                    }
                }
                isExecuted = shouldOpenSearchBar();
                if (HotkeyListener.INSTANCE.isShiftDoubleClicked() && System.currentTimeMillis() - grabFocusOnAttachModeTime > 300) {
                    eventManagement.putEvent(new GrabFocusOnAttachModeEvent());
                    grabFocusOnAttachModeTime = System.currentTimeMillis();
                }
                try {
                    TimeUnit.MILLISECONDS.sleep(10);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        });
    }

    private boolean shouldOpenSearchBar() {
        var allConfigs = AllConfigs.getInstance();
        if (allConfigs.getConfigEntity().isDoubleClickCtrlOpen()) {
            return HotkeyListener.INSTANCE.getKeyStatus() || HotkeyListener.INSTANCE.isCtrlDoubleClicked();
        }
        return HotkeyListener.INSTANCE.getKeyStatus();
    }

    @EventRegister(registerClass = CheckHotKeyAvailableEvent.class)
    private static void checkHotKeyAvailableEvent(Event event) {
        CheckHotKeyAvailableEvent event1 = (CheckHotKeyAvailableEvent) event;
        event1.setReturnValue(getInstance().isHotkeyAvailable(event1.hotkey));
    }

    @EventListener(listenClass = SetConfigsEvent.class)
    private static void registerHotKeyEvent(Event event) {
        getInstance().registerHotkey(AllConfigs.getInstance().getConfigEntity().getHotkey());
    }

    @EventListener(listenClass = {SearchBarCloseEvent.class, SearchBarReadyEvent.class})
    private static void setSearchBarVisibleStatus(Event event) {
        isSearchBarVisible = event instanceof SearchBarReadyEvent;
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void stopListen(Event event) {
        getInstance().stopListen();
    }

    private CheckHotKeyService() {
        map = new HashMap<>() {{
            put("Ctrl", KeyEvent.VK_CONTROL);
            put("Alt", KeyEvent.VK_ALT);
            put("Shift", KeyEvent.VK_SHIFT);
            put("Win", 0x5B);
            put("F1", KeyEvent.VK_F1);
            put("F2", KeyEvent.VK_F2);
            put("F3", KeyEvent.VK_F3);
            put("F4", KeyEvent.VK_F4);
            put("F5", KeyEvent.VK_F5);
            put("F6", KeyEvent.VK_F6);
            put("F7", KeyEvent.VK_F7);
            put("F8", KeyEvent.VK_F8);
            put("F9", KeyEvent.VK_F9);
            put("F10", KeyEvent.VK_F10);
            put("F11", KeyEvent.VK_F11);
            put("F12", KeyEvent.VK_F12);
        }};
        for (int i = 'A'; i <= 'Z'; i++) {
            map.put(String.valueOf((char) i), i);
        }
        startListenHotkeyThread();
    }
}

