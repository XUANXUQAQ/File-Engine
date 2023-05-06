package file.engine.dllInterface;

import java.nio.file.Path;

public enum HotkeyListener {
    INSTANCE;

    static {
        System.load(Path.of("user/hotkeyListener.dll").toAbsolutePath().toString());
    }

    /**
     * 注册全局快捷键
     *
     * @param key1 key1
     * @param key2 key2
     * @param key3 key3
     * @param key4 key4
     * @param key5 key5
     */
    public native void registerHotKey(int key1, int key2, int key3, int key4, int key5);

    /**
     * 获取快捷键点击状态
     *
     * @return 当所有快捷键都被点击且没有多点击其他按键的情况下返回true，否则返回false
     */
    public native boolean getKeyStatus();

    /**
     * 开始监听快捷键
     */
    public native void startListen();

    /**
     * 停止监听快捷键
     */
    public native void stopListen();

    /**
     * 是否ctrl被双击
     *
     * @return true如果被双击
     */
    public native boolean isCtrlDoubleClicked();

    /**
     * 是否shift被双击
     *
     * @return true如果被双击
     */
    public native boolean isShiftDoubleClicked();
}