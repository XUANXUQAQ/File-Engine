package file.engine.dllInterface;

import java.nio.file.Path;

public enum HotkeyListener {
    INSTANCE;

    static {
        System.load(Path.of("user/hotkeyListener.dll").toAbsolutePath().toString());
    }

    public native void registerHotKey(int key1, int key2, int key3, int key4, int key5);

    public native boolean getKeyStatus();

    public native void startListen();

    public native void stopListen();

    public native boolean isCtrlDoubleClicked();

    public native boolean isShiftDoubleClicked();
}