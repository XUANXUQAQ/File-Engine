package hotkeyListener;

import com.sun.jna.Library;
import com.sun.jna.Native;
import main.MainClass;

public interface HotkeyListener extends Library {
    HotkeyListener INSTANCE = Native.load(MainClass.hotkeyListenerDllName, HotkeyListener.class);

    void registerHotKey(int key1, int key2, int key3);

    boolean getKeyStatus();

    void startListen();

    void stopListen();
}
