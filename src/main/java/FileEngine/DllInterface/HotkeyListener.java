package FileEngine.DllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface HotkeyListener extends Library {
    HotkeyListener INSTANCE = Native.load("HotkeyListener", HotkeyListener.class);

    void registerHotKey(int key1, int key2, int key3, int key4, int key5);

    boolean getKeyStatus();

    void startListen();

    void stopListen();
}