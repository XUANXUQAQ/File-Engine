package DllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

import java.io.File;

public interface HotkeyListener extends Library {
    File f = new File("user");
    HotkeyListener INSTANCE = Native.load(f.getAbsolutePath() + File.separator + "hotkeyListener.dll", HotkeyListener.class);

    void registerHotKey(int key1, int key2, int key3);

    boolean getKeyStatus();

    void startListen();

    void stopListen();
}