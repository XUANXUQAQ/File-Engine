package DllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;


public interface FileMonitor extends Library {
    FileMonitor INSTANCE = Native.load("fileMonitor", FileMonitor.class);

    void monitor(String path, String output, String closePosition);
}
