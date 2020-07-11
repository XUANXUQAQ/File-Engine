package FileEngine.DllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;


public interface FileMonitor extends Library {
    FileMonitor INSTANCE = Native.load("fileMonitor", FileMonitor.class);

    void monitor(String path);

    void stop_monitor();

    void set_output(String path);
}
