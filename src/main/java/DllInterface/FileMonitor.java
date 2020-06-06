package DllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

import java.io.File;


public interface FileMonitor extends Library {
    File f = new File("user");
    FileMonitor INSTANCE = Native.load(f.getAbsolutePath() + File.separator + "fileMonitor.dll", FileMonitor.class);

    void monitor(String path, String output, String closePosition);
}
