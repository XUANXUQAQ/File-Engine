package fileMonitor;

import com.sun.jna.Library;
import com.sun.jna.Native;
import main.MainClass;


public interface FileMonitor extends Library {
    FileMonitor INSTANCE = Native.load(MainClass.fileMonitorDllName, FileMonitor.class);

    void monitor(String path, String output, String closePosition);
}
