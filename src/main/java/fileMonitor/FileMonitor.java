package fileMonitor;
import com.sun.jna.*;

import java.io.File;


public interface FileMonitor extends Library{
    File dll = new File("fileMonitor.dll");
    FileMonitor INSTANCE = (FileMonitor) Native.loadLibrary(dll.getAbsolutePath(), FileMonitor.class);


    void fileWatcher(String path, String output, String closeSignalPosition);
}
