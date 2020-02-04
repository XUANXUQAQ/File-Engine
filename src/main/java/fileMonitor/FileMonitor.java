package fileMonitor;
import com.sun.jna.*;

public interface FileMonitor extends Library{
    FileMonitor INSTANCE = (FileMonitor) Native.loadLibrary("fileMonitor", FileMonitor.class);

    void fileWatcher(String path, String output, String closeSignalPosition);
}
