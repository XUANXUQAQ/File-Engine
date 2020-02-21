package fileMonitor;

import com.sun.jna.*;

import java.io.File;


public interface FileMonitor extends Library {
    File dll = new File("./user/fileMonitor.dll");
    FileMonitor INSTANCE = (FileMonitor) Native.loadLibrary(dll.getAbsolutePath(), FileMonitor.class);

    void monitor(String path, String output, String closePosition);
}
