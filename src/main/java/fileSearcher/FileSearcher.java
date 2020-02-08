package fileSearcher;

import com.sun.jna.Library;
import com.sun.jna.Native;

import java.io.File;

public interface FileSearcher extends Library {
    File dll = new File("fileSearcher.dll");
    FileSearcher INSTANCE = (FileSearcher) Native.loadLibrary(dll.getAbsolutePath(), FileSearcher.class);

    void addIgnorePath(String path);
    void searchFiles(String path, String exd);
    void setSearchDepth(int i);
    String getResult();
    void deleteResult();
    boolean ResultReady();
}
