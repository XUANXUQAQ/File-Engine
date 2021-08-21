package file.engine.dllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface ResultPipe extends Library {
    ResultPipe INSTANCE = Native.load("resultPipe", ResultPipe.class);

    String getResult(char disk, String listName, int priority, int offset);

    void closeAllSharedMemory();

    boolean isComplete();
}
