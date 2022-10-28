package file.engine.dllInterface;

import java.nio.file.Path;

public enum ResultPipe {
    INSTANCE;

    static {
        System.load(Path.of("user/resultPipe.dll").toAbsolutePath().toString());
    }

    public native String getResult(char disk, String listName, int priority, int offset);

    public native void closeAllSharedMemory();

    public native boolean isComplete();

    public native boolean isDatabaseComplete();
}
