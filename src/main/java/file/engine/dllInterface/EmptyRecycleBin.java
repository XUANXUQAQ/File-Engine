package file.engine.dllInterface;

import java.nio.file.Path;

public enum EmptyRecycleBin {
    INSTANCE;

    static {
        System.load(Path.of("user/emptyRecycleBin.dll").toAbsolutePath().toString());
    }

    public native void emptyRecycleBin();
}
