package file.engine.dllInterface;

import java.nio.file.Path;

public enum EmptyRecycleBin {
    INSTANCE;

    static {
        System.load(Path.of("user/emptyRecycleBin.dll").toAbsolutePath().toString());
    }

    /**
     * Windows清空回收站API
     */
    public native void emptyRecycleBin();
}
