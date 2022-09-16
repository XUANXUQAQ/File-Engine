package file.engine.dllInterface;

import java.nio.file.Path;

public enum GetWindowsKnownFolder {
    INSTANCE;

    static {
        System.load(Path.of("user/getWindowsKnownFolder.dll").toAbsolutePath().toString());
    }

    public native String getKnownFolder(String guid);
}
