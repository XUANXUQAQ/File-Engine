package file.engine.dllInterface;

import java.nio.file.Path;

public enum SystemThemeInfo {
    INSTANCE;

    static {
        System.load(Path.of("user/systemThemeInfo.dll").toAbsolutePath().toString());
    }

    public native boolean isDarkThemeEnabled();
}
