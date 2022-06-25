package file.engine.dllInterface;

import java.nio.file.Path;

public enum GetStartMenu {
    INSTANCE;

    static {
        System.load(Path.of("user/getStartMenu.dll").toAbsolutePath().toString());
    }

    public native String getStartMenu();
}
