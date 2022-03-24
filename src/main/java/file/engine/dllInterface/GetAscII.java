package file.engine.dllInterface;

import java.nio.file.Path;

public enum GetAscII {
    INSTANCE;

    static {
        System.load(Path.of("user/getAscII.dll").toAbsolutePath().toString());
    }

    public native int getAscII(String str);
}
