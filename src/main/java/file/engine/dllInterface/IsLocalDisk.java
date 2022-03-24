package file.engine.dllInterface;

import java.nio.file.Path;

public enum IsLocalDisk {
    INSTANCE;

    static {
        System.load(Path.of("user/isLocalDisk.dll").toAbsolutePath().toString());
    }

    public native boolean isLocalDisk(String path);

    public native boolean isDiskNTFS(String disk);
}
