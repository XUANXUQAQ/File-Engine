package file.engine.dllInterface;

import java.nio.file.Path;

public enum GetHandle {
    INSTANCE;

    static {
        System.load(Path.of("user/getHandle.dll").toAbsolutePath().toString());
    }

    public native void start();

    public native void stop();

    public native boolean changeToAttach();

    public native boolean changeToNormal();

    public native long getExplorerX();

    public native long getExplorerY();

    public native long getExplorerWidth();

    public native long getExplorerHeight();

    public native String getExplorerPath();

    public native boolean isDialogWindow();

    public native int getToolBarX();

    public native int getToolBarY();

    public native boolean isKeyPressed(int vk_key);

    public native boolean isForegroundFullscreen();

    public native void setEditPath(String path);
}
