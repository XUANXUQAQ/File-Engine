package file.engine.dllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface GetHandle extends Library {
    GetHandle INSTANCE = Native.load("getHandle", GetHandle.class);

    void start();

    void stop();

    boolean changeToAttach();

    boolean changeToNormal();

    long getExplorerX();

    long getExplorerY();

    long getExplorerWidth();

    long getExplorerHeight();

    String getExplorerPath();

    boolean isDialogWindow();

    void bringSearchBarToTop();

    int getToolBarX();

    int getToolBarY();

    double getDpi();

    boolean isKeyPressed(int vk_key);

    boolean isForegroundFullscreen();
}
