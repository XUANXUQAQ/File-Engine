package FileEngine.dllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface GetHandle extends Library {
    GetHandle INSTANCE = Native.load("getHandle", GetHandle.class);

    int getExplorerTypeNum();

    int getDialogTypeNum();

    void start();

    void stop();

    boolean changeToAttach();

    boolean changeToNormal();

    long getExplorerX();

    long getExplorerY();

    long getExplorerWidth();

    long getExplorerHeight();

    String getExplorerPath();

    int getTopWindowType();

    void bringSearchBarToTop();

    int getToolBarX();

    int getToolBarY();
}
