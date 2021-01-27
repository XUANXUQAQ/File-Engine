package FileEngine.dllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

@SuppressWarnings("unused")
public interface GetHandle extends Library {
    GetHandle INSTANCE = Native.load("getHandle", GetHandle.class);

    int EXPLORER = 0x01;
    int DIALOG = 0x02;

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
