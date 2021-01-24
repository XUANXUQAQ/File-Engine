package FileEngine.dllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface GetHandle extends Library {
    GetHandle INSTANCE = Native.load("getHandle", GetHandle.class);

    int EXPLORER = 0x01;
    int DIALOG = 0x02;

    void start();

    void stop();

    boolean isExplorerAtTop();

    boolean isDialogNotExist();

    long getExplorerX();

    long getExplorerY();

    long getExplorerWidth();

    long getExplorerHeight();

    int getToolbarClickX();

    int getToolbarClickY();

    boolean isExplorerAndSearchbarNotFocused();

    void setExplorerPath();

    String getExplorerPath();

    void setSearchBarUsingStatus(boolean b);

    int getTopWindowStatus();
}
