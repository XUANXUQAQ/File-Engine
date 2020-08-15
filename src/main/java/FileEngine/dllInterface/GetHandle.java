package FileEngine.dllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface GetHandle extends Library {
    GetHandle INSTANCE = Native.load("getHandle", GetHandle.class);

    void start();

    void stop();

    boolean is_explorer_at_top();

    boolean isDialogNotExist();

    long getX();

    long getY();

    long getWidth();

    long getHeight();

    int get_toolbar_click_x();

    int get_toolbar_click_y();

    void set_searchBar(int x, int y, int width, int height);

    boolean isMouseClickOutSide();

    void resetMouseStatus();
}
