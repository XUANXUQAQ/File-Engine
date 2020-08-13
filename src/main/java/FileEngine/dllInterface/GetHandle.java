package FileEngine.dllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface GetHandle extends Library {
    GetHandle INSTANCE = Native.load("getHandle", GetHandle.class);

    void start();

    void stop();

    boolean is_explorer_at_top();

    long getX();

    long getY();

    long getWidth();

    long getHeight();
}
