package FileEngine.dllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface isNTFS extends Library {
    isNTFS INSTANCE = Native.load("isNTFS", isNTFS.class);

    boolean isDiskNTFS(String disk);
}
