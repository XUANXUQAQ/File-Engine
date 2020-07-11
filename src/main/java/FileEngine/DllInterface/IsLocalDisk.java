package FileEngine.DllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface IsLocalDisk extends Library {
    IsLocalDisk INSTANCE = Native.load("isLocalDisk", IsLocalDisk.class);

    boolean isLocalDisk(String path);
}
