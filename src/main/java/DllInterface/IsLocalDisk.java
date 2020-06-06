package DllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

import java.io.File;

public interface IsLocalDisk extends Library {
    File f = new File("user");
    IsLocalDisk INSTANCE = Native.load(f.getAbsolutePath() + File.separator + "isLocalDisk.dll", IsLocalDisk.class);

    boolean isLocalDisk(String path);
}
