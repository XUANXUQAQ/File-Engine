package DllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

import java.io.File;

public interface GetAscII extends Library {
    File f = new File("user");
    GetAscII INSTANCE = Native.load(f.getAbsolutePath() + File.separator + "getAscII.dll", GetAscII.class);

    int getAscII(String str);
}
