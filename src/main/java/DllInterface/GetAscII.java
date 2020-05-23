package DllInterface;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface GetAscII extends Library {
    GetAscII INSTANCE = Native.load("getAscII", GetAscII.class);

    int getAscII(String str);
}
