package getAscII;

import com.sun.jna.Library;
import com.sun.jna.Native;
import main.MainClass;

public interface GetAscII extends Library {
    GetAscII INSTANCE = Native.load(MainClass.getAscIIDllName, GetAscII.class);

    int getAscII(String str);
}
