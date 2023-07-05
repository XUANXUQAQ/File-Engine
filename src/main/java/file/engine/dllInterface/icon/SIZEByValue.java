package file.engine.dllInterface.icon;

import com.sun.jna.Structure;
import com.sun.jna.platform.win32.WinUser;

/**
 * Copyright (c) 08.11.2019
 * Developed by MrMarnic
 * GitHub: https://github.com/MrMarnic
 */
public class SIZEByValue extends WinUser.SIZE implements Structure.ByValue {
    public SIZEByValue(int w, int h) {
        super(w, h);
    }
}