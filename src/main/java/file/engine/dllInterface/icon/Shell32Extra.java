package file.engine.dllInterface.icon;


import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.WString;
import com.sun.jna.platform.win32.Guid;
import com.sun.jna.platform.win32.Shell32;
import com.sun.jna.platform.win32.WinNT;
import com.sun.jna.ptr.PointerByReference;

/**
 * Copyright (c) 31.10.2019
 * Developed by MrMarnic
 * GitHub: https://github.com/MrMarnic
 */
public interface Shell32Extra extends Shell32 {
    Shell32Extra INSTANCE = Native.load("shell32", Shell32Extra.class);

    WinNT.HRESULT SHCreateItemFromParsingName(WString path, Pointer pointer, Guid.REFIID guid, PointerByReference reference);
}