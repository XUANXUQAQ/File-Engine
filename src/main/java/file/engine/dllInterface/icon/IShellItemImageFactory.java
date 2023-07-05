package file.engine.dllInterface.icon;

import com.sun.jna.Pointer;
import com.sun.jna.platform.win32.COM.Unknown;
import com.sun.jna.platform.win32.WinNT;
import com.sun.jna.ptr.PointerByReference;

/**
 * Copyright (c) 31.10.2019
 * Developed by MrMarnic
 * GitHub: https://github.com/MrMarnic
 */
public class IShellItemImageFactory extends Unknown {

    public IShellItemImageFactory(Pointer pvInstance) {
        super(pvInstance);
    }

    public WinNT.HRESULT GetImage(
            SIZEByValue size,
            int  flags,
            PointerByReference bitmap
    ) {
        return (WinNT.HRESULT) _invokeNativeObject(3,
                new Object[]{this.getPointer(), size, flags, bitmap},WinNT.HRESULT.class);
    }
}