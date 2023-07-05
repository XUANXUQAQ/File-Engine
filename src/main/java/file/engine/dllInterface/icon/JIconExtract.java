package file.engine.dllInterface.icon;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.WString;
import com.sun.jna.platform.win32.COM.COMUtils;
import com.sun.jna.platform.win32.*;
import com.sun.jna.ptr.PointerByReference;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;

/**
 * Copyright (c) 08.11.2019
 * Developed by MrMarnic
 * GitHub: https://github.com/MrMarnic
 */
public class JIconExtract {

    /**
     * @param width  width of the returned BufferedImage
     * @param height height of the returned BufferedImage
     * @param file   Path to the requested file
     * @return BufferedImage for the given File
     */
    public static BufferedImage getIconForFile(int width, int height, Path file) {
        return getIconForFile(width, height, file.toString());
    }

    /**
     * @param width  width of the returned BufferedImage
     * @param height height of the returned BufferedImage
     * @param file   File to the requested file
     * @return BufferedImage for the given File
     */
    public static BufferedImage getIconForFile(int width, int height, File file) {
        return getIconForFile(width, height, file.getAbsolutePath());
    }

    /**
     * @param width    width of the returned BufferedImage
     * @param height   height of the returned BufferedImage
     * @param fileName Path given by String to the requested file
     * @return BufferedImage for the given File
     */
    public static BufferedImage getIconForFile(int width, int height, String fileName) {
        WinDef.HBITMAP hbitmap = getHBITMAPForFile(width, height, fileName);

        WinGDI.BITMAP bitmap = new WinGDI.BITMAP();

        try {
            int s = GDI32.INSTANCE.GetObject(hbitmap, bitmap.size(), bitmap.getPointer());

            if (s > 0) {
                bitmap.read();

                int w = bitmap.bmWidth.intValue();
                int h = bitmap.bmHeight.intValue();

                final WinDef.HDC hdc = User32.INSTANCE.GetDC(null);

                WinGDI.BITMAPINFO bitmapinfo = new WinGDI.BITMAPINFO();

                bitmapinfo.bmiHeader.biSize = bitmapinfo.bmiHeader.size();

                if (0 == GDI32.INSTANCE.GetDIBits(hdc, hbitmap, 0, 0, Pointer.NULL, bitmapinfo, WinGDI.DIB_RGB_COLORS)) {
                    throw new IllegalArgumentException("GetDIBits should not return 0");
                }

                bitmapinfo.read();

                Memory lpPixels = new Memory(bitmapinfo.bmiHeader.biSizeImage);

                bitmapinfo.bmiHeader.biCompression = WinGDI.BI_RGB;
                bitmapinfo.bmiHeader.biHeight = -h;

                if (0 == GDI32.INSTANCE.GetDIBits(hdc, hbitmap, 0, bitmapinfo.bmiHeader.biHeight, lpPixels, bitmapinfo, WinGDI.DIB_RGB_COLORS)) {
                    throw new IllegalArgumentException("GetDIBits should not return 0");
                }

                int[] colorArray = lpPixels.getIntArray(0, w * h);

                BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
                bi.setRGB(0, 0, w, h, colorArray, 0, w);

                return bi;
            }
        } finally {
            GDI32.INSTANCE.DeleteObject(hbitmap);
        }

        return null;
    }

    /**
     * @param width    width for the requested HBITMAP
     * @param height   height for the requested HBITMAP
     * @param fileName HBITMAP for the given File
     * @return Windows Native Implementation of HBITMAP (should not be used directly)
     */
    private static WinDef.HBITMAP getHBITMAPForFile(int width, int height, String fileName) {
        WinNT.HRESULT h1 = Ole32.INSTANCE.CoInitialize(null);

        if (COMUtils.SUCCEEDED(h1)) {

            PointerByReference factory = new PointerByReference();

            WinNT.HRESULT h2 = Shell32Extra.INSTANCE.SHCreateItemFromParsingName(new WString(fileName), null, new Guid.REFIID(new Guid.IID("BCC18B79-BA16-442F-80C4-8A59C30C463B")), factory);

            if (COMUtils.SUCCEEDED(h2)) {
                IShellItemImageFactory imageFactory = new IShellItemImageFactory(factory.getValue());

                PointerByReference hbitmapPointer = new PointerByReference();

                WinNT.HRESULT h3 = imageFactory.GetImage(new SIZEByValue(width, height), 0, hbitmapPointer);

                if (COMUtils.SUCCEEDED(h3)) {
                    WinDef.HBITMAP bitmap = new WinDef.HBITMAP(hbitmapPointer.getValue());

                    return bitmap;
                }

                imageFactory.Release();
            }
        }

        return null;
    }
}