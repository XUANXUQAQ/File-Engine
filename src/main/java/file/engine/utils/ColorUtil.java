package file.engine.utils;

import com.formdev.flatlaf.util.HSLColor;

import java.awt.*;

public class ColorUtil {

    public static String parseColorHex(Color color) {
        int r = color.getRed();
        int g = color.getGreen();
        int b = color.getBlue();
        StringBuilder rgb = new StringBuilder();
        if (r == 0) {
            rgb.append("00");
        } else {
            rgb.append(Integer.toHexString(r));
        }
        if (g == 0) {
            rgb.append("00");
        } else {
            rgb.append(Integer.toHexString(g));
        }
        if (b == 0) {
            rgb.append("00");
        } else {
            rgb.append(Integer.toHexString(b));
        }
        return rgb.toString();
    }

    public static Color generateHighContrastColor(Color color) {
        HSLColor hslColor = new HSLColor(color);
        float hue = hslColor.getHue();
        float luminance = hslColor.getLuminance();
        float hue2;
        hue2 = hue + 120;
        hue2 = hue2 > 360 ? hue2 - 360 : hue2;
        luminance = (100 - luminance) % 30 + 50;
        return new HSLColor(hue2, 90, luminance).getRGB();
    }


    /**
     * 根据RGB值判断 深色与浅色
     *
     * @return true if color is dark
     */
    public static boolean isDark(int r, int g, int b) {
        return !(r * 0.299 + g * 0.578 + b * 0.114 >= 192);
    }


    public static boolean isDark(int rgbHex) {
        Color color = new Color(rgbHex);
        return isDark(color.getRed(), color.getGreen(), color.getBlue());
    }

    public static boolean canParseToRGB(String str) {
        if (str != null) {
            if (!str.isEmpty()) {
                return RegexUtil.rgbHexPattern.matcher(str).matches();
            }
        }
        return false;
    }

    public static String toRGBHexString(int colorRGB) {
        return String.format("%06x", colorRGB);
    }
}
