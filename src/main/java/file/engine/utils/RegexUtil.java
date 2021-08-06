package file.engine.utils;

import java.util.regex.Pattern;

public class RegexUtil {
    public static final Pattern blank = Pattern.compile(" ");
    public static final Pattern semicolon = Pattern.compile(";");
    public static final Pattern colon = Pattern.compile(":");
    public static final Pattern slash = Pattern.compile("/");
    public static final Pattern reverseSlash = Pattern.compile("\\\\");
    public static final Pattern rgbHexPattern = Pattern.compile("^[a-fA-F0-9]{6}$");
    public static final Pattern plus = Pattern.compile(" \\+ ");
    public static final Pattern equalSign = Pattern.compile("=");
    public static final Pattern lineFeed = Pattern.compile("\n");
    public static final Pattern comma = Pattern.compile(",");
}
