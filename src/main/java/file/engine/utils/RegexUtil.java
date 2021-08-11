package file.engine.utils;

import file.engine.configs.Constants;

import java.util.concurrent.ConcurrentSkipListMap;
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

    private static final ConcurrentSkipListMap<String, Pattern> patternMap = new ConcurrentSkipListMap<>();

    /**
     * 获取正则表达式并放入缓存
     * @param patternStr 正则表达式
     * @param flags flags
     * @return 编译后的正则表达式
     */
    public static Pattern getPatter(String patternStr, int flags) {
        String key = patternStr + ":flags:" + flags;
        Pattern pattern = patternMap.get(key);
        if (pattern == null) {
            pattern = Pattern.compile(patternStr, flags);
            if (patternMap.size() < Constants.MAX_PATTERN_CACHE_NUM) {
                patternMap.put(key, pattern);
            } else {
                patternMap.remove(patternMap.firstEntry().getKey());
            }
        }
        return pattern;
    }
}
