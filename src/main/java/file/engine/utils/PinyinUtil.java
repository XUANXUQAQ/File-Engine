package file.engine.utils;

import com.github.promeg.pinyinhelper.Pinyin;
import com.github.promeg.tinypinyin.lexicons.java.cncity.CnCityDict;

import java.util.HashMap;

public class PinyinUtil {

    static {
        initPinyin();
    }

    /**
     * 判断字符串是否存在汉字
     *
     * @param str 字符串
     * @return true如果存在
     */
    public static boolean isStringContainChinese(String str) {
        final int length = str.length();
        for (int i = 0; i < length; i++) {
            if (Pinyin.isChinese(str.charAt(i))) {
                return true;
            }
        }
        return false;
    }

    /**
     * 挑出字符串中所有的中文，转换成拼音然后返回hash表
     *
     * @param str 字符串
     * @return hashMap
     */
    public static HashMap<String, String> getChinesePinyinMap(String str) {
        HashMap<String, String> ret = new HashMap<>();
        final int length = str.length();
        for (int i = 0; i < length; i++) {
            char c = str.charAt(i);
            if (Pinyin.isChinese(c)) {
                ret.put(String.valueOf(c), Pinyin.toPinyin(c));
            }
        }
        return ret;
    }

    /**
     * 将汉字转为拼音，不同拼音用separator分开
     *
     * @param str       待转换的字符串
     * @param separator 分隔符
     * @return 转换后的字符串
     */
    public static String toPinyin(String str, String separator) {
        return Pinyin.toPinyin(str, separator).toLowerCase();
    }

    private static void initPinyin() {
        CnCityDict cnCityDict = CnCityDict.getInstance();
        Pinyin.init(Pinyin.newConfig().with(cnCityDict));
    }
}
