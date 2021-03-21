package file.engine.utils;

import com.github.promeg.pinyinhelper.Pinyin;

public class PinyinUtil {

    /**
     * 判断字符串是否存在汉字
     *
     * @param str 字符串
     * @return true如果存在
     */
    public static boolean isContainChinese(String str) {
        final int length = str.length();
        for (int i = 0; i < length; i++) {
            if (Pinyin.isChinese(str.charAt(i))) {
                return true;
            }
        }
        return false;
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
}
