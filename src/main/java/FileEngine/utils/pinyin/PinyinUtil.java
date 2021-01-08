package FileEngine.utils.pinyin;

import FileEngine.utils.RegexUtil;
import com.github.promeg.pinyinhelper.Pinyin;

import java.util.Arrays;

public class PinyinUtil {
    private static final String[] ALL_PINYIN =
            {
                "a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao", "bei", "ben",
                "beng", "bi", "bian", "biao", "bie", "bin", "bing", "bo", "bu", "ca", "cai",
                "can", "cang", "cao", "ce", "cen", "ceng", "cha", "chai", "chan", "chang",
                "chao", "che", "chen", "cheng", "chi", "chong", "chou", "chu", "chua",
                "chuai", "chuan", "chuang", "chui", "chun", "chuo", "ci", "cong", "cou",
                "cu", "cuan", "cui", "cun", "cuo", "da", "dai", "dan", "dang", "dao", "de",
                "dei", "den", "deng", "di", "dia", "dian", "diao", "die", "ding", "diu",
                "dong", "dou", "du", "duan", "dui", "dun", "duo", "e", "ei", "en", "eng",
                "er", "fa", "fan", "fang", "fei", "fen", "feng", "fiao", "fo", "fou", "fu",
                "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng", "gong",
                "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo", "ha",
                "hai", "han", "hang", "hao", "he", "hei", "hen", "heng", "hm", "hng",
                "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun", "huo",
                "ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing", "jiong", "jiu",
                "ju", "juan", "jue", "jun", "ka", "kai", "kan", "kang", "kao", "ke", "kei",
                "ken", "keng", "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui",
                "kun", "kuo", "la", "lai", "lan", "lang", "lao", "le", "lei", "leng", "li",
                "lia", "lian", "liang", "liao", "lie", "lin", "ling", "liu", "lo", "long",
                "lou", "lu", "luan", "lue", "lun", "luo", "lv", "m", "ma", "mai", "man",
                "mang", "mao", "me", "mei", "men", "meng", "mi", "mian", "miao", "mie",
                "min", "ming", "miu", "mo", "mou", "mu", "n", "na", "nai", "nan", "nang",
                "nao", "ne", "nei", "nen", "neng", "ng", "ni", "nian", "niang", "niao",
                "nie", "nin", "ning", "niu", "nong", "nou", "nu", "nuan", "nue", "nuo",
                "nv", "o", "ou", "pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng",
                "pi", "pian", "piao", "pie", "pin", "ping", "po", "pou", "pu", "qi", "qia",
                "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu", "qu", "quan",
                "que", "qun", "ran", "rang", "rao", "re", "ren", "reng", "ri", "rong",
                "rou", "ru", "ruan", "rui", "run", "ruo", "sa", "sai", "san", "sang", "sao",
                "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao", "she", "shei",
                "shen", "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan", "shuang",
                "shui", "shun", "shuo", "si", "song", "sou", "su", "suan", "sui", "sun",
                "suo", "ta", "tai", "tan", "tang", "tao", "te", "tei", "teng", "ti", "tian",
                "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui", "tun", "tuo",
                "wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia",
                "xian", "xiang", "xiao", "xie", "xin", "xing", "xiong", "xiu", "xu", "xuan",
                "xue", "xun", "ya", "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo",
                "yong", "you", "yu", "yuan", "yue", "yun", "za", "zai", "zan", "zang",
                "zao", "ze", "zei", "zen", "zeng", "zha", "zhai", "zhan", "zhang", "zhao",
                "zhe", "zhei", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu", "zhua",
                "zhuai", "zhuan", "zhuang", "zhui", "zhun", "zhuo", "zi", "zong", "zou",
                "zu", "zuan", "zui", "zun", "zuo"
            };
    private static boolean isPinyin0(String pinyin) {
        pinyin = pinyin.toLowerCase();
        return Arrays.binarySearch(ALL_PINYIN, pinyin) >= 0;
    }

    /**
     * 对输入的拼音进行分词，再逐一检测分词正确性，如果有一个不为拼音，则判定整个字符串不为拼音
     * @param str 字符串
     * @return true如果是由拼音组成的字符串
     */
    public static boolean isPinyin(String str) {
        String[] pinyin = RegexUtil.blank.split(SeparateTool.trimSpell(str));
        for (String eachPinyin : pinyin) {
            if (!isPinyin0(eachPinyin)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 判断字符串是否存在汉字
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
     * @param str 待转换的字符串
     * @param separator 分隔符
     * @return 转换后的字符串
     */
    public static String toPinyin(String str, String separator) {
        return Pinyin.toPinyin(str, separator).toLowerCase();
    }
}
