package file.engine.services.utils;

import file.engine.utils.PinyinUtil;
import file.engine.utils.RegexUtil;
import file.engine.utils.file.FileUtil;

import java.nio.file.Files;
import java.nio.file.Path;

@SuppressWarnings({"IndexOfReplaceableByContains"})
public class PathMatchUtil {

    /**
     * 判断文件路径是否满足当前匹配结果（该方法由check方法使用），检查文件路径使用check方法。
     *
     * @param path         文件路径
     * @param isIgnoreCase 是否忽略大小写
     * @return 如果匹配成功则返回true
     * @see #check(String, String[], boolean, String, String[], String[], boolean[])
     */
    private static boolean notMatched(String path, boolean isIgnoreCase, String[] keywords, String[] keywordsLowerCase, boolean[] isKeywordPath) {
        final int length = keywords.length;
        for (int i = 0; i < length; ++i) {
            String eachKeyword;
            final boolean isPath = isKeywordPath[i];
            String matcherStrFromFilePath = isPath ? FileUtil.getParentPath(path) : FileUtil.getFileName(path);
            if (isIgnoreCase) {
                eachKeyword = keywordsLowerCase[i];
                matcherStrFromFilePath = matcherStrFromFilePath.toLowerCase();
            } else {
                eachKeyword = keywords[i];
            }
            if (eachKeyword == null || eachKeyword.isEmpty()) {
                continue;
            }
            //开始匹配
            if (matcherStrFromFilePath.indexOf(eachKeyword) == -1) {
                if (isPath || !PinyinUtil.isStringContainChinese(matcherStrFromFilePath)) {
                    return true;
                } else {
                    String pinyin = PinyinUtil.toPinyin(matcherStrFromFilePath, ",");
                    String[] pinyinList = RegexUtil.comma.split(pinyin);
                    StringBuilder pinyinInitials = new StringBuilder();
                    for (String eachPinyin : pinyinList) {
                        if (eachPinyin.isEmpty()) {
                            continue;
                        }
                        pinyinInitials.append(eachPinyin.charAt(0));
                    }
                    pinyin = RegexUtil.comma.matcher(pinyin).replaceAll("");
                    if (pinyin.indexOf(eachKeyword) == -1 && pinyinInitials.indexOf(eachKeyword) == -1) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * 检查文件路径是否匹配所有输入规则
     *
     * @param path              将要匹配的文件路径
     * @param searchCase        匹配规则 f d case full
     * @param isIgnoreCase      当searchCase中包含case则为false，该变量用于防止searchCase重复计算
     * @param searchText        用户输入字符串，由关键字通过 ; 连接
     * @param keywords          用户输入字符串生成的关键字
     * @param keywordsLowerCase 防止重复计算
     * @param isKeywordPath     keyword是否为路径或者文件名
     * @return true如果满足所有条件 否则返回false
     */
    public static boolean check(String path,
                                String[] searchCase,
                                boolean isIgnoreCase,
                                String searchText,
                                String[] keywords,
                                String[] keywordsLowerCase,
                                boolean[] isKeywordPath) {
        if (notMatched(path, isIgnoreCase, keywords, keywordsLowerCase, isKeywordPath)) {
            return false;
        }
        if (searchCase == null) {
            return true;
        }
        for (String eachCase : searchCase) {
            switch (eachCase) {
                case SearchCase.F -> {
                    if (!Files.isRegularFile(Path.of(path))) {
                        return false;
                    }
                }
                case SearchCase.D -> {
                    if (!Files.isDirectory(Path.of(path))) {
                        return false;
                    }
                }
                case SearchCase.FULL -> {
                    if (!searchText.equalsIgnoreCase(FileUtil.getFileName(path))) {
                        return false;
                    }
                }
            }
        }
        //所有规则均已匹配
        return true;
    }

    public static class SearchCase {
        public static final String D = "d";
        public static final String F = "f";
        public static final String FULL = "full";
        public static final String CASE = "case";
    }
}
