package file.engine.utils;

import file.engine.utils.file.FileUtil;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;

@SuppressWarnings({"IndexOfReplaceableByContains"})
public class PathMatchUtil {

    /**
     * 判断文件路径是否满足当前匹配结果（该方法由check方法使用），检查文件路径使用check方法。
     *
     * @param path         文件路径
     * @param isIgnoreCase 是否忽略大小写
     * @return 如果匹配成功则返回true
     * @see #check(String, String[], String, String[])  ;
     */
    private static boolean notMatched(String path, boolean isIgnoreCase, String[] keywords) {
        String matcherStrFromFilePath;
        boolean isPath;
        for (String eachKeyword : keywords) {
            if (eachKeyword == null || eachKeyword.isEmpty()) {
                continue;
            }
            char firstChar = eachKeyword.charAt(0);
            if (firstChar == '/' || firstChar == File.separatorChar) {
                //匹配路径
                isPath = true;
                Matcher matcher = RegexUtil.slash.matcher(eachKeyword);
                eachKeyword = matcher.replaceAll(Matcher.quoteReplacement(""));
                //获取父路径
                matcherStrFromFilePath = FileUtil.getParentPath(path);
            } else {
                //获取名字
                isPath = false;
                matcherStrFromFilePath = FileUtil.getFileName(path);
            }
            //转换大小写
            if (isIgnoreCase) {
                matcherStrFromFilePath = matcherStrFromFilePath.toLowerCase();
                eachKeyword = eachKeyword.toLowerCase();
            }
            //开始匹配
            if (matcherStrFromFilePath.indexOf(eachKeyword) == -1) {
                if (isPath) {
                    return true;
                } else {
                    if (PinyinUtil.isStringContainChinese(matcherStrFromFilePath)) {
                        if (PinyinUtil.toPinyin(matcherStrFromFilePath, "").indexOf(eachKeyword) == -1) {
                            return true;
                        }
                    } else {
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
     * @param path 文件路径
     * @return true如果满足所有条件 否则false
     */
    public static boolean check(String path, String[] searchCase, String searchText, String[] keywords) {
        if (notMatched(path, true, keywords)) {
            return false;
        }
        if (searchCase == null || searchCase.length == 0) {
            return true;
        }
        Path pathVar = Path.of(path);
        for (String eachCase : searchCase) {
            switch (eachCase) {
                case "f":
                    if (!Files.isRegularFile(pathVar)) {
                        return false;
                    }
                    break;
                case "d":
                    if (!Files.isDirectory(pathVar)) {
                        return false;
                    }
                    break;
                case "full":
                    if (!searchText.equalsIgnoreCase(FileUtil.getFileName(path))) {
                        return false;
                    }
                    break;
                case "case":
                    if (notMatched(path, false, keywords)) {
                        return false;
                    }
            }
        }
        //所有规则均已匹配
        return true;
    }
}
