package FileEngine.translate;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Locale;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

public class TranslateUtil {
    private static class TranslateUtilBuilder {
        private static final TranslateUtil INSTANCE = new TranslateUtil();
    }

    private volatile static String language;
    private final HashSet<String> languageSet = new HashSet<>();
    private final ConcurrentHashMap<String, String> translationMap = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, String> fileMap = new ConcurrentHashMap<>();
    private static final Pattern EQUAL_SIGN = Pattern.compile("=");

    private TranslateUtil() {
        initAll();
    }

    public static TranslateUtil getInstance() {
        return TranslateUtilBuilder.INSTANCE;
    }

    public String getTranslation(String text) {
        String translated;
        if ("English(US)".equals(language)) {
            translated = text;
        } else {
            translated = translationMap.get(text);
        }
        if (translated != null) {
            return translated;
        } else {
            return text;
        }
    }

    public String getDefaultLang() {
        //TODO 添加语言
        Locale l = Locale.getDefault();
        String lang = l.toLanguageTag();
        switch (lang) {
            case "zh-CN":
                return "简体中文";
            case "ja-JP":
            case "ja-JP-u-ca-japanese":
            case "ja-JP-x-lvariant-JP":
                return "日本語";
            case "zh-HK":
            case "zh-TW":
                return "繁體中文";
            default:
                return "English(US)";
        }
    }

    private void initLanguageSet() {
        //TODO 添加语言
        languageSet.add("简体中文");
        languageSet.add("English(US)");
        languageSet.add("日本語");
        languageSet.add("繁體中文");
    }

    private void initLanguageFileMap() {
        //TODO 添加语言
        fileMap.put("简体中文", "/language/Chinese(Simplified).txt");
        fileMap.put("日本語", "/language/Japanese.txt");
        fileMap.put("繁體中文", "/language/Chinese(Traditional).txt");
    }

    private void initTranslations() {
        if (!"English(US)".equals(language)) {
            String filePath = fileMap.get(language);
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(TranslateUtil.class.getResourceAsStream(filePath), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] record = TranslateUtil.EQUAL_SIGN.split(line);
                    translationMap.put(record[0].trim(), record[1].trim());
                }
            } catch (IOException ignored) {
            }
        } else {
            translationMap.put("#frame_width", String.valueOf(1000));
            translationMap.put("#frame_height", String.valueOf(600));
        }
    }

    private void initAll() {
        initLanguageFileMap();
        initLanguageSet();
        language = getDefaultLang();
        initTranslations();
    }

    public void setLanguage(String language) {
        TranslateUtil.language = language;
        initTranslations();
    }

    public String getLanguage() {
        return language;
    }

    public HashSet<String> getLanguageSet() {
        return languageSet;
    }

    public String getFrameWidth() {
        return translationMap.get("#frame_width");
    }

    public String getFrameHeight() {
        return translationMap.get("#frame_height");
    }
}
