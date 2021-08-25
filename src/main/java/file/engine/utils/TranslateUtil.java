package file.engine.utils;

import file.engine.utils.system.properties.IsDebug;
import lombok.Getter;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public enum TranslateUtil {
    INSTANCE;

    private volatile @Getter
    String language;
    private final ConcurrentHashMap<String, String> translationMap = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, String> fileMap = new ConcurrentHashMap<>();
    private Font[] fList;

    TranslateUtil() {
        initAll();
    }

    public static TranslateUtil getInstance() {
        return INSTANCE;
    }

    /**
     * 获取翻译
     *
     * @param text 原文
     * @return 翻译
     */
    public String getTranslation(String text) {
        String translated;
        if ("English(US)".equals(language)) {
            translated = text;
        } else {
            translated = translationMap.get(text.toLowerCase());
        }
        if (translated != null) {
            return warpStringIfTooLong(translated);
        } else {
            return text;
        }
    }

    /**
     * 如果翻译太长则换行
     *
     * @param str 翻译
     * @return html包裹的翻译
     */
    private String warpStringIfTooLong(String str) {
        if (str.length() < 60) {
            return str;
        }
        String[] split = RegexUtil.blank.split(str);
        int totalLength = split.length;
        int splitCount = totalLength / 2;
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("<html><body>");
        for (int i = 0; i < totalLength; i++) {
            if (i == splitCount) {
                stringBuilder.append("<br>");
            }
            stringBuilder.append(split[i]).append(" ");
        }
        stringBuilder.append("<body><html>");
        return stringBuilder.toString();
    }

    /**
     * 获取系统默认语言
     *
     * @return 系统区域语言信息
     */
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
            case "ru-RU":
                return "русский";
            case "it-IT":
            case "it-CH":
                return "italiano";
            case "de-AT":
            case "de-DE":
            case "de-LU":
            case "de-CH":
                return "German";
            case "ko-KR":
                return "한국어";
            case "fr-BE":
            case "fr-CA":
            case "fr-FR":
            case "fr-LU":
            case "fr-CH":
                return "français";
            default:
                return "English(US)";
        }
    }

    private void initLanguageFileMap() {
        //TODO 添加语言
        fileMap.put("简体中文", "/language/Chinese(Simplified).txt");
        fileMap.put("日本語", "/language/Japanese.txt");
        fileMap.put("繁體中文", "/language/Chinese(Traditional).txt");
        fileMap.put("русский", "/language/Russian.txt");
        fileMap.put("italiano", "/language/Italian.txt");
        fileMap.put("Deutsche", "/language/German.txt");
        fileMap.put("한국어", "/language/Korean.txt");
        fileMap.put("français", "/language/French.txt");
        fileMap.put("English(US)", "");
    }

    private void initTranslations() {
        if (!"English(US)".equals(language)) {
            String filePath = fileMap.get(language);
            translationMap.clear();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(Objects.requireNonNull(TranslateUtil.class.getResourceAsStream(filePath)), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] record = RegexUtil.equalSign.split(line);
                    if (IsDebug.isDebug()) {
                        if (translationMap.get(record[0].toLowerCase()) != null) {
                            System.err.println("警告：翻译重复   " + record[0]);
                        }
                    }
                    if (record.length == 2) {
                        translationMap.put(record[0].trim().toLowerCase(), record[1].trim());
                    }
                }
            } catch (IOException ignored) {
            }
        } else {
            translationMap.put("#frame_width", String.valueOf(1000));
            translationMap.put("#frame_height", String.valueOf(600));
        }
    }

    /**
     * 初始化
     */
    private void initAll() {
        initLanguageFileMap();
        language = getDefaultLang();
        initTranslations();
        initFontList();
    }

    /**
     * 初始化Font列表
     */
    private void initFontList() {
        String[] lstr = GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames();
        fList = new Font[lstr.length];
        for (int i = 0; i < lstr.length; i++) {
            fList[i] = new Font(lstr[i], Font.PLAIN, 13);
        }
    }

    /**
     * 设置语言
     *
     * @param language 语言
     */
    public void setLanguage(String language) {
        this.language = language;
        setUIFont();
        initTranslations();
    }

    public Set<String> getLanguageSet() {
        return fileMap.keySet();
    }

    /**
     * 获取当前语言下定义的窗口宽度
     *
     * @return width
     */
    public String getFrameWidth() {
        return translationMap.get("#frame_width");
    }

    /**
     * 获取当前语言下定义的窗口高度
     *
     * @return height
     */
    public String getFrameHeight() {
        return translationMap.get("#frame_height");
    }

    /**
     * 自动寻找可以显示的字体并加载
     *
     * @param fontStyle 字体风格
     * @param size      大小
     * @param testStr   测试字符串，用于系统检测是否可以显示该字符串
     * @return 字体
     */
    public Font getFitFont(int fontStyle, int size, String testStr) {
        Font defaultFont = new Font(Font.SANS_SERIF, fontStyle, size);
        if (defaultFont.canDisplayUpTo(testStr) != -1) {
            for (Font each : fList) {
                if (each.canDisplayUpTo(testStr) == -1) {
                    return each;
                }
            }
        }
        return defaultFont;
    }

    /**
     * 加载字体
     */
    private void setUIFont() {
        Font f = getFitFont(Font.PLAIN, 13, language);
        String[] names = {"Label", "CheckBox", "PopupMenu", "MenuItem", "CheckBoxMenuItem",
                "JRadioButtonMenuItem", "ComboBox", "Button", "Tree", "ScrollPane",
                "TabbedPane", "EditorPane", "TitledBorder", "Menu", "TextArea",
                "OptionPane", "MenuBar", "ToolBar", "ToggleButton", "ToolTip",
                "ProgressBar", "TableHeader", "Panel", "List", "ColorChooser",
                "PasswordField", "TextField", "Table", "Label", "Viewport",
                "RadioButtonMenuItem", "RadioButton", "DesktopPane", "InternalFrame"
        };
        for (String item : names) {
            UIManager.put(item + ".font", f);
        }
    }
}

