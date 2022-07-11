package file.engine.services;

import file.engine.utils.RegexUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.Getter;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

public enum TranslateService {
    INSTANCE;

    private volatile @Getter
    String language;
    private final ConcurrentHashMap<String, String> translationMap = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, String> fileMap = new ConcurrentHashMap<>();
    private Font[] fList;

    TranslateService() {
        initAll();
    }

    public static TranslateService getInstance() {
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
        stringBuilder.append("</body></html>");
        return stringBuilder.toString();
    }

    /**
     * 获取系统默认语言
     *
     * @return 系统区域语言信息
     */
    public static String getDefaultLang() {
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
            if (IsDebug.isDebug()) {
                translationMap.clear();
            }
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(Objects.requireNonNull(TranslateService.class.getResourceAsStream(filePath)), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] record = RegexUtil.getPattern("=", 0).split(line);
                    if (IsDebug.isDebug()) {
                        if (translationMap.get(record[0].toLowerCase()) != null) {
                            System.err.println("警告：翻译重复   " + record[0]);
                        }
                    }
                    if (record.length == 2) {
                        translationMap.put(record[0].trim().toLowerCase(), record[1].trim());
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
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
        initTranslations();
        setUIFont();
    }

    public String[] getLanguageArray() {
        int size = fileMap.size();
        ArrayList<String> languages = new ArrayList<>(fileMap.keySet());
        String[] languageArray = new String[size];
        for (int i = 0; i < size; i++) {
            languageArray[i] = languages.get(i);
        }
        return languageArray;
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
     * 测试字体是否能显示所有的翻译文字
     * @param font 字体
     * @return boolean
     */
    private boolean canDisplay(Font font) {
        for (String each : translationMap.values()) {
            if (font.canDisplayUpTo(each) != -1) {
                return false;
            }
        }
        return true;
    }

    /**
     * 自动寻找可以显示的字体并加载
     *
     * @return 字体
     */
    private Font getFitFont() {
        Font defaultFont = new Font(Font.SANS_SERIF, Font.PLAIN, 13);
        if (!canDisplay(defaultFont)) {
            for (Font each : fList) {
                if (canDisplay(each)) {
                    return each;
                }
            }
        }
        return defaultFont;
    }

    /**
     * 自动寻找可以显示的字体并加载
     *
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
        Font f = getFitFont();
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

