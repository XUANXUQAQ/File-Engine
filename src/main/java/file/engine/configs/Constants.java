package file.engine.configs;

import file.engine.dllInterface.SystemThemeInfo;
import file.engine.utils.system.properties.IsDebug;
import lombok.NoArgsConstructor;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import static file.engine.utils.BeanUtil.copyMatchingFields;

public class Constants {
    public static String version;

    public static final int ALL_TABLE_NUM = 40;

    public static final int MIN_FRAME_VISIBLE_TIME = 500;

    public static final int CLOSE_DATABASE_TIMEOUT_MILLS = 60 * 1000;

    public static final String FILE_NAME = "File-Engine.jar";

    public static final String LAUNCH_WRAPPER_NAME = "File-Engine.exe";

    public static final int API_VERSION = 6;

    public static final int[] COMPATIBLE_API_VERSIONS = {3, 4, 5, 6};

    public static final String DEFAULT_SWING_THEME = "MaterialLighter";

    static {
        version = "0";
        if (!IsDebug.isDebug()) {
            Properties properties = new Properties();
            try (InputStream projectInfo = Constants.class.getResourceAsStream("/project-info.properties")) {
                properties.load(projectInfo);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            version = properties.getProperty("project.version");
        }
    }

    private Constants() {
        throw new RuntimeException("not allowed");
    }

    public static class Enums {

        public enum DatabaseStatus {
            NORMAL, _TEMP, _SHARED_MEMORY, VACUUM, MANUAL_UPDATE
        }

        public enum DownloadStatus {
            DOWNLOAD_DONE, DOWNLOAD_ERROR, DOWNLOAD_DOWNLOADING, DOWNLOAD_INTERRUPTED, DOWNLOAD_NO_TASK
        }

        public enum ShowingSearchBarMode {
            NORMAL_SHOWING, EXPLORER_ATTACH
        }

        public static class ProxyType {
            public static final int PROXY_HTTP = 0x100;
            public static final int PROXY_SOCKS = 0x200;
            public static final int PROXY_DIRECT = 0x300;
        }

        public enum SwingThemes {
            CoreFlatDarculaLaf, CoreFlatDarkLaf, CoreFlatLightLaf, CoreFlatIntelliJLaf,
            Arc, ArcDark, ArcDarkOrange, Carbon,
            CyanLight, DarkFlat, DarkPurple, Dracula,
            Gray, LightFlat, MaterialDesignDark, Monocai,
            Nord, OneDark, MaterialDarker, MaterialLighter, SolarizedDark, SolarizedLight,
            Spacegray, Vuesion, XcodeDark, SystemDefault
        }

        public enum BorderType {
            EMPTY, AROUND, FULL
        }
    }

    public static class DefaultColors {
        private static final SearchBarColor dark;
        private static final SearchBarColor light;

        static {
            dark = new SearchBarColor();
            light = new SearchBarColor();
            copyMatchingFields(new Dark(), dark);
            copyMatchingFields(new Light(), light);
        }

        @NoArgsConstructor
        public static class SearchBarColor {
            public int DEFAULT_LABEL_COLOR;
            public int DEFAULT_WINDOW_BACKGROUND_COLOR;
            public int DEFAULT_BORDER_COLOR;
            public int DEFAULT_FONT_COLOR;
            public int DEFAULT_FONT_COLOR_WITH_COVERAGE;
            public int DEFAULT_SEARCHBAR_COLOR;
            public int DEFAULT_SEARCHBAR_FONT_COLOR;
        }

        @SuppressWarnings("unused")
        private static class Light {
            int DEFAULT_LABEL_COLOR = 0xff9933;
            int DEFAULT_WINDOW_BACKGROUND_COLOR = 0xf6f6f6;
            int DEFAULT_BORDER_COLOR = 0xf6f6f6;
            int DEFAULT_FONT_COLOR = 0;
            int DEFAULT_FONT_COLOR_WITH_COVERAGE = 0xff3333;
            int DEFAULT_SEARCHBAR_COLOR = 0xf6f6f6;
            int DEFAULT_SEARCHBAR_FONT_COLOR = 0;
        }

        @SuppressWarnings("unused")
        private static class Dark {
            int DEFAULT_LABEL_COLOR = 0x9999ff;
            int DEFAULT_WINDOW_BACKGROUND_COLOR = 0x333333;
            int DEFAULT_BORDER_COLOR = 0x333333;
            int DEFAULT_FONT_COLOR = 0xffffff;
            int DEFAULT_FONT_COLOR_WITH_COVERAGE = 0xff3333;
            int DEFAULT_SEARCHBAR_COLOR = 0x333333;
            int DEFAULT_SEARCHBAR_FONT_COLOR = 0xffffff;
        }

        public static SearchBarColor getDefaultSearchBarColor() {
            if (SystemThemeInfo.INSTANCE.isDarkThemeEnabled()) {
                return light;
            }
            return dark;
        }

        public static SearchBarColor getDark() {
            return dark;
        }

        public static SearchBarColor getLight() {
            return light;
        }
    }
}
