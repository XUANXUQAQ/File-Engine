package file.engine.configs;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class Constants {
    public static final String version;

    public static final int ALL_TABLE_NUM = 40;

    public static final int MIN_FRAME_VISIBLE_TIME = 500;

    public static final int CLOSE_DATABASE_TIMEOUT_MILLS = 5 * 60 * 1000;

    public static final String FILE_NAME = "File-Engine.jar";

    public static final String LAUNCH_WRAPPER_NAME = "File-Engine.exe";

    public static final int API_VERSION = 6;

    public static final int[] COMPATIBLE_API_VERSIONS = {3, 4, 5, 6};

    public static final String DEFAULT_SWING_THEME = "MaterialLighter";

    static {
        Properties properties = new Properties();
        try (InputStream projectInfo = Constants.class.getResourceAsStream("/project-info.properties")) {
            properties.load(projectInfo);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        version = properties.getProperty("project.version");
    }

    public static class Enums {

        public static class DefaultColors {
            public static final int DEFAULT_LABEL_COLOR = 0xff9933;
            public static final int DEFAULT_WINDOW_BACKGROUND_COLOR = 0xf6f6f6;
            public static final int DEFAULT_BORDER_COLOR = 0xf6f6f6;
            public static final int DEFAULT_FONT_COLOR = 0;
            public static final int DEFAULT_FONT_COLOR_WITH_COVERAGE = 0xff3333;
            public static final int DEFAULT_SEARCHBAR_COLOR = 0xf6f6f6;
            public static final int DEFAULT_SEARCHBAR_FONT_COLOR = 0;
        }

        public enum DatabaseStatus {
            NORMAL, VACUUM, MANUAL_UPDATE
        }

        public enum DownloadStatus {
            DOWNLOAD_DONE, DOWNLOAD_ERROR, DOWNLOAD_DOWNLOADING, DOWNLOAD_INTERRUPTED, DOWNLOAD_NO_TASK
        }

        public enum ShowingSearchBarMode {
            NORMAL_SHOWING, EXPLORER_ATTACH
        }

        public enum RunningMode {
            NORMAL_MODE, COMMAND_MODE, PLUGIN_MODE
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
            Nord, OneDark, MaterialDarker, MaterialLighter
        }

        public enum BorderType {
            EMPTY, AROUND, FULL
        }

        public static class SearchCase {
            public static final String D = "d";
            public static final String F = "f";
            public static final String FULL = "full";
            public static final String CASE = "case";
        }
    }
}
