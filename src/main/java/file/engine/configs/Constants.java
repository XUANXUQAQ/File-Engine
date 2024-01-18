package file.engine.configs;

import file.engine.dllInterface.SystemThemeInfo;
import file.engine.utils.system.properties.IsDebug;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.io.InputStream;
import java.util.Properties;

import static file.engine.utils.BeanUtil.copyMatchingFields;

/**
 * 项目使用到的常量
 */
@Slf4j
public class Constants {
    public static String version;

    public static String buildVersion;

    // 搜索框最短应该显示的时间，单位毫秒
    public static final int MIN_FRAME_VISIBLE_TIME = 500;

    // 启动文件名
    public static final String FILE_NAME = "File-Engine.jar";

    // 启动器文件（进程）名
    public static final String LAUNCH_WRAPPER_NAME = "File-Engine.exe";

    public static final String FILE_ENGINE_CORE_CMD_NAME = "startup.txt";

    // 插件API版本
    public static final int API_VERSION = 8;

    // 兼容的插件API版本
    public static final int[] COMPATIBLE_API_VERSIONS = {3, 4, 5, 6, 7, API_VERSION};

    // 默认swing主题
    public static final String DEFAULT_SWING_THEME = "MaterialLighter";

    public static final String FILE_ENGINE_CORE_DIR = "core/";

    static {
        version = "0";
        buildVersion = "Debug";
        if (!IsDebug.isDebug()) {
            /*
             * 读取maven自动生成的版本信息
             */
            Properties properties = new Properties();
            try (InputStream projectInfo = Constants.class.getResourceAsStream("/project-info.properties")) {
                properties.load(projectInfo);
                version = properties.getProperty("project.version");
                buildVersion = properties.getProperty("project.build.version");
            } catch (Exception e) {
                log.error("error: {}", e.getMessage(), e);
            }
        }
    }

    private Constants() {
        throw new RuntimeException("not allowed");
    }

    public static class Enums {

        /**
         * 数据库运行状态
         * NORMAL：正常
         * _TEMP：正在搜索中，已经切换到临时数据库
         * VACUUM：正在整理数据库
         * MANUAL_UPDATE：正在搜索中，未切换到临时数据库
         */
        public enum DatabaseStatus {
            NORMAL, _TEMP, VACUUM, MANUAL_UPDATE
        }

        /**
         * 文件下载状态
         */
        public enum DownloadStatus {
            DOWNLOAD_DONE, DOWNLOAD_ERROR, DOWNLOAD_DOWNLOADING, DOWNLOAD_INTERRUPTED, DOWNLOAD_NO_TASK
        }

        /**
         * 搜索框显示状态
         * NORMAL_SHOWING：正常显示
         * EXPLORER_ATTACH：贴靠在资源管理器右下方
         */
        public enum ShowingSearchBarMode {
            NORMAL_SHOWING, EXPLORER_ATTACH
        }

        /**
         * 设置代理，HTTP，SOCKS，直连
         */
        public static class ProxyType {
            public static final int PROXY_HTTP = 0x100;
            public static final int PROXY_SOCKS = 0x200;
            public static final int PROXY_DIRECT = 0x300;
        }

        /**
         * 所有可用的Swing主题
         */
        public enum SwingThemes {
            CoreFlatDarculaLaf, CoreFlatDarkLaf, CoreFlatLightLaf, CoreFlatIntelliJLaf,
            Arc, ArcDark, ArcDarkOrange, Carbon,
            CyanLight, DarkFlat, DarkPurple, Dracula,
            Gray, LightFlat, MaterialDesignDark, Monocai,
            Nord, OneDark, MaterialDarker, MaterialLighter, SolarizedDark, SolarizedLight,
            Spacegray, Vuesion, XcodeDark, SystemDefault
        }

        /**
         * 搜索框边框类型
         */
        public enum BorderType {
            EMPTY, AROUND, FULL
        }
    }

    /**
     * 默认亮色主题以及暗色主题
     */
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
            int DEFAULT_BORDER_COLOR = 0x999999;
            int DEFAULT_FONT_COLOR = 0;
            int DEFAULT_FONT_COLOR_WITH_COVERAGE = 0xff3333;
            int DEFAULT_SEARCHBAR_COLOR = 0xf6f6f6;
            int DEFAULT_SEARCHBAR_FONT_COLOR = 0;
        }

        @SuppressWarnings("unused")
        private static class Dark {
            int DEFAULT_LABEL_COLOR = 0x9999ff;
            int DEFAULT_WINDOW_BACKGROUND_COLOR = 0x333333;
            int DEFAULT_BORDER_COLOR = 0x999999;
            int DEFAULT_FONT_COLOR = 0xffffff;
            int DEFAULT_FONT_COLOR_WITH_COVERAGE = 0xff3333;
            int DEFAULT_SEARCHBAR_COLOR = 0x333333;
            int DEFAULT_SEARCHBAR_FONT_COLOR = 0xffffff;
        }

        public static SearchBarColor getDefaultSearchBarColor() {
            if (SystemThemeInfo.INSTANCE.isDarkThemeEnabled()) {
                return dark;
            }
            return light;
        }

        public static SearchBarColor getDark() {
            return dark;
        }

        public static SearchBarColor getLight() {
            return light;
        }
    }
}
