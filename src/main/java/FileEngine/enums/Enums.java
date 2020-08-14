package FileEngine.enums;

public class Enums {
    public static class DownloadStatus {
        public static final int DOWNLOAD_DONE = 0;
        public static final int DOWNLOAD_ERROR = 1;
        public static final int DOWNLOAD_DOWNLOADING = 2;
        public static final int DOWNLOAD_INTERRUPTED = 3;
        public static final int DOWNLOAD_NO_TASK = 4;
    }

    public static class ShowingSearchBarMode {
        public static final int NORMAL_SHOWING = 5;
        public static final int EXPLORER_ATTACH = 6;
    }

    public static class runningMode {
        public static final int NORMAL_MODE = 7;
        public static final int COMMAND_MODE = 8;
        public static final int PLUGIN_MODE = 9;
    }
}
