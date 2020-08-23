package FileEngine.modesAndStatus;

public class Enums {

    public static class ShowingSearchBarMode {
        public static final int NORMAL_SHOWING = 0;
        public static final int EXPLORER_ATTACH = 1;
    }

    public static class RunningMode {
        public static final int NORMAL_MODE = 2;
        public static final int COMMAND_MODE = 3;
        public static final int PLUGIN_MODE = 4;
    }

    public static class ProxyType {
        public static final int PROXY_HTTP = 5;
        public static final int PROXY_SOCKS = 6;
        public static final int PROXY_DIRECT = 7;
    }

    public enum DownloadStatus {
        DOWNLOAD_DONE, DOWNLOAD_ERROR, DOWNLOAD_DOWNLOADING, DOWNLOAD_INTERRUPTED, DOWNLOAD_NO_TASK
    }
}
