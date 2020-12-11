package FileEngine.configs;

public class Enums {

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
}
