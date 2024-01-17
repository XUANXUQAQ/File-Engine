package file.engine.dllInterface;

import java.nio.file.Path;

public enum PathMatcher {
    INSTANCE;

    static {
        System.load(Path.of("user/pathMatcher.dll").toAbsolutePath().toString());
    }

    public native String[] match(String sql,
                                 String dbPath,
                                 String[] searchCase,
                                 boolean isIgnoreCase,
                                 String searchText,
                                 String[] keywords,
                                 String[] keywordsLowerCase,
                                 boolean[] isKeywordPath,
                                 int maxResultNumber);

    public native void openConnection(String dbPath);

    public native void closeConnections();
}
