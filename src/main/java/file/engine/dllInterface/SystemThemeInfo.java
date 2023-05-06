package file.engine.dllInterface;

import java.nio.file.Path;

public enum SystemThemeInfo {
    INSTANCE;

    static {
        System.load(Path.of("user/systemThemeInfo.dll").toAbsolutePath().toString());
    }

    /**
     * 判断Windows系统是否已开启暗色模式
     *
     * @return true如果是暗色模式
     */
    public native boolean isDarkThemeEnabled();
}
