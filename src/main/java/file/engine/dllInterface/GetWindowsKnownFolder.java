package file.engine.dllInterface;

import java.nio.file.Path;

public enum GetWindowsKnownFolder {
    INSTANCE;

    static {
        System.load(Path.of("user/getWindowsKnownFolder.dll").toAbsolutePath().toString());
    }

    /**
     * 根据guid获取文件夹路径
     *
     * @param guid guid
     * @return 文件夹路径
     * <a href="https://learn.microsoft.com/en-us/windows/win32/api/shlobj_core/nf-shlobj_core-shgetknownfolderpath">SHGetKnownFolderPath function (shlobj_core.h)</a>
     * <a href="https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid">knownfolderid</a>
     */
    public native String getKnownFolder(String guid);
}
