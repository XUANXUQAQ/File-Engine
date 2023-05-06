package file.engine.dllInterface;

import java.nio.file.Path;

public enum IsLocalDisk {
    INSTANCE;

    static {
        System.load(Path.of("user/isLocalDisk.dll").toAbsolutePath().toString());
    }

    /**
     * 判断是否是本地磁盘
     *
     * @param path 磁盘路径
     * @return true如果是本地磁盘，false如果是U盘或移动硬盘
     */
    public native boolean isLocalDisk(String path);

    /**
     * 磁盘文件系统是否为NTFS
     *
     * @param disk 磁盘路径
     * @return true如果是NTFS
     */
    public native boolean isDiskNTFS(String disk);
}
