package file.engine.dllInterface;

import java.nio.file.Path;

public enum FileMonitor {
    INSTANCE;

    static {
        System.load(Path.of("user/fileMonitor.dll").toAbsolutePath().toString());
    }

    /**
     * 开始监控文件变化，该函数将会阻塞，直到stop_monitor被调用
     *
     * @param path 磁盘路径，C:\
     */
    public native void monitor(String path);

    /**
     * 停止监控文件变化
     *
     * @param path 磁盘路径
     */
    public native void stop_monitor(String path);

    /**
     * 监控是否已经停止或未开始
     *
     * @param path 磁盘路径
     * @return true如果已经停止
     */
    public native boolean is_monitor_stopped(String path);

    /**
     * 获取一个刚才新增的文件记录
     *
     * @return 文件路径
     */
    public native String pop_add_file();

    /**
     * 获取一个刚才删除的文件记录
     *
     * @return 文件路径
     */
    public native String pop_del_file();

    /**
     * 设置是否在程序退出时删除NTFS文件系统的USN日志
     *
     * @param path 磁盘盘符
     */
    public native void delete_usn_on_exit(String path);
}
