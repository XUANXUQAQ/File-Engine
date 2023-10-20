package file.engine.dllInterface;

import java.nio.file.Path;

public enum GetHandle {
    INSTANCE;

    static {
        System.load(Path.of("user/getHandle.dll").toAbsolutePath().toString());
    }

    /**
     * 初始化
     */
    public native void start();

    /**
     * 停止
     */
    public native void stop();

    /**
     * 是否需要使搜索框切换到贴靠模式
     *
     * @return true如果当资源管理器获得了窗口焦点
     */
    public native boolean changeToAttach();

    /**
     * 是否需要使搜索框切换到正常模式
     *
     * @return true如果资源管理器失去焦点或搜索框被关闭
     */
    public native boolean changeToNormal();

    /**
     * 获取当前贴靠的资源管理器左上角X坐标
     *
     * @return X
     */
    public native long getExplorerX();

    /**
     * 获取当前贴靠的资源管理器左上角Y坐标
     *
     * @return Y
     */
    public native long getExplorerY();

    /**
     * 获取资源管理器窗口的宽度
     *
     * @return width
     */
    public native long getExplorerWidth();

    /**
     * 获取资源管理器窗口的高度
     *
     * @return height
     */
    public native long getExplorerHeight();

    /**
     * 获取资源管理器当前打开的文件夹的路径
     *
     * @return 文件夹路径
     */
    public native String getExplorerPath();

    /**
     * 判断是否为对话框，比如选择文件弹出窗口
     *
     * @return true如果是对话框
     */
    public native boolean isDialogWindow();

    @Deprecated
    public native int getToolBarX();

    @Deprecated
    public native int getToolBarY();

    /**
     * 判断键盘某个键是否被点击
     *
     * @param vk_key keyCode
     * @return true如果被点击
     */
    public native boolean isKeyPressed(int vk_key);

    /**
     * 判断当前是否有全屏任务，如游戏全屏，全屏看电影等
     *
     * @return true如果有
     */
    @SuppressWarnings("BooleanMethodIsAlwaysInverted")
    public native boolean isForegroundFullscreen();

    /**
     * 设置资源管理器上方文件夹路径，用于快速跳转
     *
     * @param path 文件夹路径
     */
    public native void setEditPath(String path, String fileName);

    /**
     * 将搜索框设置为前台窗口
     */
    public native boolean bringWindowToTop();
}
