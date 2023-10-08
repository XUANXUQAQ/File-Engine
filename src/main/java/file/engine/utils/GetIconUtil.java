package file.engine.utils;

import file.engine.configs.AllConfigs;
import file.engine.dllInterface.icon.JIconExtract;
import file.engine.event.handler.EventManagement;
import file.engine.utils.system.properties.IsDebug;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class GetIconUtil {
    private static final int MAX_CONSUMER_THREAD_NUM = 4;
    private static final FileSystemView FILE_SYSTEM_VIEW = FileSystemView.getFileSystemView();
    private static final Task EMPTY_TASK = new Task(null, 0, 0);
    private final ConcurrentHashMap<String, ImageIcon> constantIconMap = new ConcurrentHashMap<>();
    private final SynchronousQueue<Task> workingQueue = new SynchronousQueue<>();
    private final ConcurrentHashMap<String, IconCache> cacheIconMap = new ConcurrentHashMap<>();

    private static volatile GetIconUtil INSTANCE = null;

    private record IconCache(
            long cacheCreateTime,
            ImageIcon icon
    ) {
    }

    private GetIconUtil() {
        initIconCache();
        startWorkingThread();
        clearIconCacheThread();
    }

    public static GetIconUtil getInstance() {
        if (INSTANCE == null) {
            synchronized (GetIconUtil.class) {
                if (INSTANCE == null) {
                    INSTANCE = new GetIconUtil();
                }
            }
        }
        return INSTANCE;
    }

    public ImageIcon changeIcon(ImageIcon icon, int width, int height) {
        if (icon == null) {
            return null;
        }
        Image image = icon.getImage().getScaledInstance(width, height, Image.SCALE_FAST);
        return new ImageIcon(image);
    }

    private void initIconCache() {
        constantIconMap.put("dllImageIcon", Objects.requireNonNull((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user", "fileMonitor.dll"))));
        constantIconMap.put("folderImageIcon", Objects.requireNonNull((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user"))));
        constantIconMap.put("txtImageIcon", Objects.requireNonNull((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user\\cmds.txt"))));
        constantIconMap.put("blankIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/blank.png"))));
        constantIconMap.put("recycleBin", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/recyclebin.png"))));
        constantIconMap.put("updateIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/update.png"))));
        constantIconMap.put("helpIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/help.png"))));
        constantIconMap.put("completeIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/complete.png"))));
        constantIconMap.put("loadingIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/loading.gif"))));
    }

    public ImageIcon getCommandIcon(String commandName, int width, int height) {
        if (commandName == null || commandName.isEmpty()) {
            if (IsDebug.isDebug()) {
                System.err.println("Command is empty");
            }
            return null;
        }
        return switch (commandName) {
            case "clearbin" -> changeIcon(constantIconMap.get("recycleBin"), width, height);
            case "update", "clearUpdate" -> changeIcon(constantIconMap.get("updateIcon"), width, height);
            case "help" -> changeIcon(constantIconMap.get("helpIcon"), width, height);
            case "version" -> changeIcon(constantIconMap.get("blankIcon"), width, height);
            default -> null;
        };
    }

    public ImageIcon getBigIcon(String pathOrKey, int width, int height) {
        final ImageIcon[] image = new ImageIcon[1];
        getBigIcon(pathOrKey, width, height, null, (img, isTimeout) -> image[0] = img);
        return image[0];
    }

    public void clearIconCacheThread() {
        Runnable clearCacheFunc = () -> {
            var eventManagement = EventManagement.getInstance();
            var allConfigs = AllConfigs.getInstance();
            long startCheck = System.currentTimeMillis();
            while (eventManagement.notMainExit()) {
                long clearIconCacheTimeoutInMills = allConfigs
                        .getConfigEntity()
                        .getAdvancedConfigEntity()
                        .getClearIconCacheTimeoutInMills();
                if (System.currentTimeMillis() - startCheck > clearIconCacheTimeoutInMills) {
                    startCheck = System.currentTimeMillis();
                    ArrayList<String> keyToRemove = new ArrayList<>();
                    cacheIconMap.forEach((path, iconCache) -> {
                        if (System.currentTimeMillis() - iconCache.cacheCreateTime > clearIconCacheTimeoutInMills) {
                            keyToRemove.add(path);
                        }
                    });
                    keyToRemove.forEach(cacheIconMap::remove);
                }
                try {
                    TimeUnit.MILLISECONDS.sleep(100);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        };
        Thread.startVirtualThread(clearCacheFunc);
    }

    public void getBigIcon(String pathOrKey,
                           int width,
                           int height,
                           Consumer<ImageIcon> timeoutCallback,
                           @NonNull BiConsumer<ImageIcon, Boolean> normalCallback) {
        if (pathOrKey == null || pathOrKey.isEmpty()) {
            normalCallback.accept(changeIcon(constantIconMap.get("blankIcon"), width, height), false);
            return;
        }
        if (constantIconMap.containsKey(pathOrKey)) {
            normalCallback.accept(changeIcon(constantIconMap.get(pathOrKey), width, height), false);
            return;
        }
        File f = new File(pathOrKey);
        if (!f.exists()) {
            normalCallback.accept(changeIcon(constantIconMap.get("blankIcon"), width, height), false);
            return;
        }
        //已保存的常量图标
        var toLower = pathOrKey.toLowerCase();
        if (toLower.endsWith(".dll") || toLower.endsWith(".sys")) {
            normalCallback.accept(changeIcon(constantIconMap.get("dllImageIcon"), width, height), false);
            return;
        }
        if (toLower.endsWith(".txt")) {
            normalCallback.accept(changeIcon(constantIconMap.get("txtImageIcon"), width, height), false);
            return;
        }
        //检测是否为文件夹
        if (f.isDirectory()) {
            normalCallback.accept(changeIcon(constantIconMap.get("folderImageIcon"), width, height), false);
            return;
        }
        var cachedIcon = cacheIconMap.get(pathOrKey);
        if (cachedIcon != null) {
            normalCallback.accept(changeIcon(cachedIcon.icon(), width, height), false);
            return;
        }
        Task task = new Task(f, width, height);
        if (!workingQueue.offer(task)) {
            throw new RuntimeException("add to working queue failed");
        }
        final long startMills = System.currentTimeMillis();
        final long timeoutThreshold = timeoutCallback == null ? 10_000 : 200; // 最长等待时间
        while (!task.isDone) {
            if (System.currentTimeMillis() - startMills > timeoutThreshold) {
                // 超时，添加timeoutCallback后退出
                normalCallback.accept(changeIcon(constantIconMap.get("blankIcon"), width, height), true);
                if (timeoutCallback != null) {
                    task.timeoutCallBack = timeoutCallback;
                    workingQueue.offer(task);
                }
                return;
            }
            try {
                TimeUnit.MILLISECONDS.sleep(1);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        //图标获取完成
        normalCallback.accept(task.icon, false);
    }

    private void startWorkingThread() {
        for (int i = 0; i < MAX_CONSUMER_THREAD_NUM; i++) {
            Thread.startVirtualThread(this::consumeTaskFunc);
        }
    }

    @SneakyThrows
    private void consumeTaskFunc() {
        var eventManagement = EventManagement.getInstance();
        while (eventManagement.notMainExit()) {
            var task = workingQueue.take();
            if (task == EMPTY_TASK) {
                return;
            }
            if (task.timeoutCallBack != null) {
                ImageIcon icon;
                if (task.isDone) {
                    icon = task.icon;
                } else {
                    BufferedImage iconForFile = JIconExtract.getIconForFile(task.width, task.height, task.path);
                    if (iconForFile == null) {
                        icon = changeIcon(constantIconMap.get("blankIcon"), task.width, task.height);
                    } else {
                        icon = new ImageIcon(iconForFile);
                        cacheIconMap.put(task.path.getAbsolutePath(), new IconCache(System.currentTimeMillis(), icon));
                    }
//                     icon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(task.path), task.width, task.height);
//                    if (null != icon) {
//                        cacheIconMap.put(task.path.getAbsolutePath(), new IconCache(System.currentTimeMillis(), icon));
//                    } else {
//                        icon = changeIcon(constantIconMap.get("blankIcon"), task.width, task.height);
//                    }
                }
                task.timeoutCallBack.accept(icon);
            } else {
                BufferedImage iconForFile = JIconExtract.getIconForFile(task.width, task.height, task.path);
                if (iconForFile == null) {
                    task.icon = changeIcon(constantIconMap.get("blankIcon"), task.width, task.height);
                } else {
                    task.icon = new ImageIcon(iconForFile);
                    cacheIconMap.put(task.path.getAbsolutePath(), new IconCache(System.currentTimeMillis(), task.icon));
                }
//                task.icon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(task.path), task.width, task.height);
                task.isDone = true;
            }
        }
    }

    public void stopWorkingThread() {
        for (int i = 0; i < MAX_CONSUMER_THREAD_NUM; i++) {
            workingQueue.offer(EMPTY_TASK);
        }
    }

    @RequiredArgsConstructor
    private static class Task {
        final File path;
        volatile boolean isDone = false;
        volatile ImageIcon icon;
        final int width;
        final int height;
        volatile Consumer<ImageIcon> timeoutCallBack;
    }
}
