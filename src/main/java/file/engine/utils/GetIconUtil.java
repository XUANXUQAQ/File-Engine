package file.engine.utils;

import file.engine.event.handler.EventManagement;
import file.engine.utils.system.properties.IsDebug;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.File;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class GetIconUtil {
    private static final int MAX_WORKING_QUEUE_SIZE = 200;
    private static final int MAX_CONSUMER_THREAD_NUM = 4;
    private static final FileSystemView FILE_SYSTEM_VIEW = FileSystemView.getFileSystemView();
    private final ConcurrentHashMap<String, ImageIcon> iconMap = new ConcurrentHashMap<>();
    private final LinkedBlockingQueue<Task> workingQueue = new LinkedBlockingQueue<>(MAX_WORKING_QUEUE_SIZE);
    private final ExecutorService iconTaskConsumer = Executors.newFixedThreadPool(MAX_CONSUMER_THREAD_NUM);

    private static volatile GetIconUtil INSTANCE = null;

    private GetIconUtil() {
        initIconCache();
        startWorkingThread();
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
        iconMap.put("dllImageIcon", Objects.requireNonNull((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user", "fileMonitor.dll"))));
        iconMap.put("folderImageIcon", Objects.requireNonNull((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user"))));
        iconMap.put("txtImageIcon", Objects.requireNonNull((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(new File("user\\cmds.txt"))));
        iconMap.put("blankIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/blank.png"))));
        iconMap.put("recycleBin", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/recyclebin.png"))));
        iconMap.put("updateIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/update.png"))));
        iconMap.put("helpIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/help.png"))));
        iconMap.put("completeIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/complete.png"))));
        iconMap.put("loadingIcon", new ImageIcon(Objects.requireNonNull(GetIconUtil.class.getResource("/icons/loading.gif"))));
    }

    public ImageIcon getCommandIcon(String commandName, int width, int height) {
        if (commandName == null || commandName.isEmpty()) {
            if (IsDebug.isDebug()) {
                System.err.println("Command is empty");
            }
            return null;
        }
        switch (commandName) {
            case "clearbin":
                return changeIcon(iconMap.get("recycleBin"), width, height);
            case "update":
            case "clearUpdate":
                return changeIcon(iconMap.get("updateIcon"), width, height);
            case "help":
                return changeIcon(iconMap.get("helpIcon"), width, height);
            case "version":
                return changeIcon(iconMap.get("blankIcon"), width, height);
            default:
                return null;
        }
    }

    public ImageIcon getBigIcon(String pathOrKey, int width, int height) {
        final ImageIcon[] image = new ImageIcon[1];
        getBigIcon(pathOrKey, width, height, null, (img, isTimeout) -> image[0] = img);
        return image[0];
    }

    public void getBigIcon(String pathOrKey,
                           int width,
                           int height,
                           Consumer<ImageIcon> timeoutCallback,
                           @NonNull BiConsumer<ImageIcon, Boolean> normalCallback) {
        if (pathOrKey == null || pathOrKey.isEmpty()) {
            normalCallback.accept(changeIcon(iconMap.get("blankIcon"), width, height), false);
            return;
        }
        if (iconMap.containsKey(pathOrKey)) {
            normalCallback.accept(changeIcon(iconMap.get(pathOrKey), width, height), false);
            return;
        }
        File f = new File(pathOrKey);
        if (!f.exists()) {
            normalCallback.accept(changeIcon(iconMap.get("blankIcon"), width, height), false);
            return;
        }
        pathOrKey = pathOrKey.toLowerCase();
        //已保存的常量图标
        if (pathOrKey.endsWith(".dll") || pathOrKey.endsWith(".sys")) {
            normalCallback.accept(changeIcon(iconMap.get("dllImageIcon"), width, height), false);
            return;
        }
        if (pathOrKey.endsWith(".txt")) {
            normalCallback.accept(changeIcon(iconMap.get("txtImageIcon"), width, height), false);
            return;
        }
        //检测是否为文件夹
        if (f.isDirectory()) {
            normalCallback.accept(changeIcon(iconMap.get("folderImageIcon"), width, height), false);
            return;
        }
        Task task = new Task(f, width, height);
        if (workingQueue.offer(task)) {
            final long start = System.currentTimeMillis();
            final long timeout = timeoutCallback == null ? 10_000 : 50; // 最长等待时间
            while (!task.isDone) {
                if (System.currentTimeMillis() - start > timeout) {
                    if (timeoutCallback != null) {
                        task.timeoutCallBack = timeoutCallback;
                        workingQueue.offer(task);
                    }
                    normalCallback.accept(changeIcon(iconMap.get("blankIcon"), width, height), true);
                    return;
                }
                Thread.onSpinWait();
            }
            //图标获取完成
            normalCallback.accept(task.icon, false);
        }
    }

    private void startWorkingThread() {
        for (int i = 0; i < MAX_CONSUMER_THREAD_NUM; i++) {
            iconTaskConsumer.execute(() -> {
                EventManagement eventManagement = EventManagement.getInstance();
                try {
                    while (eventManagement.notMainExit()) {
                        var take = workingQueue.take();
                        if (take.timeoutCallBack != null) {
                            if (!take.isDone) {
                                take.icon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(take.path), take.width, take.height);
                            }
                            take.timeoutCallBack.accept(take.icon);
                        } else {
                            take.icon = changeIcon((ImageIcon) FILE_SYSTEM_VIEW.getSystemIcon(take.path), take.width, take.height);
                            take.isDone = true;
                        }
                    }
                } catch (InterruptedException ignored) {
                    // ignore InterruptedException
                }
            });
        }
    }

    public void stopWorkingThread() {
        iconTaskConsumer.shutdownNow();
    }

    @RequiredArgsConstructor
    private static class Task {
        final File path;
        volatile boolean isDone;
        volatile ImageIcon icon;
        final int width;
        final int height;
        volatile Consumer<ImageIcon> timeoutCallBack;
    }
}
