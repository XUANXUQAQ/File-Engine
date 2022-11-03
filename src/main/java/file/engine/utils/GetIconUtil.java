package file.engine.utils;

import file.engine.event.handler.EventManagement;
import file.engine.utils.system.properties.IsDebug;
import lombok.RequiredArgsConstructor;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.awt.*;
import java.io.File;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.function.Consumer;

/**
 * @author XUANXU
 */
public class GetIconUtil {
    private static final int MAX_WORKING_QUEUE_SIZE = 200;
    private static final FileSystemView FILE_SYSTEM_VIEW = FileSystemView.getFileSystemView();
    private final ConcurrentHashMap<String, ImageIcon> iconMap = new ConcurrentHashMap<>();
    private final LinkedBlockingQueue<Task> workingQueue = new LinkedBlockingQueue<>(MAX_WORKING_QUEUE_SIZE);
    private Thread workingThread;

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

    private ImageIcon getCommandIcon(String commandName, int width, int height) {
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

    public ImageIcon getBigIcon(String pathOrKey, int width, int height, Consumer<ImageIcon> timeoutCallback) {
        if (pathOrKey == null || pathOrKey.isEmpty()) {
            return changeIcon(iconMap.get("blankIcon"), width, height);
        }
        if (iconMap.containsKey(pathOrKey)) {
            return changeIcon(iconMap.get(pathOrKey), width, height);
        }
        ImageIcon commandIcon = getCommandIcon(pathOrKey, width, height);
        if (commandIcon != null) {
            return commandIcon;
        }
        pathOrKey = pathOrKey.toLowerCase();
        File f = new File(pathOrKey);
        if (f.exists()) {
            //已保存的常量图标
            if (pathOrKey.endsWith(".dll") || pathOrKey.endsWith(".sys")) {
                return changeIcon(iconMap.get("dllImageIcon"), width, height);
            }
            if (pathOrKey.endsWith(".txt")) {
                return changeIcon(iconMap.get("txtImageIcon"), width, height);
            }
            //检测是否为文件夹
            if (f.isDirectory()) {
                return changeIcon(iconMap.get("folderImageIcon"), width, height);
            }
            Task task = new Task(f, width, height);
            if (workingQueue.offer(task)) {
                final long start = System.currentTimeMillis();
                final long timeout; // 最长等待时间
                if (timeoutCallback == null) {
                    // 无callback
                    timeout = 10_000; // 延长超时时间到10s
                } else {
                    timeout = 50;
                }
                while (!task.isDone) {
                    if (System.currentTimeMillis() - start > timeout) {
                        if (timeoutCallback != null) {
                            task.timeoutCallBack = timeoutCallback;
                            workingQueue.offer(task);
                        }
                        return changeIcon(iconMap.get("blankIcon"), width, height);
                    }
                    Thread.onSpinWait();
                }
                //图标获取完成
                return task.icon;
            }
        }
        return changeIcon(iconMap.get("blankIcon"), width, height);
    }

    private void startWorkingThread() {
        workingThread = new Thread(() -> {
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
        workingThread.start();
    }

    public void stopWorkingThread() {
        if (workingThread != null) {
            workingThread.interrupt();
        }
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
