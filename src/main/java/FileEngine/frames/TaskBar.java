package FileEngine.frames;

import FileEngine.IsDebug;
import FileEngine.taskHandler.TaskUtil;
import FileEngine.taskHandler.Task;
import FileEngine.taskHandler.TaskHandler;
import FileEngine.taskHandler.impl.taskbar.ShowTaskBarIconTask;
import FileEngine.taskHandler.impl.taskbar.ShowTaskBarMessageTask;
import FileEngine.taskHandler.impl.stop.CloseTask;
import FileEngine.taskHandler.impl.stop.RestartTask;
import FileEngine.taskHandler.impl.frame.settingsFrame.ShowSettingsFrameTask;
import FileEngine.threadPool.CachedThreadPool;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.net.URL;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;


public class TaskBar {
    private TrayIcon trayIcon = null;
    private SystemTray systemTray;
    private final ConcurrentLinkedQueue<MessageStruct> messageQueue = new ConcurrentLinkedQueue<>();

    private static volatile TaskBar INSTANCE = null;


    private static class MessageStruct {
        private final String caption;
        private final String message;

        private MessageStruct(String caption, String message) {
            this.caption = caption;
            this.message = message;
        }
    }

    private TaskBar() {
        startShowMessageThread();
        showTaskBar();
    }

    private void startShowMessageThread() {
        CachedThreadPool.getInstance().executeTask(() -> {
            try {
                MessageStruct message;
                while (TaskUtil.getInstance().isNotMainExit()) {
                    message = messageQueue.poll();
                    if (message != null) {
                        showMessageOnTrayIcon(message.caption, message.message);
                        TimeUnit.SECONDS.sleep(5);
                    }
                    TimeUnit.MILLISECONDS.sleep(200);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    public static TaskBar getInstance() {
        initInstance();
        return INSTANCE;
    }

    private static void initInstance() {
        if (INSTANCE == null) {
            synchronized (TaskBar.class) {
                if (INSTANCE == null) {
                    INSTANCE = new TaskBar();
                }
            }
        }
    }

    private void showTaskBar() {
        SettingsFrame settingsFrame = SettingsFrame.getInstance();
        // 判断是否支持系统托盘
        if (SystemTray.isSupported()) {
            Image image;
            URL icon;
            icon = getClass().getResource("/icons/taskbar.png");
            image = new ImageIcon(icon).getImage();
            systemTray = SystemTray.getSystemTray();
            // 创建托盘图标
            trayIcon = new TrayIcon(image);
            // 添加工具提示文本
            if (IsDebug.isDebug()) {
                trayIcon.setToolTip("File-Engine(Debug)");
            } else {
                trayIcon.setToolTip("File-Engine");
            }
            // 创建弹出菜单
            PopupMenu popupMenu = new PopupMenu();

            MenuItem settings = new MenuItem("Settings");
            settings.addActionListener(e -> {
                TaskUtil.getInstance().putTask(new ShowSettingsFrameTask());
            });
            MenuItem restartProc = new MenuItem("Restart");
            restartProc.addActionListener(e -> restart());
            MenuItem close = new MenuItem("Exit");
            close.addActionListener(e -> closeAndExit());

            popupMenu.add(settings);
            popupMenu.add(restartProc);
            popupMenu.add(close);

            // 为托盘图标加弹出菜弹
            trayIcon.setPopupMenu(popupMenu);
            trayIcon.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    if (MouseEvent.BUTTON1 == e.getButton() && !settingsFrame.isSettingsFrameVisible()) {
                        TaskUtil.getInstance().putTask(new ShowSettingsFrameTask());
                    }
                }
            });
            // 获得系统托盘对象
            try {
                // 为系统托盘加托盘图标
                systemTray.add(trayIcon);
            } catch (Exception ignored) {
            }
        }
    }

    private void closeAndExit() {
        TaskUtil taskUtil = TaskUtil.getInstance();
        taskUtil.putTask(new CloseTask());
        systemTray.remove(trayIcon);
    }

    public void restart() {
        TaskUtil taskUtil = TaskUtil.getInstance();
        taskUtil.putTask(new RestartTask());
        systemTray.remove(trayIcon);
    }

    private void showMessage(String caption, String message) {
        messageQueue.add(new MessageStruct(caption, message));
    }

    private void showMessageOnTrayIcon(String caption, String message) {
        if (trayIcon != null) {
            trayIcon.displayMessage(caption, message, TrayIcon.MessageType.INFO);
        }
    }

    public static void registerTaskHandler() {
        TaskUtil.getInstance().registerTaskHandler(ShowTaskBarMessageTask.class, new TaskHandler() {
            @Override
            public void todo(Task task) {
                ShowTaskBarMessageTask showTaskBarMessageTask = (ShowTaskBarMessageTask) task;
                getInstance().showMessage(showTaskBarMessageTask.caption, showTaskBarMessageTask.message);
            }
        });

        TaskUtil.getInstance().registerTaskHandler(ShowTaskBarIconTask.class, new TaskHandler() {
            @Override
            public void todo(Task task) {
                getInstance();
            }
        });
    }
}
