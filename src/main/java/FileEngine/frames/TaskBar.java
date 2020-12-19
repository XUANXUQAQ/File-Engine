package FileEngine.frames;

import FileEngine.IsDebug;
import FileEngine.eventHandler.EventUtil;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventHandler;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarIconEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarMessageEvent;
import FileEngine.eventHandler.impl.stop.CloseEvent;
import FileEngine.eventHandler.impl.stop.RestartEvent;
import FileEngine.eventHandler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
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
                while (EventUtil.getInstance().isNotMainExit()) {
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
                EventUtil.getInstance().putTask(new ShowSettingsFrameEvent());
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
                        EventUtil.getInstance().putTask(new ShowSettingsFrameEvent());
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
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.putTask(new CloseEvent());
        systemTray.remove(trayIcon);
    }

    public void restart() {
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.putTask(new RestartEvent());
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

    public static void registerEventHandler() {
        EventUtil.getInstance().register(ShowTaskBarMessageEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                ShowTaskBarMessageEvent showTaskBarMessageTask = (ShowTaskBarMessageEvent) event;
                getInstance().showMessage(showTaskBarMessageTask.caption, showTaskBarMessageTask.message);
            }
        });

        EventUtil.getInstance().register(ShowTaskBarIconEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance();
            }
        });
    }
}
