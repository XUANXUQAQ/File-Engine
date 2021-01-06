package FileEngine.frames;

import FileEngine.IsDebug;
import FileEngine.eventHandler.EventUtil;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventHandler;
import FileEngine.eventHandler.impl.taskbar.HideTrayIconEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarIconEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarMessageEvent;
import FileEngine.eventHandler.impl.stop.CloseEvent;
import FileEngine.eventHandler.impl.stop.RestartEvent;
import FileEngine.eventHandler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import FileEngine.utils.CachedThreadPoolUtil;

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
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventUtil eventUtil = EventUtil.getInstance();
                MessageStruct message;
                while (eventUtil.isNotMainExit()) {
                    message = messageQueue.poll();
                    if (message != null) {
                        showMessageOnTrayIcon(message.caption, message.message);
                        TimeUnit.SECONDS.sleep(5);
                    }
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    public static TaskBar getInstance() {
        if (INSTANCE == null) {
            synchronized (TaskBar.class) {
                if (INSTANCE == null) {
                    INSTANCE = new TaskBar();
                }
            }
        }
        return INSTANCE;
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
            settings.addActionListener(e -> EventUtil.getInstance().putEvent(new ShowSettingsFrameEvent()));
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
                        EventUtil.getInstance().putEvent(new ShowSettingsFrameEvent());
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
        eventUtil.putEvent(new CloseEvent());
    }

    private void restart() {
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.putEvent(new RestartEvent());
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
        EventUtil eventUtil = EventUtil.getInstance();
        eventUtil.register(ShowTaskBarMessageEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                ShowTaskBarMessageEvent showTaskBarMessageTask = (ShowTaskBarMessageEvent) event;
                getInstance().showMessage(showTaskBarMessageTask.caption, showTaskBarMessageTask.message);
            }
        });

        eventUtil.register(ShowTaskBarIconEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance();
            }
        });

        eventUtil.register(HideTrayIconEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                TaskBar taskBar = getInstance();
                taskBar.systemTray.remove(taskBar.trayIcon);
            }
        });
    }
}
