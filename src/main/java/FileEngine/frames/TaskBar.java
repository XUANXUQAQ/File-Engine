package FileEngine.frames;

import FileEngine.IsDebug;
import FileEngine.annotation.EventRegister;
import FileEngine.dllInterface.GetHandle;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventManagement;
import FileEngine.eventHandler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import FileEngine.eventHandler.impl.stop.CloseEvent;
import FileEngine.eventHandler.impl.stop.RestartEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTaskBarMessageEvent;
import FileEngine.eventHandler.impl.taskbar.ShowTrayIconEvent;
import FileEngine.utils.CachedThreadPoolUtil;
import FileEngine.utils.TranslateUtil;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.net.URL;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class TaskBar {
    private TrayIcon trayIcon = null;
    private SystemTray systemTray;
    private final ConcurrentLinkedQueue<MessageStruct> messageQueue = new ConcurrentLinkedQueue<>();
    private final AtomicBoolean isMessageClear = new AtomicBoolean(true);
    private volatile Event currentShowingMessageWithEvent = null;

    private static volatile TaskBar INSTANCE = null;

    private static class MessageStruct {
        private final String caption;
        private final String message;
        private final Event event;

        private MessageStruct(String caption, String message, Event event) {
            this.caption = caption;
            this.message = message;
            this.event = event;
        }
    }

    private TaskBar() {
        startShowMessageThread();
        showTaskBar();
        addActionListener();
    }

    private void startShowMessageThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                MessageStruct message;
                int count = 0;
                while (eventManagement.isNotMainExit()) {
                    if (isMessageClear.get()) {
                        currentShowingMessageWithEvent = null;
                        message = messageQueue.poll();
                        if (message != null) {
                            currentShowingMessageWithEvent = message.event;
                            showMessageOnTrayIcon(message.caption, message.message);
                            isMessageClear.set(false);
                            count = 0;
                        }
                    } else {
                        count++;
                    }
                    if (count > 100) {
                        isMessageClear.set(true);
                    }
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private static TaskBar getInstance() {
        if (INSTANCE == null) {
            synchronized (TaskBar.class) {
                if (INSTANCE == null) {
                    INSTANCE = new TaskBar();
                }
            }
        }
        return INSTANCE;
    }

    private void addActionListener() {
        if (trayIcon == null) {
            return;
        }
        EventManagement eventManagement = EventManagement.getInstance();
        trayIcon.addActionListener(e -> {
                    isMessageClear.set(true);
                    if (currentShowingMessageWithEvent != null) {
                        eventManagement.putEvent(currentShowingMessageWithEvent);
                    }
                }
        );
    }

    private void showTaskBar() {
        // 判断是否支持系统托盘
        if (SystemTray.isSupported()) {
            Image image;
            URL icon;
            icon = this.getClass().getResource("/icons/taskbar.png");
            image = new ImageIcon(icon).getImage();
            systemTray = SystemTray.getSystemTray();
            EventManagement eventManagement = EventManagement.getInstance();
            // 创建托盘图标
            trayIcon = new TrayIcon(image);
            // 添加工具提示文本
            if (IsDebug.isDebug()) {
                trayIcon.setToolTip("File-Engine(Debug)");
            } else {
                trayIcon.setToolTip("File-Engine," + TranslateUtil.getInstance().getTranslation("Double click to open settings"));
            }

            // 为托盘图标加弹出菜弹
            trayIcon.addMouseListener(new MouseAdapter() {
                JPopupMenu popupMenu;

                @Override
                public void mouseClicked(MouseEvent e) {
                    if (MouseEvent.BUTTON1 == e.getButton() && e.getClickCount() == 2) {
                        eventManagement.putEvent(new ShowSettingsFrameEvent());
                    }
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                    if (e.isPopupTrigger()) {
                        popupMenu = new JPopupMenu();
                        getPopupMenu(popupMenu);
                        popupMenu.setInvoker(popupMenu);
                        popupMenu.setVisible(true);
                        double dpi = GetHandle.INSTANCE.getDpi();
                        popupMenu.setLocation((int) (e.getX() / dpi), (int) ((e.getY() - popupMenu.getHeight()) / dpi));
                    }
                }
            });
            // 获得系统托盘对象
            try {
                // 为系统托盘加托盘图标
                systemTray.add(trayIcon);
            } catch (AWTException e) {
                e.printStackTrace();
            }
        }
    }

    private void getPopupMenu(JPopupMenu popupMenu) {
        // 创建弹出菜单
        EventManagement eventManagement = EventManagement.getInstance();
        TranslateUtil translateUtil = TranslateUtil.getInstance();
        JMenuItem settings = new JMenuItem(translateUtil.getTranslation("Settings"));
        settings.addActionListener(e -> eventManagement.putEvent(new ShowSettingsFrameEvent()));
        JMenuItem restartProc = new JMenuItem(translateUtil.getTranslation("Restart"));
        restartProc.addActionListener(e -> restart());
        JMenuItem close = new JMenuItem(translateUtil.getTranslation("Exit"));
        close.addActionListener(e -> closeAndExit());

        popupMenu.add(settings);
        popupMenu.add(restartProc);
        popupMenu.add(close);
    }

    private void closeAndExit() {
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(new CloseEvent());
    }

    private void restart() {
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(new RestartEvent());
    }

    private void showMessage(String caption, String message, Event event) {
        messageQueue.add(new MessageStruct(caption, message, event));
    }

    private void showMessageOnTrayIcon(String caption, String message) {
        if (trayIcon != null) {
            TranslateUtil translateUtil = TranslateUtil.getInstance();
            TrayIcon.MessageType type = TrayIcon.MessageType.INFO;
            if (caption.equals(translateUtil.getTranslation("Warning"))) {
                type = TrayIcon.MessageType.WARNING;
            }
            trayIcon.displayMessage(caption, message, type);
        }
    }

    @EventRegister
    @SuppressWarnings("unused")
    public static void registerEventHandler() {
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.register(ShowTaskBarMessageEvent.class, event -> {
            ShowTaskBarMessageEvent showTaskBarMessageTask = (ShowTaskBarMessageEvent) event;
            getInstance().showMessage(showTaskBarMessageTask.caption, showTaskBarMessageTask.message, showTaskBarMessageTask.event);
        });

        eventManagement.register(ShowTrayIconEvent.class, event -> getInstance());

        eventManagement.registerListener(RestartEvent.class, () -> {
            TaskBar taskBar = getInstance();
            taskBar.systemTray.remove(taskBar.trayIcon);
        });
    }
}
