package file.engine.frames;

import file.engine.utils.DpiUtil;
import file.engine.utils.system.properties.IsDebug;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.dllInterface.GetHandle;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.frame.settingsFrame.ShowSettingsFrameEvent;
import file.engine.event.handler.impl.stop.CloseEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.event.handler.impl.taskbar.ShowTrayIconEvent;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.services.TranslateService;
import lombok.Data;

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
    private volatile JPopupMenu popupMenu = null;
    private final int L_BUTTON = 0x01;
    private final int R_BUTTON = 0x02;

    private static volatile TaskBar INSTANCE = null;

    @Data
    private static class MessageStruct {
        private final String caption;
        private final String message;
        private final Event event;
    }

    private TaskBar() {
        startShowMessageThread();
        showTaskBar();
        addActionListener();

        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement instance = EventManagement.getInstance();
            try {
                while (instance.notMainExit()) {
                    if (popupMenu != null && popupMenu.isVisible() && (GetHandle.INSTANCE.isKeyPressed(L_BUTTON) || GetHandle.INSTANCE.isKeyPressed(R_BUTTON))) {
                        Point point = java.awt.MouseInfo.getPointerInfo().getLocation();
                        Point location = popupMenu.getLocationOnScreen();
                        int X = location.x;
                        int Y = location.y;
                        int width = popupMenu.getWidth();
                        int height = popupMenu.getHeight();
                        if (!(X <= point.x && point.x <= X + width && Y < point.y && point.y <= Y + height)) {
                            SwingUtilities.invokeLater(() -> popupMenu.setVisible(false));
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    /**
     * 显示托盘消息线程
     */
    private void startShowMessageThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                EventManagement eventManagement = EventManagement.getInstance();
                MessageStruct message;
                int count = 0;
                while (eventManagement.notMainExit()) {
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
            } catch (InterruptedException e) {
                e.printStackTrace();
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

    /**
     * 响应托盘点击事件
     */
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
            systemTray = SystemTray.getSystemTray();
            EventManagement eventManagement = EventManagement.getInstance();
            // 创建托盘图标
            icon = TaskBar.class.getResource("/icons/taskbar.png");
            if (icon != null) {
                image = new ImageIcon(icon).getImage();
                trayIcon = new TrayIcon(image);
            } else {
                throw new RuntimeException("初始化图片失败/icons/taskbar.png");
            }
            // 添加工具提示文本
            if (IsDebug.isDebug()) {
                trayIcon.setToolTip("File-Engine(Debug)");
            } else {
                trayIcon.setToolTip("File-Engine," + TranslateService.getInstance().getTranslation("Double click to open settings"));
            }
            // 为托盘图标加弹出菜单

            trayIcon.addMouseListener(new MouseAdapter() {

                @Override
                public void mouseClicked(MouseEvent e) {
                    if (MouseEvent.BUTTON1 == e.getButton() && e.getClickCount() == 2) {
                        eventManagement.putEvent(new ShowSettingsFrameEvent());
                    }
                }

                @Override
                public void mouseReleased(MouseEvent e) {
                    if (e.isPopupTrigger()) {
                        popupMenu = getPopupMenu();
                        popupMenu.setInvoker(popupMenu);
                        popupMenu.setVisible(true);
                        double dpi = DpiUtil.getDpi();
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

    /**
     * 创建弹出菜单
     *
     * @return 菜单
     */
    private JPopupMenu getPopupMenu() {
        JPopupMenu popupMenu = new JPopupMenu();
        EventManagement eventManagement = EventManagement.getInstance();
        TranslateService translateService = TranslateService.getInstance();
        JMenuItem settings = new JMenuItem(translateService.getTranslation("Settings"));
        settings.addActionListener(e -> eventManagement.putEvent(new ShowSettingsFrameEvent()));

        JSeparator separator = new JSeparator();

        JMenuItem restartProc = new JMenuItem(translateService.getTranslation("Restart"));
        restartProc.addActionListener(e -> restart());
        JMenuItem close = new JMenuItem(translateService.getTranslation("Exit"));
        close.addActionListener(e -> closeAndExit());

        popupMenu.add(settings);
        popupMenu.add(separator);
        popupMenu.add(restartProc);
        popupMenu.add(close);
        return popupMenu;
    }

    /**
     * 点击退出
     */
    private void closeAndExit() {
        systemTray.remove(trayIcon);
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(new CloseEvent());
    }

    /**
     * 点击重启
     */
    private void restart() {
        systemTray.remove(trayIcon);
        EventManagement eventManagement = EventManagement.getInstance();
        eventManagement.putEvent(new RestartEvent());
    }

    /**
     * 将消息放入队列中
     *
     * @param caption 标题
     * @param message 消息
     * @param event   携带的事件
     */
    private void showMessage(String caption, String message, Event event) {
        messageQueue.add(new MessageStruct(caption, message, event));
    }

    /**
     * 显示消息
     *
     * @param caption 标题
     * @param message 消息
     */
    private void showMessageOnTrayIcon(String caption, String message) {
        if (trayIcon != null) {
            TranslateService translateService = TranslateService.getInstance();
            TrayIcon.MessageType type = TrayIcon.MessageType.INFO;
            if (caption.equals(translateService.getTranslation("Warning"))) {
                type = TrayIcon.MessageType.WARNING;
            }
            trayIcon.displayMessage(caption, message, type);
        }
    }

    @EventRegister(registerClass = ShowTaskBarMessageEvent.class)
    private static void showTaskBarMessageEvent(Event event) {
        ShowTaskBarMessageEvent showTaskBarMessageTask = (ShowTaskBarMessageEvent) event;
        getInstance().showMessage(showTaskBarMessageTask.caption, showTaskBarMessageTask.message, showTaskBarMessageTask.event);
    }

    @EventRegister(registerClass = ShowTrayIconEvent.class)
    private static void showTrayIconEvent(Event event) {
        getInstance();
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void restartEvent(Event event) {
        TaskBar taskBar = getInstance();
        taskBar.systemTray.remove(taskBar.trayIcon);
    }
}
