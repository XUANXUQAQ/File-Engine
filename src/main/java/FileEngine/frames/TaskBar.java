package FileEngine.frames;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.net.URL;


public class TaskBar {
    private TrayIcon trayIcon = null;
    private SystemTray systemTray;

    private static class TaskBarBuilder {
        private static final TaskBar INSTANCE = new TaskBar();
    }

    private TaskBar() {
    }


    public static TaskBar getInstance() {
        return TaskBarBuilder.INSTANCE;
    }

    public void showTaskBar() {
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
            trayIcon.setToolTip("File-Engine");
            // 创建弹出菜单
            PopupMenu popupMenu = new PopupMenu();

            MenuItem settings = new MenuItem("Settings");
            settings.addActionListener(e -> settingsFrame.showWindow());
            MenuItem close = new MenuItem("Exit");
            close.addActionListener(e -> closeAndExit());

            popupMenu.add(settings);
            popupMenu.add(close);

            // 为托盘图标加弹出菜弹
            trayIcon.setPopupMenu(popupMenu);
            trayIcon.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    if (MouseEvent.BUTTON1 == e.getButton() && !settingsFrame.isSettingsVisible()) {
                        settingsFrame.showWindow();
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

    public void closeAndExit() {
        SettingsFrame.setMainExit(true);
        systemTray.remove(trayIcon);
        SettingsFrame.getInstance().hideFrame();
        PluginMarket.getInstance().hideWindow();
    }

    public void showMessage(String caption, String message) {
        if (trayIcon != null) {
            trayIcon.displayMessage(caption, message, TrayIcon.MessageType.INFO);
        }
    }
}
