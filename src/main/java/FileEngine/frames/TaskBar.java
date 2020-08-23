package FileEngine.frames;

import FileEngine.SQLiteConfig.SQLiteUtil;
import FileEngine.checkHotkey.CheckHotKeyUtil;
import FileEngine.configs.AllConfigs;
import FileEngine.daemon.DaemonUtil;
import FileEngine.dllInterface.FileMonitor;
import FileEngine.pluginSystem.PluginUtil;

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
            if (AllConfigs.isDebug()) {
                trayIcon.setToolTip("File-Engine(Debug)");
            } else {
                trayIcon.setToolTip("File-Engine");
            }
            // 创建弹出菜单
            PopupMenu popupMenu = new PopupMenu();

            MenuItem settings = new MenuItem("Settings");
            settings.addActionListener(e -> settingsFrame.showWindow());
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

    private void closeAndExit() {
        AllConfigs.setMainExit(true);
        systemTray.remove(trayIcon);
        SettingsFrame.getInstance().hideFrame();
        PluginMarket.getInstance().hideWindow();
        SearchBar.getInstance().closeSearchBar();
        PluginUtil.getInstance().unloadAllPlugins();
        CheckHotKeyUtil.getInstance().stopListen();
        FileMonitor.INSTANCE.stop_monitor();
        SQLiteUtil.closeConnection();
        DaemonUtil.stopDaemon();
    }

    private void restart() {
        AllConfigs.setMainExit(true);
        systemTray.remove(trayIcon);
        SettingsFrame.getInstance().hideFrame();
        PluginMarket.getInstance().hideWindow();
        SearchBar.getInstance().closeSearchBar();
        PluginUtil.getInstance().unloadAllPlugins();
        CheckHotKeyUtil.getInstance().stopListen();
        FileMonitor.INSTANCE.stop_monitor();
        SQLiteUtil.closeConnection();
    }

    public void showMessage(String caption, String message) {
        if (trayIcon != null) {
            trayIcon.displayMessage(caption, message, TrayIcon.MessageType.INFO);
        }
    }
}
