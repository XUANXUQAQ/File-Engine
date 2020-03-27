package frames;

import main.MainClass;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;
import java.net.URL;


public class TaskBar {
    private TrayIcon trayIcon = null;
    private SettingsFrame settingsFrame = new SettingsFrame();
    private SystemTray systemTray;

    public void showTaskBar() {
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
            trayIcon.setToolTip("超级搜索");
            // 创建弹出菜单
            PopupMenu popupMenu = new PopupMenu();

            MenuItem settings = new MenuItem("设置");
            settings.addActionListener(e -> settingsFrame.showWindow());
            MenuItem close = new MenuItem("退出");
            close.addActionListener(e -> closeAndExit());
            MenuItem restart = new MenuItem("重启");
            restart.addActionListener(e -> {
                File restartExe = new File("user/restart.exe");
                File mainExe = new File(MainClass.name);
                try {
                    String command = "cmd /c " + restartExe.getAbsolutePath().substring(0, 2) + "\"" + restartExe.getAbsolutePath().substring(2) + "\" \""
                            + mainExe.getAbsolutePath() + "\" " + "\"" + MainClass.name + "\"";
                    Process p = Runtime.getRuntime().exec(command);
                    p.getInputStream().close();
                    p.getOutputStream().close();
                    p.getErrorStream().close();
                    closeAndExit();
                } catch (IOException ex) {
                    showMessage("提示", "重启失败");
                }
            });
            popupMenu.add(settings);
            popupMenu.add(restart);
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
        } else {
            JOptionPane.showMessageDialog(null, "not support");
        }
    }

    public void closeAndExit() {
        MainClass.setMainExit(true);
        systemTray.remove(trayIcon);
        settingsFrame.hideFrame();
    }

    public void showMessage(String caption, String message) {
        if (trayIcon != null) {
            trayIcon.displayMessage(caption, message, TrayIcon.MessageType.INFO);
        }
    }
}
