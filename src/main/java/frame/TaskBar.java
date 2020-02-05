package frame;
import main.MainClass;

import javax.swing.*;
import java.awt.*;
import java.net.URL;


public class TaskBar {
    private TrayIcon trayIcon = null;
    public void showTaskBar()
    {
        // 判断是否支持系统托盘
        if (SystemTray.isSupported())
        {
            Image image;
            URL icon;
            icon = getClass().getResource("/icons/taskbar.png");
            image = new ImageIcon(icon).getImage();
            SystemTray systemTray = SystemTray.getSystemTray();
            // 创建托盘图标
            trayIcon = new TrayIcon(image);
            // 添加工具提示文本
            trayIcon.setToolTip("超级搜索");
            // 创建弹出菜单
            PopupMenu popupMenu = new PopupMenu();

            MenuItem settings = new MenuItem("设置");
            settings.addActionListener(e -> {
                SettingsFrame settingsFrame = new SettingsFrame();
                settingsFrame.showWindow();
            });
            MenuItem close = new MenuItem("退出");
            close.addActionListener(e->{
                MainClass.setMainExit(true);
                systemTray.remove(trayIcon);
            });
            popupMenu.add(settings);
            popupMenu.add(close);

            // 为托盘图标加弹出菜弹
            trayIcon.setPopupMenu(popupMenu);
            // 获得系统托盘对象
            try
            {
                // 为系统托盘加托盘图标
                systemTray.add(trayIcon);
            }
            catch (Exception ignored)
            {

            }
        }
        else
        {
            JOptionPane.showMessageDialog(null, "not support");
        }
    }
    public void showMessage(String caption, String message){
        if (trayIcon!=null){
            trayIcon.displayMessage(caption, message, TrayIcon.MessageType.INFO);
        }
    }
}
