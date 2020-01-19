package frame;
import java.awt.*;
import java.net.URL;
import javax.swing.*;


public class TaskBar {
    public TaskBar()
    {
        // 判断是否支持系统托盘
        if (SystemTray.isSupported())
        {
            Image image;
            URL icon;
            icon = getClass().getResource("/icons/taskbar.png");
            image = new ImageIcon(icon).getImage();
            // 创建托盘图标
            TrayIcon trayIcon = new TrayIcon(image);
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
            close.addActionListener(e -> System.exit(0));
            popupMenu.add(settings);
            popupMenu.add(close);

            // 为托盘图标加弹出菜弹
            trayIcon.setPopupMenu(popupMenu);
            // 获得系统托盘对象
            SystemTray systemTray = SystemTray.getSystemTray();
            try
            {
                // 为系统托盘加托盘图标
                systemTray.add(trayIcon);
            }
            catch (Exception e)
            {
                //e.printStackTrace();
            }
        }
        else
        {
            JOptionPane.showMessageDialog(null, "not support");
        }
    }
}
