package frame;
import main.Main;

import java.awt.*;
import java.net.URL;
import javax.swing.*;


public class TaskBar {
    public void showTaskBar()
    {
        // �ж��Ƿ�֧��ϵͳ����
        if (SystemTray.isSupported())
        {
            Image image;
            URL icon;
            icon = getClass().getResource("/icons/taskbar.png");
            image = new ImageIcon(icon).getImage();
            SystemTray systemTray = SystemTray.getSystemTray();
            // ��������ͼ��
            TrayIcon trayIcon = new TrayIcon(image);
            // ��ӹ�����ʾ�ı�
            trayIcon.setToolTip("��������");
            // ���������˵�
            PopupMenu popupMenu = new PopupMenu();

            MenuItem settings = new MenuItem("����");
            settings.addActionListener(e -> {
                SettingsFrame settingsFrame = new SettingsFrame();
                settingsFrame.showWindow();
            });
            MenuItem close = new MenuItem("�˳�");
            close.addActionListener(e->{
                Main.setMainExit(true);
                systemTray.remove(trayIcon);
            });
            popupMenu.add(settings);
            popupMenu.add(close);

            // Ϊ����ͼ��ӵ����˵�
            trayIcon.setPopupMenu(popupMenu);
            // ���ϵͳ���̶���
            try
            {
                // Ϊϵͳ���̼�����ͼ��
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
}
