package frame;
import main.MainClass;

import javax.swing.*;
import java.awt.*;
import java.net.URL;


public class TaskBar {
    private TrayIcon trayIcon = null;
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
            trayIcon = new TrayIcon(image);
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
                MainClass.setMainExit(true);
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
    public void showMessage(String caption, String message){
        if (trayIcon!=null){
            trayIcon.displayMessage(caption, message, TrayIcon.MessageType.INFO);
        }
    }
}
