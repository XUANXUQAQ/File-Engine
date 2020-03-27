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
        // �ж��Ƿ�֧��ϵͳ����
        if (SystemTray.isSupported()) {
            Image image;
            URL icon;
            icon = getClass().getResource("/icons/taskbar.png");
            image = new ImageIcon(icon).getImage();
            systemTray = SystemTray.getSystemTray();
            // ��������ͼ��
            trayIcon = new TrayIcon(image);
            // ��ӹ�����ʾ�ı�
            trayIcon.setToolTip("��������");
            // ���������˵�
            PopupMenu popupMenu = new PopupMenu();

            MenuItem settings = new MenuItem("����");
            settings.addActionListener(e -> settingsFrame.showWindow());
            MenuItem close = new MenuItem("�˳�");
            close.addActionListener(e -> closeAndExit());
            MenuItem restart = new MenuItem("����");
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
                    showMessage("��ʾ", "����ʧ��");
                }
            });
            popupMenu.add(settings);
            popupMenu.add(restart);
            popupMenu.add(close);

            // Ϊ����ͼ��ӵ����˵�
            trayIcon.setPopupMenu(popupMenu);
            trayIcon.addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    if (MouseEvent.BUTTON1 == e.getButton() && !settingsFrame.isSettingsVisible()) {
                        settingsFrame.showWindow();
                    }
                }
            });
            // ���ϵͳ���̶���
            try {
                // Ϊϵͳ���̼�����ͼ��
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
