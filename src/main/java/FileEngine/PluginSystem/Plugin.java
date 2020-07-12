package FileEngine.PluginSystem;

import javax.swing.*;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.lang.reflect.Method;

public class Plugin {
    private final Object instance;
    private Method _textChanged;
    private Method _loadPlugin;
    private Method _unloadPlugin;
    private Method _keyReleased;
    private Method _keyPressed;
    private Method _keyTyped;
    private Method _mousePressed;
    private Method _mouseReleased;
    private Method _getPluginIcon;
    private Method _getOfficialSite;
    private Method _getVersion;
    private Method _getDescription;
    private Method _isLatest;
    private Method _getUpdateURL;
    private Method _showResultOnLabel;
    private Method _getMessage;
    private Method _pollFromResultQueue;

    public Plugin(PluginUtil.PluginInfo pluginInfo) {
        Class<?> aClass = pluginInfo.cls;
        this.instance = pluginInfo.clsInstance;
        try {
            _textChanged = aClass.getDeclaredMethod("textChanged", String.class);
            _loadPlugin = aClass.getDeclaredMethod("loadPlugin");
            _unloadPlugin = aClass.getDeclaredMethod("unloadPlugin");
            _keyPressed = aClass.getDeclaredMethod("keyPressed", KeyEvent.class, String.class);
            _keyReleased = aClass.getDeclaredMethod("keyReleased", KeyEvent.class, String.class);
            _keyTyped = aClass.getDeclaredMethod("keyTyped", KeyEvent.class, String.class);
            _mousePressed = aClass.getDeclaredMethod("mousePressed", MouseEvent.class, String.class);
            _mouseReleased = aClass.getDeclaredMethod("mouseReleased", MouseEvent.class, String.class);
            _getPluginIcon = aClass.getDeclaredMethod("getPluginIcon");
            _getOfficialSite = aClass.getDeclaredMethod("getOfficialSite");
            _getVersion = aClass.getDeclaredMethod("getVersion");
            _getDescription = aClass.getDeclaredMethod("getDescription");
            _isLatest = aClass.getDeclaredMethod("isLatest");
            _getUpdateURL = aClass.getDeclaredMethod("getUpdateURL");
            _showResultOnLabel = aClass.getDeclaredMethod("showResultOnLabel", String.class, JLabel.class, boolean.class);
            _getMessage = aClass.getDeclaredMethod("getMessage");
            _pollFromResultQueue = aClass.getDeclaredMethod("pollFromResultQueue");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String[] getMessage() {
        try {
            return (String[]) _getMessage.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String pollFromResultQueue() {
        try {
            return (String) _pollFromResultQueue.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public void textChanged(String text) {
        try {
            if (text != null) {
                _textChanged.invoke(instance, text);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void loadPlugin() {
        try {
            _loadPlugin.invoke(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void unloadPlugin() {
        try {
            _unloadPlugin.invoke(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void keyPressed(KeyEvent e, String result) {
        try {
            _keyPressed.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void keyReleased(KeyEvent e, String result) {
        try {
            _keyReleased.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void keyTyped(KeyEvent e, String result) {
        try {
            _keyTyped.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void mousePressed(MouseEvent e, String result) {
        try {
            _mousePressed.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void mouseReleased(MouseEvent e, String result) {
        try {
            _mouseReleased.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public ImageIcon getPluginIcon() {
        try {
            return (ImageIcon) _getPluginIcon.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String getOfficialSite() {
        try {
            return (String) _getOfficialSite.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String getVersion() {
        try {
            return (String) _getVersion.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String getDescription() {
        try {
            return (String) _getDescription.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public boolean isLatest() {
        try {
            return (boolean) _isLatest.invoke(instance);
        } catch (Exception e) {
            return true;
        }
    }

    public String getUpdateURL() {
        try {
            return (String) _getUpdateURL.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public void showResultOnLabel(String result, JLabel label, boolean isChosen) {
        try {
            _showResultOnLabel.invoke(instance, result, label, isChosen);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
