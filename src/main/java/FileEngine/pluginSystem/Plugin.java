package FileEngine.pluginSystem;

import javax.swing.*;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.lang.reflect.Method;
import java.util.ArrayList;

public class Plugin {
    private static final int API_VERSION = 2;
    private final Object instance;
    private final ArrayList<String> methodList = new ArrayList<>();
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
    private Method _getApiVersion;
    private Method _getAuthor;

    public Plugin(PluginUtil.PluginInfo pluginInfo) {
        Class<?> aClass = pluginInfo.cls;
        this.instance = pluginInfo.clsInstance;
        initMethodList();
        for (String each : methodList) {
            try {
                loadMethod(each, aClass);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void initMethodList() {
        methodList.add("textChanged");
        methodList.add("loadPlugin");
        methodList.add("unloadPlugin");
        methodList.add("keyPressed");
        methodList.add("keyReleased");
        methodList.add("keyTyped");
        methodList.add("mousePressed");
        methodList.add("mouseReleased");
        methodList.add("getPluginIcon");
        methodList.add("getOfficialSite");
        methodList.add("getVersion");
        methodList.add("getDescription");
        methodList.add("isLatest");
        methodList.add("getUpdateURL");
        methodList.add("showResultOnLabel");
        methodList.add("getMessage");
        methodList.add("pollFromResultQueue");
        methodList.add("getApiVersion");
        methodList.add("getAuthor");
    }

    private void loadMethod(String methodName, Class<?> aClass) throws Exception {
        switch (methodName) {
            case "textChanged":
                _textChanged = aClass.getDeclaredMethod("textChanged", String.class);
                break;
            case "loadPlugin":
                _loadPlugin = aClass.getDeclaredMethod("loadPlugin");
                break;
            case "unloadPlugin":
                _unloadPlugin = aClass.getDeclaredMethod("unloadPlugin");
                break;
            case "keyPressed":
                _keyPressed = aClass.getDeclaredMethod("keyPressed", KeyEvent.class, String.class);
                break;
            case "keyReleased":
                _keyReleased = aClass.getDeclaredMethod("keyReleased", KeyEvent.class, String.class);
                break;
            case "keyTyped":
                _keyTyped = aClass.getDeclaredMethod("keyTyped", KeyEvent.class, String.class);
                break;
            case "mousePressed":
                _mousePressed = aClass.getDeclaredMethod("mousePressed", MouseEvent.class, String.class);
                break;
            case "mouseReleased":
                _mouseReleased = aClass.getDeclaredMethod("mouseReleased", MouseEvent.class, String.class);
                break;
            case "getPluginIcon":
                _getPluginIcon = aClass.getDeclaredMethod("getPluginIcon");
                break;
            case "getOfficialSite":
                _getOfficialSite = aClass.getDeclaredMethod("getOfficialSite");
                break;
            case "getVersion":
                _getVersion = aClass.getDeclaredMethod("getVersion");
                break;
            case "getDescription":
                _getDescription = aClass.getDeclaredMethod("getDescription");
                break;
            case "isLatest":
                _isLatest = aClass.getDeclaredMethod("isLatest");
                break;
            case "getUpdateURL":
                _getUpdateURL = aClass.getDeclaredMethod("getUpdateURL");
                break;
            case "showResultOnLabel":
                _showResultOnLabel = aClass.getDeclaredMethod("showResultOnLabel", String.class, JLabel.class, boolean.class);
                break;
            case "getMessage":
                _getMessage = aClass.getDeclaredMethod("getMessage");
                break;
            case "pollFromResultQueue":
                _pollFromResultQueue = aClass.getDeclaredMethod("pollFromResultQueue");
                break;
            case "getApiVersion":
                _getApiVersion = aClass.getDeclaredMethod("getApiVersion");
                break;
            case "getAuthor":
                _getAuthor = aClass.getDeclaredMethod("getAuthor");
                break;
            default:
                break;
        }
    }

    public static int getLatestApiVersion() {
        return API_VERSION;
    }

    public int getApiVersion() {
        try {
            return (Integer) _getApiVersion.invoke(instance);
        }catch (Exception e) {
            return 1;
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

    public String getAuthor() {
        try {
            return (String) _getAuthor.invoke(instance);
        } catch (Exception e) {
            return null;
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
