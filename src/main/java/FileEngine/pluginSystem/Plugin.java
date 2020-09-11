package FileEngine.pluginSystem;

import javax.swing.*;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.lang.reflect.Method;
import java.util.LinkedList;

public class Plugin {
    private static final int API_VERSION = 3;
    private final Object instance;
    private final LinkedList<String> methodList = new LinkedList<>();
    private boolean isAllMethodLoaded = false;
    private Method pluginTextChanged;
    private Method pluginLoadPlugin;
    private Method pluginUnloadPlugin;
    private Method pluginKeyReleased;
    private Method pluginKeyPressed;
    private Method pluginKeyTyped;
    private Method pluginMousePressed;
    private Method pluginMouseReleased;
    private Method pluginGetPluginIcon;
    private Method pluginGetOfficialSite;
    private Method pluginGetVersion;
    private Method pluginGetDescription;
    private Method pluginIsLatest;
    private Method pluginGetUpdateURL;
    private Method pluginShowResultOnLabel;
    private Method pluginGetMessage;
    private Method pluginPollFromResultQueue;
    private Method pluginGetApiVersion;
    private Method pluginGetAuthor;
    private Method pluginClearResultQueue;

    public Plugin(PluginUtil.PluginClassAndInstanceInfo pluginClassAndInstanceInfo) {
        Class<?> aClass = pluginClassAndInstanceInfo.cls;
        this.instance = pluginClassAndInstanceInfo.clsInstance;
        initMethodList();
        for (String each : methodList) {
            try {
                loadMethod(each, aClass);
                isAllMethodLoaded = true;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public boolean isPluginLoadedSuccessfully() {
        return isAllMethodLoaded;
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
        methodList.add("clearResultQueue");
    }

    private void loadMethod(String methodName, Class<?> aClass) throws Exception {
        switch (methodName) {
            case "textChanged":
                pluginTextChanged = aClass.getDeclaredMethod("textChanged", String.class);
                break;
            case "loadPlugin":
                pluginLoadPlugin = aClass.getDeclaredMethod("loadPlugin");
                break;
            case "unloadPlugin":
                pluginUnloadPlugin = aClass.getDeclaredMethod("unloadPlugin");
                break;
            case "keyPressed":
                pluginKeyPressed = aClass.getDeclaredMethod("keyPressed", KeyEvent.class, String.class);
                break;
            case "keyReleased":
                pluginKeyReleased = aClass.getDeclaredMethod("keyReleased", KeyEvent.class, String.class);
                break;
            case "keyTyped":
                pluginKeyTyped = aClass.getDeclaredMethod("keyTyped", KeyEvent.class, String.class);
                break;
            case "mousePressed":
                pluginMousePressed = aClass.getDeclaredMethod("mousePressed", MouseEvent.class, String.class);
                break;
            case "mouseReleased":
                pluginMouseReleased = aClass.getDeclaredMethod("mouseReleased", MouseEvent.class, String.class);
                break;
            case "getPluginIcon":
                pluginGetPluginIcon = aClass.getDeclaredMethod("getPluginIcon");
                break;
            case "getOfficialSite":
                pluginGetOfficialSite = aClass.getDeclaredMethod("getOfficialSite");
                break;
            case "getVersion":
                pluginGetVersion = aClass.getDeclaredMethod("getVersion");
                break;
            case "getDescription":
                pluginGetDescription = aClass.getDeclaredMethod("getDescription");
                break;
            case "isLatest":
                pluginIsLatest = aClass.getDeclaredMethod("isLatest");
                break;
            case "getUpdateURL":
                pluginGetUpdateURL = aClass.getDeclaredMethod("getUpdateURL");
                break;
            case "showResultOnLabel":
                pluginShowResultOnLabel = aClass.getDeclaredMethod("showResultOnLabel", String.class, JLabel.class, boolean.class);
                break;
            case "getMessage":
                pluginGetMessage = aClass.getDeclaredMethod("getMessage");
                break;
            case "pollFromResultQueue":
                pluginPollFromResultQueue = aClass.getDeclaredMethod("pollFromResultQueue");
                break;
            case "getApiVersion":
                pluginGetApiVersion = aClass.getDeclaredMethod("getApiVersion");
                break;
            case "getAuthor":
                pluginGetAuthor = aClass.getDeclaredMethod("getAuthor");
                break;
            case "clearResultQueue":
                pluginClearResultQueue = aClass.getDeclaredMethod("clearResultQueue");
                break;
            default:
                break;
        }
    }

    public void clearResultQueue() {
        try {
            pluginClearResultQueue.invoke(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static int getLatestApiVersion() {
        return API_VERSION;
    }

    public int getApiVersion() {
        try {
            return (Integer) pluginGetApiVersion.invoke(instance);
        } catch (Exception e) {
            return 1;
        }
    }

    public String[] getMessage() {
        try {
            return (String[]) pluginGetMessage.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String pollFromResultQueue() {
        try {
            return (String) pluginPollFromResultQueue.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public void textChanged(String text) {
        try {
            if (text != null) {
                pluginTextChanged.invoke(instance, text);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void loadPlugin() {
        try {
            pluginLoadPlugin.invoke(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void unloadPlugin() {
        try {
            pluginUnloadPlugin.invoke(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void keyPressed(KeyEvent e, String result) {
        try {
            pluginKeyPressed.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void keyReleased(KeyEvent e, String result) {
        try {
            pluginKeyReleased.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void keyTyped(KeyEvent e, String result) {
        try {
            pluginKeyTyped.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void mousePressed(MouseEvent e, String result) {
        try {
            pluginMousePressed.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void mouseReleased(MouseEvent e, String result) {
        try {
            pluginMouseReleased.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public String getAuthor() {
        try {
            return (String) pluginGetAuthor.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public ImageIcon getPluginIcon() {
        try {
            return (ImageIcon) pluginGetPluginIcon.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String getOfficialSite() {
        try {
            return (String) pluginGetOfficialSite.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String getVersion() {
        try {
            return (String) pluginGetVersion.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String getDescription() {
        try {
            return (String) pluginGetDescription.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public boolean isLatest() {
        try {
            return (boolean) pluginIsLatest.invoke(instance);
        } catch (Exception e) {
            return true;
        }
    }

    public String getUpdateURL() {
        try {
            return (String) pluginGetUpdateURL.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public void showResultOnLabel(String result, JLabel label, boolean isChosen) {
        try {
            pluginShowResultOnLabel.invoke(instance, result, label, isChosen);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
