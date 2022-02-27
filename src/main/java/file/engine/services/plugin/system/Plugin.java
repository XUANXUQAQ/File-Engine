package file.engine.services.plugin.system;

import javax.swing.*;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Plugin {
    public final String name;
    public final String identifier;
    private final Object instance;
    private final ConcurrentHashMap<String, Method> methodHashMap = new ConcurrentHashMap<>();
    private static final HashSet<String> methodList = new HashSet<>();

    static {
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
        methodList.add("setCurrentTheme");
        methodList.add("searchBarVisible");
        methodList.add("configsChanged");
        methodList.add("eventProcessed");
        methodList.add("pollFromEventQueue");
        methodList.add("pollFromEventHandlerQueue");
        methodList.add("restoreFileEngineEventHandler");
    }

    public Plugin(String name, String identifier, PluginClassAndInstanceInfo pluginClassAndInstanceInfo) {
        this.name = name;
        this.identifier = identifier;
        Class<?> aClass = pluginClassAndInstanceInfo.cls;
        this.instance = pluginClassAndInstanceInfo.clsInstance;
        for (Method method : aClass.getMethods()) {
            String methodName = method.getName();
            Class<?>[] parameterTypes = method.getParameterTypes();
            String paramTypes = Arrays.toString(parameterTypes);
            String key = methodName + paramTypes;
            if (methodList.contains(methodName)) {
                methodHashMap.put(key, method);
            }
        }
    }

    public String restoreFileEngineEventHandler() {
        try {
            return (String) methodHashMap.get("restoreFileEngineEventHandler[]").invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public Object[] pollFromEventHandlerQueue() {
        try {
            return (Object[]) methodHashMap.get("pollFromEventHandlerQueue[]").invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public Object[] pollFromEventQueue() {
        try {
            return (Object[]) methodHashMap.get("pollFromEventQueue[]").invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public void eventProcessed(Class<?> c, Object eventInstance) {
        try {
            String key = "eventProcessed" + Arrays.toString(new Class<?>[]{Class.class, Object.class});
            Method pluginEventProcessed = methodHashMap.get(key);
            if (pluginEventProcessed == null) {
                return;
            }
            pluginEventProcessed.invoke(instance, c, eventInstance);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void configsChanged(Map<String, Object> configs) {
        try {
            String key = "configsChanged" + Arrays.toString(new Class<?>[]{Map.class});
            Method pluginConfigsChanged = methodHashMap.get(key);
            pluginConfigsChanged.invoke(instance, configs);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void loadPlugin(Map<String, Object> configs) {
        try {
            String key = "loadPlugin" + Arrays.toString(new Class<?>[]{Map.class});
            Method method = methodHashMap.get(key);
            if (method == null) {
                return;
            }
            method.invoke(instance, configs);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void searchBarVisible(String showingMode) {
        try {
            String key = "searchBarVisible" + Arrays.toString(new Class<?>[]{String.class});
            Method method = methodHashMap.get(key);
            if (method == null) {
                return;
            }
            method.invoke(instance, showingMode);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setCurrentTheme(int defaultColor, int chosenLabelColor, int borderColor) {
        try {
            String key = "setCurrentTheme" + Arrays.toString(new Class<?>[]{int.class, int.class, int.class});
            Method pluginSetCurrentTheme = methodHashMap.get(key);
            if (pluginSetCurrentTheme == null) {
                return;
            }
            pluginSetCurrentTheme.invoke(instance, defaultColor, chosenLabelColor, borderColor);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void clearResultQueue() {
        try {
            String key = "clearResultQueue[]";
            Method pluginClearResultQueue = methodHashMap.get(key);
            pluginClearResultQueue.invoke(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public int getApiVersion() {
        try {
            String key = "getApiVersion[]";
            Method pluginGetApiVersion = methodHashMap.get(key);
            return (Integer) pluginGetApiVersion.invoke(instance);
        } catch (Exception e) {
            return 1;
        }
    }

    public String[] getMessage() {
        try {
            String key = "getMessage[]";
            Method pluginGetMessage = methodHashMap.get(key);
            return (String[]) pluginGetMessage.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String pollFromResultQueue() {
        try {
            String key = "pollFromResultQueue[]";
            Method pluginPollFromResultQueue = methodHashMap.get(key);
            return (String) pluginPollFromResultQueue.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public void textChanged(String text) {
        try {
            String key = "textChanged" + Arrays.toString(new Class<?>[]{String.class});
            Method pluginTextChanged = methodHashMap.get(key);
            if (text != null) {
                pluginTextChanged.invoke(instance, text);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void loadPlugin() {
        try {
            String key = "loadPlugin[]";
            Method pluginLoadPlugin = methodHashMap.get(key);
            pluginLoadPlugin.invoke(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void unloadPlugin() {
        try {
            String key = "unloadPlugin[]";
            Method pluginUnloadPlugin = methodHashMap.get(key);
            pluginUnloadPlugin.invoke(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void keyPressed(KeyEvent e, String result) {
        try {
            String key = "keyPressed" + Arrays.toString(new Class<?>[]{KeyEvent.class, String.class});
            Method pluginKeyPressed = methodHashMap.get(key);
            pluginKeyPressed.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void keyReleased(KeyEvent e, String result) {
        try {
            String key = "keyReleased" + Arrays.toString(new Class<?>[]{KeyEvent.class, String.class});
            Method pluginKeyReleased = methodHashMap.get(key);
            pluginKeyReleased.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void keyTyped(KeyEvent e, String result) {
        try {
            String key = "keyTyped" + Arrays.toString(new Class<?>[]{KeyEvent.class, String.class});
            Method pluginKeyTyped = methodHashMap.get(key);
            pluginKeyTyped.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void mousePressed(MouseEvent e, String result) {
        try {
            String key = "mousePressed" + Arrays.toString(new Class<?>[]{MouseEvent.class, String.class});
            Method pluginMousePressed = methodHashMap.get(key);
            pluginMousePressed.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public void mouseReleased(MouseEvent e, String result) {
        try {
            String key = "mouseReleased" + Arrays.toString(new Class<?>[]{MouseEvent.class, String.class});
            Method pluginMouseReleased = methodHashMap.get(key);
            pluginMouseReleased.invoke(instance, e, result);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
    }

    public String getAuthor() {
        try {
            String key = "getAuthor[]";
            Method pluginGetAuthor = methodHashMap.get(key);
            return (String) pluginGetAuthor.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public ImageIcon getPluginIcon() {
        try {
            String key = "getPluginIcon[]";
            Method pluginGetPluginIcon = methodHashMap.get(key);
            return (ImageIcon) pluginGetPluginIcon.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String getOfficialSite() {
        try {
            String key = "getOfficialSite[]";
            Method pluginGetOfficialSite = methodHashMap.get(key);
            return (String) pluginGetOfficialSite.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String getVersion() {
        try {
            String key = "getVersion[]";
            Method pluginGetVersion = methodHashMap.get(key);
            return (String) pluginGetVersion.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public String getDescription() {
        try {
            String key = "getDescription[]";
            Method pluginGetDescription = methodHashMap.get(key);
            return (String) pluginGetDescription.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public boolean isLatest() throws Exception {
        String key = "isLatest[]";
        Method pluginIsLatest = methodHashMap.get(key);
        return (boolean) pluginIsLatest.invoke(instance);
    }

    public String getUpdateURL() {
        try {
            String key = "getUpdateURL[]";
            Method pluginGetUpdateURL = methodHashMap.get(key);
            return (String) pluginGetUpdateURL.invoke(instance);
        } catch (Exception e) {
            return null;
        }
    }

    public void showResultOnLabel(String result, JLabel label, boolean isChosen) {
        try {
            String key = "showResultOnLabel" + Arrays.toString(new Class<?>[]{String.class, JLabel.class, boolean.class});
            Method pluginShowResultOnLabel = methodHashMap.get(key);
            pluginShowResultOnLabel.invoke(instance, result, label, isChosen);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 插件的类和实例的基本信息
     */
    protected static class PluginClassAndInstanceInfo {
        protected PluginClassAndInstanceInfo(Class<?> cls, Object instance) {
            this.cls = cls;
            this.clsInstance = instance;
        }

        public final Class<?> cls;
        public final Object clsInstance;
    }
}
