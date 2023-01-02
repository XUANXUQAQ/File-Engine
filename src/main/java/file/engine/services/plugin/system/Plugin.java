package file.engine.services.plugin.system;


import javax.swing.*;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
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

    @SuppressWarnings("unchecked")
    private <T> T invokeByKey(String key, Object... args) {
        if (methodHashMap.containsKey(key)) {
            try {
                return (T) methodHashMap.get(key).invoke(instance, args);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private <T> T invokeByKeyNoExcept(String key, Object... args) {
        if (methodHashMap.containsKey(key)) {
            try {
                return (T) methodHashMap.get(key).invoke(instance, args);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    public String restoreFileEngineEventHandler() {
        return invokeByKeyNoExcept("restoreFileEngineEventHandler[]");
    }

    public Object[] pollFromEventHandlerQueue() {
        return invokeByKeyNoExcept("pollFromEventHandlerQueue[]");
    }

    public Object[] pollFromEventQueue() {
        return invokeByKeyNoExcept("pollFromEventQueue[]");
    }

    public void eventProcessed(Class<?> c, Object eventInstance) {
        String key = "eventProcessed" + Arrays.toString(new Class<?>[]{Class.class, Object.class});
        invokeByKeyNoExcept(key, c, eventInstance);
    }

    public void configsChanged(Map<String, Object> configs) {
        String key = "configsChanged" + Arrays.toString(new Class<?>[]{Map.class});
        invokeByKeyNoExcept(key, configs);
    }

    public void loadPlugin(Map<String, Object> configs) {
        String key = "loadPlugin" + Arrays.toString(new Class<?>[]{Map.class});
        invokeByKey(key, configs);
    }

    public void searchBarVisible(String showingMode) {
        String key = "searchBarVisible" + Arrays.toString(new Class<?>[]{String.class});
        invokeByKeyNoExcept(key, showingMode);
    }

    public void setCurrentTheme(int defaultColor, int chosenLabelColor, int borderColor) {
        String key = "setCurrentTheme" + Arrays.toString(new Class<?>[]{int.class, int.class, int.class});
        invokeByKeyNoExcept(key, defaultColor, chosenLabelColor, borderColor);
    }

    public void clearResultQueue() {
        String key = "clearResultQueue[]";
        invokeByKeyNoExcept(key);
    }

    public int getApiVersion() {
        String key = "getApiVersion[]";
        Object api = invokeByKeyNoExcept(key);
        if (api == null) {
            return 1;
        }
        return (int) api;
    }

    public String[] getMessage() {
        String key = "getMessage[]";
        return invokeByKeyNoExcept(key);
    }

    public String pollFromResultQueue() {
        String key = "pollFromResultQueue[]";
        return invokeByKeyNoExcept(key);
    }

    public void textChanged(String text) {
        String key = "textChanged" + Arrays.toString(new Class<?>[]{String.class});
        invokeByKeyNoExcept(key, text);
    }

    public void loadPlugin() {
        String key = "loadPlugin[]";
        invokeByKey(key);
    }

    public void unloadPlugin() {
        String key = "unloadPlugin[]";
        invokeByKeyNoExcept(key);
    }

    public void keyPressed(KeyEvent e, String result) {
        String key = "keyPressed" + Arrays.toString(new Class<?>[]{KeyEvent.class, String.class});
        invokeByKeyNoExcept(key, e, result);
    }

    public void keyReleased(KeyEvent e, String result) {
        String key = "keyReleased" + Arrays.toString(new Class<?>[]{KeyEvent.class, String.class});
        invokeByKeyNoExcept(key, e, result);
    }

    public void keyTyped(KeyEvent e, String result) {
        String key = "keyTyped" + Arrays.toString(new Class<?>[]{KeyEvent.class, String.class});
        invokeByKeyNoExcept(key, e, result);
    }

    public void mousePressed(MouseEvent e, String result) {
        String key = "mousePressed" + Arrays.toString(new Class<?>[]{MouseEvent.class, String.class});
        invokeByKeyNoExcept(key, e, result);
    }

    public void mouseReleased(MouseEvent e, String result) {
        String key = "mouseReleased" + Arrays.toString(new Class<?>[]{MouseEvent.class, String.class});
        invokeByKeyNoExcept(key, e, result);
    }

    public String getAuthor() {
        String key = "getAuthor[]";
        return invokeByKeyNoExcept(key);
    }

    public ImageIcon getPluginIcon() {
        String key = "getPluginIcon[]";
        return invokeByKeyNoExcept(key);
    }

    public String getOfficialSite() {
        String key = "getOfficialSite[]";
        return invokeByKeyNoExcept(key);
    }

    public String getVersion() {
        String key = "getVersion[]";
        return invokeByKeyNoExcept(key);
    }

    public String getDescription() {
        String key = "getDescription[]";
        return invokeByKeyNoExcept(key);
    }

    public boolean isLatest() throws Exception {
        String key = "isLatest[]";
        Object o = invokeByKey(key);
        if (o == null) {
            throw new Exception("failed");
        }
        return (boolean) o;
    }

    public String getUpdateURL() {
        String key = "getUpdateURL[]";
        return invokeByKeyNoExcept(key);
    }

    public void showResultOnLabel(String result, JLabel label, boolean isChosen) {
        String key = "showResultOnLabel" + Arrays.toString(new Class<?>[]{String.class, JLabel.class, boolean.class});
        invokeByKeyNoExcept(key, result, label, isChosen);
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
