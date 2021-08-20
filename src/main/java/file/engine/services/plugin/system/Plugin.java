package file.engine.services.plugin.system;

import javax.swing.*;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.lang.reflect.Method;
import java.util.LinkedList;
import java.util.Map;

public class Plugin {
    private final Object instance;
    private final LinkedList<String> methodList = new LinkedList<>();
    private Method pluginTextChanged;
    private Method pluginLoadPlugin;
    private Method pluginLoadPluginWithConfigs;
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
    private Method pluginSetCurrentTheme;
    private Method pluginSearchBarVisible;
    private Method pluginConfigsChanged;

    public Plugin(PluginClassAndInstanceInfo pluginClassAndInstanceInfo) {
        Class<?> aClass = pluginClassAndInstanceInfo.cls;
        this.instance = pluginClassAndInstanceInfo.clsInstance;
        initMethodList();
        for (String each : methodList) {
            try {
                loadMethod(each, aClass);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 所有插件名
     */
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
        methodList.add("setCurrentTheme");
        methodList.add("searchBarVisible");
        methodList.add("configsChanged");
    }

    /**
     * 初始化方法
     *
     * @param methodName 方法名
     * @param aClass     插件类名
     * @throws Exception exception
     */
    private void loadMethod(String methodName, Class<?> aClass) throws Exception {
        if ("textChanged".equals(methodName)) {
            pluginTextChanged = aClass.getDeclaredMethod("textChanged", String.class);
        } else if ("configsChanged".equals(methodName)) {
            pluginConfigsChanged = aClass.getDeclaredMethod("configsChanged", Map.class);
        } else if ("loadPlugin".equals(methodName)) {
            pluginLoadPlugin = aClass.getDeclaredMethod("loadPlugin");
            pluginLoadPluginWithConfigs = aClass.getDeclaredMethod("loadPlugin", Map.class);
        } else if ("unloadPlugin".equals(methodName)) {
            pluginUnloadPlugin = aClass.getDeclaredMethod("unloadPlugin");
        } else if ("keyPressed".equals(methodName)) {
            pluginKeyPressed = aClass.getDeclaredMethod("keyPressed", KeyEvent.class, String.class);
        } else if ("keyReleased".equals(methodName)) {
            pluginKeyReleased = aClass.getDeclaredMethod("keyReleased", KeyEvent.class, String.class);
        } else if ("keyTyped".equals(methodName)) {
            pluginKeyTyped = aClass.getDeclaredMethod("keyTyped", KeyEvent.class, String.class);
        } else if ("mousePressed".equals(methodName)) {
            pluginMousePressed = aClass.getDeclaredMethod("mousePressed", MouseEvent.class, String.class);
        } else if ("mouseReleased".equals(methodName)) {
            pluginMouseReleased = aClass.getDeclaredMethod("mouseReleased", MouseEvent.class, String.class);
        } else if ("getPluginIcon".equals(methodName)) {
            pluginGetPluginIcon = aClass.getDeclaredMethod("getPluginIcon");
        } else if ("getOfficialSite".equals(methodName)) {
            pluginGetOfficialSite = aClass.getDeclaredMethod("getOfficialSite");
        } else if ("getVersion".equals(methodName)) {
            pluginGetVersion = aClass.getDeclaredMethod("getVersion");
        } else if ("getDescription".equals(methodName)) {
            pluginGetDescription = aClass.getDeclaredMethod("getDescription");
        } else if ("isLatest".equals(methodName)) {
            pluginIsLatest = aClass.getDeclaredMethod("isLatest");
        } else if ("getUpdateURL".equals(methodName)) {
            pluginGetUpdateURL = aClass.getDeclaredMethod("getUpdateURL");
        } else if ("showResultOnLabel".equals(methodName)) {
            pluginShowResultOnLabel = aClass.getDeclaredMethod("showResultOnLabel", String.class, JLabel.class, boolean.class);
        } else if ("getMessage".equals(methodName)) {
            pluginGetMessage = aClass.getDeclaredMethod("getMessage");
        } else if ("pollFromResultQueue".equals(methodName)) {
            pluginPollFromResultQueue = aClass.getDeclaredMethod("pollFromResultQueue");
        } else if ("getApiVersion".equals(methodName)) {
            pluginGetApiVersion = aClass.getDeclaredMethod("getApiVersion");
        } else if ("getAuthor".equals(methodName)) {
            pluginGetAuthor = aClass.getDeclaredMethod("getAuthor");
        } else if ("clearResultQueue".equals(methodName)) {
            pluginClearResultQueue = aClass.getDeclaredMethod("clearResultQueue");
        } else if ("setCurrentTheme".equals(methodName)) {
            pluginSetCurrentTheme = aClass.getDeclaredMethod("setCurrentTheme", int.class, int.class, int.class);
        } else if ("searchBarVisible".equals(methodName)) {
            pluginSearchBarVisible = aClass.getDeclaredMethod("searchBarVisible", String.class);
        }
    }

    public void configsChanged(Map<String, Object> configs) {
        try {
            if (pluginConfigsChanged == null) {
                return;
            }
            pluginConfigsChanged.invoke(instance, configs);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void loadPlugin(Map<String, Object> configs) {
        try {
            if (pluginLoadPluginWithConfigs == null) {
                return;
            }
            pluginLoadPluginWithConfigs.invoke(instance, configs);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void searchBarVisible(String showingMode) {
        try {
            if (pluginSearchBarVisible == null) {
                return;
            }
            pluginSearchBarVisible.invoke(instance, showingMode);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setCurrentTheme(int defaultColor, int chosenLabelColor, int borderColor) {
        try {
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
            pluginClearResultQueue.invoke(instance);
        } catch (Exception e) {
            e.printStackTrace();
        }
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

    public boolean isLatest() throws Exception {
        return (boolean) pluginIsLatest.invoke(instance);
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
