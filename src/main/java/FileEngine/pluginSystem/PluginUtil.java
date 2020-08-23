package FileEngine.pluginSystem;

import com.alibaba.fastjson.JSONObject;

import java.io.*;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public class PluginUtil {
    protected static class PluginInfo {
        public PluginInfo(Class<?> cls, Object instance) {
            this.cls = cls;
            this.clsInstance = instance;
        }

        public final Class<?> cls;
        public final Object clsInstance;
    }

    private static class PluginUtilBuilder {
        private static final PluginUtil INSTANCE = new PluginUtil();
    }

    public static PluginUtil getInstance() {
        return PluginUtilBuilder.INSTANCE;
    }

    private PluginUtil() {
    }

    private final ConcurrentHashMap<String, Plugin> PLUGIN_MAP = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, String> NAME_IDENTIFIER_MAP = new ConcurrentHashMap<>();
    private final HashSet<String> OLD_PLUGINS = new HashSet<>();
    private final HashSet<String> REPEAT_PLUGINS = new HashSet<>();
    private boolean isTooOld = false;
    private boolean isRepeat = false;

    public boolean isPluginTooOld() {
        return isTooOld;
    }

    public boolean isPluginRepeat() {
        return isRepeat;
    }

    public String getRepeatPlugins() {
        StringBuilder strb = new StringBuilder();
        for (String repeatPlugins : REPEAT_PLUGINS) {
            strb.append(repeatPlugins).append(",");
        }
        return strb.substring(0, strb.length() - 1);
    }

    public Iterator<Plugin> getPluginMapIter() {
        return PLUGIN_MAP.values().iterator();
    }

    public Plugin getPluginByIdentifier(String identifier) {
        return PLUGIN_MAP.get(identifier);
    }

    public int getInstalledPluginNum() {
        return PLUGIN_MAP.size();
    }

    public String getAllOldPluginsName() {
        StringBuilder strb = new StringBuilder();
        for (String oldPlugin : OLD_PLUGINS) {
            strb.append(oldPlugin).append(",");
        }
        return strb.substring(0, strb.length() - 1);
    }

    private String readAll(InputStream inputStream) {
        StringBuilder strb = new StringBuilder();
        String line;
        try (BufferedReader buffr = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
            while ((line = buffr.readLine()) != null) {
                strb.append(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return strb.toString();
    }

    private void unloadPlugin(Plugin plugin) {
        if (plugin != null) {
            plugin.unloadPlugin();
        }
    }

    private boolean loadPlugin(File pluginFile, String className, String identifier) throws Exception {
        boolean isTooOld = false;
        URLClassLoader classLoader = new URLClassLoader(
                new URL[]{pluginFile.toURI().toURL()},
                PluginUtil.class.getClassLoader()
        );
        Class<?> c = Class.forName(className, true, classLoader);
        Object instance = c.getDeclaredConstructor().newInstance();
        PluginInfo pluginInfo = new PluginInfo(c, instance);
        Plugin plugin = new Plugin(pluginInfo);
        plugin.loadPlugin();
        if (plugin.getApiVersion() != Plugin.getLatestApiVersion()) {
            isTooOld = true;
        }
        PLUGIN_MAP.put(identifier, plugin);
        return isTooOld;
    }

    public void unloadAllPlugins() {
        for (String each : PLUGIN_MAP.keySet()) {
            unloadPlugin(PLUGIN_MAP.get(each));
        }
    }

    public String getIdentifierByName(String name) {
        return NAME_IDENTIFIER_MAP.get(name);
    }

    public Object[] getPluginArray() {
        ArrayList<String> list = new ArrayList<>(NAME_IDENTIFIER_MAP.keySet());
        return list.toArray();
    }

    public void loadAllPlugins(String pluginPath) throws Exception {
        FilenameFilter filter = (dir, name) -> name.endsWith(".jar");
        File[] files = new File(pluginPath).listFiles(filter);
        if (files == null || files.length == 0) {
            return;
        }
        for (File jar : files) {
            JarFile jarFile = new JarFile(jar);
            Enumeration<?> enu = jarFile.entries();
            while (enu.hasMoreElements()) {
                JarEntry element = (JarEntry) enu.nextElement();
                String name = element.getName();
                if ("PluginInfo.json".equals(name)) {
                    String pluginInfo = readAll(jarFile.getInputStream(element));
                    JSONObject json = JSONObject.parseObject(pluginInfo);
                    String className = json.getString("className");
                    String identifier = json.getString("identifier");
                    String pluginName = json.getString("name");
                    if (getIdentifierByName(pluginName) == null) {
                        isTooOld |= loadPlugin(jar, className, identifier);
                        NAME_IDENTIFIER_MAP.put(pluginName, identifier);
                        if (isTooOld) {
                            OLD_PLUGINS.add(pluginName);
                        }
                    } else {
                        REPEAT_PLUGINS.add(jar.getName());
                        isRepeat = true;
                    }
                }
            }
        }
    }
}
