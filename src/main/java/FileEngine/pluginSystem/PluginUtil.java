package FileEngine.pluginSystem;

import FileEngine.configs.AllConfigs;
import FileEngine.threadPool.CachedThreadPool;
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
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public class PluginUtil {
    protected static class PluginClassAndInstanceInfo {
        public PluginClassAndInstanceInfo(Class<?> cls, Object instance) {
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

    private final ConcurrentHashMap<String, Plugin> IDENTIFIER_PLUGIN_MAP = new ConcurrentHashMap<>();
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
        return IDENTIFIER_PLUGIN_MAP.values().iterator();
    }

    public Plugin getPluginByIdentifier(String identifier) {
        return IDENTIFIER_PLUGIN_MAP.get(identifier);
    }

    public int getInstalledPluginNum() {
        return IDENTIFIER_PLUGIN_MAP.size();
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
        PluginClassAndInstanceInfo pluginClassAndInstanceInfo = new PluginClassAndInstanceInfo(c, instance);
        Plugin plugin = new Plugin(pluginClassAndInstanceInfo);
        plugin.loadPlugin();
        plugin.setCurrentTheme(AllConfigs.getDefaultBackgroundColor(), AllConfigs.getLabelColor(), AllConfigs.getBorderColor());
        if (plugin.getApiVersion() != Plugin.getLatestApiVersion()) {
            isTooOld = true;
        }
        IDENTIFIER_PLUGIN_MAP.put(identifier, plugin);
        return isTooOld;
    }

    public void unloadAllPlugins() {
        for (Plugin each : IDENTIFIER_PLUGIN_MAP.values()) {
            unloadPlugin(each);
        }
    }

    public void setCurrentTheme(int defaultColor, int chosenLabelColor, int borderColor) {
        for (Plugin each : IDENTIFIER_PLUGIN_MAP.values()) {
            each.setCurrentTheme(defaultColor, chosenLabelColor, borderColor);
        }
    }

    public String getIdentifierByName(String name) {
        return NAME_IDENTIFIER_MAP.get(name);
    }

    public Object[] getPluginArray() {
        ArrayList<Plugin> list = new ArrayList<>(IDENTIFIER_PLUGIN_MAP.values());
        return list.toArray();
    }

    public Object[] getPluginNameArray() {
        ArrayList<String> list = new ArrayList<>(NAME_IDENTIFIER_MAP.keySet());
        return list.toArray();
    }

    public void loadAllPlugins(String pluginPath) {
        FilenameFilter filter = (dir, name) -> name.endsWith(".jar");
        File[] files = new File(pluginPath).listFiles(filter);
        if (files == null || files.length == 0) {
            return;
        }
        try {
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
                            try {
                                boolean isPluginApiTooOld= loadPlugin(jar, className, identifier);
                                isTooOld |= isPluginApiTooOld;
                                NAME_IDENTIFIER_MAP.put(pluginName, identifier);
                                if (isPluginApiTooOld) {
                                    OLD_PLUGINS.add(pluginName);
                                }
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        } else {
                            REPEAT_PLUGINS.add(jar.getName());
                            isRepeat = true;
                        }
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void isAllPluginLatest(StringBuilder stringBuilder, AtomicBoolean isFinished) {
        boolean isLatest;
        for (String each : NAME_IDENTIFIER_MAP.keySet()) {
            try {
                isLatest = isPluginLatest(IDENTIFIER_PLUGIN_MAP.get(NAME_IDENTIFIER_MAP.get(each)));
                if (AllConfigs.isDebug()) {
                    System.out.println("++++++++++++++++++++++++++++++++++++++++++++");
                    System.out.println("插件：" + each + "已检查完毕，结果：" + isLatest);
                    System.out.println("++++++++++++++++++++++++++++++++++++++++++++\n");
                }
                if (!isLatest) {
                    stringBuilder.append(each).append(",");
                }
            }catch (Exception e) {
                System.err.println("++++++++++++++++++++++++++++++++++++++++++++");
                System.err.println("插件：" + each + "检查更新失败");
                System.err.println("++++++++++++++++++++++++++++++++++++++++++++");
            }
        }
        isFinished.set(true);
    }

    private boolean isPluginLatest(Plugin plugin) throws Exception {
        AtomicLong startCheckTime = new AtomicLong();
        AtomicBoolean isVersionLatest = new AtomicBoolean();
        Thread checkUpdateThread = new Thread(() -> {
            startCheckTime.set(System.currentTimeMillis());
            isVersionLatest.set(plugin.isLatest());
            if (!Thread.interrupted()) {
                startCheckTime.set(0x100L); //表示检查成功
            }
        });
        CachedThreadPool.getInstance().executeTask(checkUpdateThread);
        //等待获取插件更新信息
        try {
            while (startCheckTime.get() != 0x100L) {
                TimeUnit.MILLISECONDS.sleep(200);
                if (System.currentTimeMillis() - startCheckTime.get() > 5000L && startCheckTime.get() != 0x100L) {
                    checkUpdateThread.interrupt();
                    throw new Exception("check update failed.");
                }
                if (!AllConfigs.isNotMainExit()) {
                    break;
                }
            }
            return isVersionLatest.get();
        } catch (InterruptedException e) {
            throw new Exception("check update failed.");
        }
    }
}
