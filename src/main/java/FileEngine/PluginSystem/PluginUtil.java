package FileEngine.PluginSystem;

import com.alibaba.fastjson.JSONObject;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.concurrent.ConcurrentHashMap;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public class PluginUtil {
    private static final ConcurrentHashMap<String, Plugin> pluginMap = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, String> nameIdentifierMap = new ConcurrentHashMap<>();

    public static Plugin getPluginByIdentifier(String identifier) {
        return pluginMap.get(identifier);
    }

    public static int getInstalledPluginNum() {
        return pluginMap.size();
    }

    private static String readAll(InputStream inputStream) {
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

    private static void unloadPlugin(Plugin plugin) {
        if (plugin != null) {
            plugin.unloadPlugin();
        }
    }

    private static void loadPlugin(File pluginFile, String className, String identifier) throws Exception {
        //TODO 实例化插件
        /*Plugin plugin = (Plugin) ;
        plugin.loadPlugin();
        pluginMap.put(identifier, plugin);*/
    }

    public static void unloadAllPlugins() {
        for (String each : pluginMap.keySet()) {
            unloadPlugin(pluginMap.get(each));
        }
    }

    public static String getIdentifierByName(String name) {
        return nameIdentifierMap.get(name);
    }

    public static Object[] getPluginArray() {
        ArrayList<String> list = new ArrayList<>(nameIdentifierMap.keySet());
        return list.toArray();
    }

    public static void loadAllPlugins(String pluginPath) throws Exception {
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
                if (name.equals("PluginInfo.json")) {
                    String pluginInfo = readAll(jarFile.getInputStream(element));
                    JSONObject json = JSONObject.parseObject(pluginInfo);
                    String className = json.getString("className");
                    String identifier = json.getString("identifier");
                    String pluginName = json.getString("name");
                    loadPlugin(jar, className, identifier);
                    nameIdentifierMap.put(pluginName, identifier);
                }
            }
        }
    }
}
