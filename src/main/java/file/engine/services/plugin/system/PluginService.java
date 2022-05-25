package file.engine.services.plugin.system;

import com.google.gson.Gson;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.Constants;
import file.engine.event.handler.Event;
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.BuildEventRequestEvent;
import file.engine.event.handler.impl.frame.searchBar.SearchBarReadyEvent;
import file.engine.event.handler.impl.plugin.*;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.utils.gson.GsonUtil;
import file.engine.utils.system.properties.IsDebug;

import java.io.*;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

public class PluginService {
    private final Set<PluginInfo> pluginInfoSet = ConcurrentHashMap.newKeySet();
    private final Set<String> NOT_LATEST_PLUGINS = ConcurrentHashMap.newKeySet();
    private final HashSet<String> OLD_API_PLUGINS = new HashSet<>();
    private final HashSet<String> REPEAT_PLUGINS = new HashSet<>();
    private final HashSet<String> LOAD_ERROR_PLUGINS = new HashSet<>();
    private final PluginInfo nullPluginInfo = new PluginInfo(null, "", "");

    private static volatile PluginService INSTANCE = null;

    public static PluginService getInstance() {
        if (INSTANCE == null) {
            synchronized (PluginService.class) {
                if (INSTANCE == null) {
                    INSTANCE = new PluginService();
                }
            }
        }
        return INSTANCE;
    }

    /**
     * 检查所有的插件，若有任务栏信息则显示
     */
    private void checkPluginThread() {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                String[] message;
                Plugin plugin;
                EventManagement eventManagement = EventManagement.getInstance();
                long lastCheckEventTime = System.currentTimeMillis();
                while (eventManagement.notMainExit()) {
                    for (PluginInfo each : pluginInfoSet) {
                        plugin = each.plugin;
                        message = plugin.getMessage();
                        if (message != null) {
                            eventManagement.putEvent(new ShowTaskBarMessageEvent(message[0], message[1]));
                        }
                        Object[] registerEventHandler = plugin.pollFromEventHandlerQueue();
                        if (registerEventHandler != null) {
                            //noinspection unchecked
                            eventManagement.registerPluginHandler((String) registerEventHandler[0], (BiConsumer<Class<?>, Object>) registerEventHandler[1]);
                        }
                        String className = plugin.restoreFileEngineEventHandler();
                        if (!(className == null || className.isEmpty())) {
                            eventManagement.unregisterPluginHandler(className);
                        }
                    }
                    if (System.currentTimeMillis() - lastCheckEventTime > 1000) {
                        lastCheckEventTime = System.currentTimeMillis();
                        for (PluginInfo pluginInfo : pluginInfoSet) {
                            plugin = pluginInfo.plugin;
                            Object[] eventInfo = plugin.pollFromEventQueue();
                            // 构建事件并发出
                            if (eventInfo != null) {
                                eventManagement.putEvent(new BuildEventRequestEvent(eventInfo));
                            }
                        }
                    }
                    TimeUnit.MILLISECONDS.sleep(50);
                }
            } catch (InterruptedException ignored) {
            }
        });
    }

    private PluginService() {
        checkPluginThread();
    }

    /**
     * 检查是否有过旧的插件
     *
     * @return boolean
     */
    public boolean hasPluginTooOld() {
        return !OLD_API_PLUGINS.isEmpty();
    }

    /**
     * 检查是否有重复插件
     *
     * @return boolean
     */
    public boolean hasPluginRepeat() {
        return !REPEAT_PLUGINS.isEmpty();
    }

    /**
     * 检查是否有加载失败的插件
     *
     * @return boolean
     */
    public boolean hasPluginLoadError() {
        return !LOAD_ERROR_PLUGINS.isEmpty();
    }

    /**
     * 检查是否含有可更新的插件
     *
     * @param pluginName 插件名
     * @return boolean
     */
    public boolean hasPluginNotLatest(String pluginName) {
        return NOT_LATEST_PLUGINS.contains(pluginName);
    }

    private void removeFromPluginsCanUpdate(String pluginName) {
        NOT_LATEST_PLUGINS.remove(pluginName);
    }

    private void addPluginsCanUpdate(String pluginName) {
        NOT_LATEST_PLUGINS.add(pluginName);
    }

    /**
     * 获取所有加载失败的插件
     *
     * @return 插件名称，以逗号隔开
     */
    public String getLoadingErrorPlugins() {
        StringBuilder strb = new StringBuilder();
        for (String each : LOAD_ERROR_PLUGINS) {
            strb.append(each).append(",");
        }
        return strb.substring(0, strb.length() - 1);
    }

    public Set<PluginInfo> getAllPlugins() {
        return pluginInfoSet;
    }

    /**
     * 获取所有重复插件
     *
     * @return 插件名称，以逗号隔开
     */
    public String getRepeatPlugins() {
        StringBuilder strb = new StringBuilder();
        for (String repeatPlugins : REPEAT_PLUGINS) {
            strb.append(repeatPlugins).append(",");
        }
        return strb.substring(0, strb.length() - 1);
    }

    public ArrayList<PluginInfo> searchPluginByKeyword(String keyword) {
        ArrayList<PluginInfo> pluginInfos = new ArrayList<>();
        for (PluginInfo each : pluginInfoSet) {
            if (keyword.isBlank() || each.identifier.toLowerCase().contains(keyword)) {
                pluginInfos.add(each);
            }
        }
        return pluginInfos;
    }

    /**
     * 根据插件名获取插件实例
     *
     * @param name 插件名
     * @return 插件实例
     */
    private PluginInfo getPluginInfoByName(String name) {
        for (PluginInfo each : pluginInfoSet) {
            if (each.name.equals(name)) {
                return each;
            }
        }
        return nullPluginInfo;
    }

    /**
     * 根据插件出发关键字获取插件
     *
     * @param identifier 触发关键字
     * @return 插件实例
     */
    private PluginInfo getPluginInfoByIdentifier(String identifier) {
        for (PluginInfo each : pluginInfoSet) {
            if (each.identifier.equals(identifier)) {
                return each;
            }
        }
        return nullPluginInfo;
    }

    /**
     * 获取插件数量
     *
     * @return 插件数量
     */
    public int getInstalledPluginNum() {
        return pluginInfoSet.size();
    }

    public String getAllOldPluginsName() {
        StringBuilder strb = new StringBuilder();
        for (String oldPlugin : OLD_API_PLUGINS) {
            strb.append(oldPlugin).append(",");
        }
        return strb.substring(0, strb.length() - 1);
    }

    /**
     * 读取插件中的配置信息
     *
     * @param inputStream inputStream
     * @return 插件信息
     */
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

    /**
     * 卸载插件
     *
     * @param plugin 插件
     */
    private void unloadPlugin(Plugin plugin) {
        if (plugin != null) {
            plugin.unloadPlugin();
        }
    }

    /**
     * 加载插件
     *
     * @param pluginFile 插件jar文件路径
     * @param className  插件全类名
     * @param identifier 触发关键字
     * @param pluginName 插件名
     * @param configs    File-Engine配置信息
     * @return 是否有过旧的插件
     * @throws Exception 加载失败
     */
    private boolean loadPlugin(File pluginFile, String className, String identifier, String pluginName, Map<String, Object> configs) throws Exception {
        boolean isTooOld = false;
        URLClassLoader classLoader = new URLClassLoader(
                new URL[]{pluginFile.toURI().toURL()},
                PluginService.class.getClassLoader()
        );
        Class<?> c = Class.forName(className, true, classLoader);
        Object instance = c.getDeclaredConstructor().newInstance();
        Plugin.PluginClassAndInstanceInfo pluginClassAndInstanceInfo = new Plugin.PluginClassAndInstanceInfo(c, instance);
        Plugin plugin = new Plugin(pluginName, identifier, pluginClassAndInstanceInfo);
        plugin.loadPlugin(); // 兼容以前版本
        plugin.loadPlugin(configs);
        AllConfigs allConfigs = AllConfigs.getInstance();
        plugin.setCurrentTheme(allConfigs.getDefaultBackgroundColor(), allConfigs.getLabelColor(), allConfigs.getBorderColor());
        if (Arrays.stream(Constants.COMPATIBLE_API_VERSIONS).noneMatch(each -> each == plugin.getApiVersion())) {
            throw new RuntimeException("api version incompatible");
        }
        if (Constants.API_VERSION != plugin.getApiVersion()) {
            isTooOld = true;
        }
        pluginInfoSet.add(new PluginInfo(plugin, pluginName, identifier));
        return isTooOld;
    }

    /**
     * 卸载所有插件
     */
    private void unloadAllPlugins() {
        for (PluginInfo each : pluginInfoSet) {
            unloadPlugin(each.plugin);
        }
    }

    /**
     * 向所有插件发出配置更改事件
     *
     * @param defaultColor     背景颜色，兼容老API
     * @param chosenLabelColor 选中框颜色，兼容老API
     * @param borderColor      边框颜色，兼容老API
     * @param configs          配置信息
     */
    private void configsChanged(int defaultColor, int chosenLabelColor, int borderColor, Map<String, Object> configs) {
        for (PluginInfo each : pluginInfoSet) {
            each.plugin.setCurrentTheme(defaultColor, chosenLabelColor, borderColor); // 兼容以前版本
            each.plugin.configsChanged(configs);
        }
    }

    /**
     * 检查该插件名所对应的插件是否存在
     *
     * @param pluginName 插件名
     * @return boolean
     */
    private boolean hasPlugin(String pluginName) {
        for (PluginInfo each : pluginInfoSet) {
            if (each.name.equals(pluginName)) {
                return true;
            }
        }
        return false;
    }

    public Object[] getPluginNameArray() {
        ArrayList<String> list = new ArrayList<>();
        for (PluginInfo each : pluginInfoSet) {
            list.add(each.name);
        }
        return list.toArray();
    }

    /**
     * 加载所有插件
     *
     * @param pluginPath 插件目录
     */
    private void loadAllPlugins(String pluginPath) {
        FilenameFilter filter = (dir, name) -> name.endsWith(".jar");
        File[] files = new File(pluginPath).listFiles(filter);
        if (files == null || files.length == 0) {
            return;
        }
        Gson gson = GsonUtil.getInstance().getGson();
        for (File jar : files) {
            try (JarFile jarFile = new JarFile(jar)) {
                Enumeration<?> enu = jarFile.entries();
                while (enu.hasMoreElements()) {
                    JarEntry element = (JarEntry) enu.nextElement();
                    String name = element.getName();
                    if ("PluginInfo.json".equals(name)) {
                        String pluginInfo = readAll(jarFile.getInputStream(element));
                        //noinspection unchecked
                        Map<String, Object> json = gson.fromJson(pluginInfo, Map.class);
                        String className = (String) json.get("className");
                        String identifier = (String) json.get("identifier");
                        String pluginName = (String) json.get("name");
                        if (!hasPlugin(pluginName)) {
                            try {
                                boolean isPluginApiTooOld = loadPlugin(jar, className, identifier, pluginName, AllConfigs.getInstance().getConfigMap());
                                if (isPluginApiTooOld) {
                                    OLD_API_PLUGINS.add(pluginName);
                                }
                            } catch (Exception e) {
                                LOAD_ERROR_PLUGINS.add(pluginName);
                                e.printStackTrace();
                            }
                        } else {
                            REPEAT_PLUGINS.add(jar.getName());
                        }
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void checkAllPluginsVersion(StringBuilder oldPlugins) {
        boolean isLatest;
        for (PluginInfo each : pluginInfoSet) {
            try {
                isLatest = isPluginLatest(each.plugin);
                if (IsDebug.isDebug()) {
                    System.out.println("++++++++++++++++++++++++++++++++++++++++++++");
                    System.out.println("插件：" + each.name + "已检查完毕，结果：" + isLatest);
                    System.out.println("++++++++++++++++++++++++++++++++++++++++++++");
                }
                if (!isLatest) {
                    oldPlugins.append(each.name).append(" ");
                    addPluginsCanUpdate(each.name);
                }
            } catch (Exception e) {
                if (IsDebug.isDebug()) {
                    System.err.println("++++++++++++++++++++++++++++++++++++++++++++");
                    System.err.println("插件：" + each + "检查更新失败");
                    System.err.println("++++++++++++++++++++++++++++++++++++++++++++");
                }
            }
        }
    }

    /**
     * 向所有插件发出窗口已打开事件
     *
     * @param showingType 显示模式
     */
    private void onSearchBarVisible(String showingType) {
        for (PluginInfo pluginInfo : pluginInfoSet) {
            pluginInfo.plugin.searchBarVisible(showingType);
        }
    }

    private boolean isPluginLatest(Plugin plugin) throws InterruptedException {
        final long checkSuccess = 0x100L;
        AtomicLong startCheckTime = new AtomicLong();
        AtomicBoolean isVersionLatest = new AtomicBoolean();
        Thread checkUpdateThread = new Thread(() -> {
            startCheckTime.set(System.currentTimeMillis());
            try {
                isVersionLatest.set(plugin.isLatest());
                if (!Thread.interrupted()) {
                    startCheckTime.set(checkSuccess); //表示检查成功
                }
            } catch (Exception e) {
                e.printStackTrace();
                startCheckTime.set(0xFFFL);
            }
        });
        CachedThreadPoolUtil.getInstance().executeTask(checkUpdateThread);
        //等待获取插件更新信息
        try {
            while (startCheckTime.get() != checkSuccess) {
                TimeUnit.MILLISECONDS.sleep(200);
                if ((System.currentTimeMillis() - startCheckTime.get() > 5000L && startCheckTime.get() != checkSuccess) || startCheckTime.get() == 0xFFFL) {
                    checkUpdateThread.interrupt();
                    throw new InterruptedException("check update failed.");
                }
                if (!EventManagement.getInstance().notMainExit()) {
                    break;
                }
            }
            return isVersionLatest.get();
        } catch (InterruptedException e) {
            throw new InterruptedException("check update failed.");
        }
    }

    @EventRegister(registerClass = CheckPluginExistEvent.class)
    private static void checkPluginExistEvent(Event event) {
        CheckPluginExistEvent event1 = (CheckPluginExistEvent) event;
        boolean b = getInstance().hasPlugin(event1.pluginName);
        event.setReturnValue(b);
    }

    @EventRegister(registerClass = GetPluginByNameEvent.class)
    private static void getPluginByNameEvent(Event event) {
        GetPluginByNameEvent event1 = (GetPluginByNameEvent) event;
        PluginInfo pluginInfoByName = getInstance().getPluginInfoByName(event1.pluginName);
        event1.setReturnValue(pluginInfoByName);
    }

    @EventRegister(registerClass = GetPluginByIdentifierEvent.class)
    private static void getPluginByIdentifier(Event event) {
        GetPluginByIdentifierEvent event1 = (GetPluginByIdentifierEvent) event;
        PluginInfo pluginInfoByIdentifier = getInstance().getPluginInfoByIdentifier(event1.identifier);
        event1.setReturnValue(pluginInfoByIdentifier);
    }

    @EventRegister(registerClass = AddPluginsCanUpdateEvent.class)
    private static void addPluginsCanUpdateEvent(Event event) {
        getInstance().addPluginsCanUpdate(((AddPluginsCanUpdateEvent) event).pluginName);
    }

    @EventRegister(registerClass = LoadAllPluginsEvent.class)
    private static void loadAllPluginsEvent(Event event) {
        getInstance().loadAllPlugins(((LoadAllPluginsEvent) event).pluginDirPath);
    }

    @EventRegister(registerClass = RemoveFromPluginsCanUpdateEvent.class)
    private static void removeFromPluginsCanUpdateEvent(Event event) {
        getInstance().removeFromPluginsCanUpdate(((RemoveFromPluginsCanUpdateEvent) event).pluginName);
    }

    @EventRegister(registerClass = ConfigsChangedEvent.class)
    private static void setPluginsCurrentThemeEvent(Event event) {
        ConfigsChangedEvent task1 = (ConfigsChangedEvent) event;
        getInstance().configsChanged(task1.defaultColor, task1.chosenColor, task1.borderColor, AllConfigs.getInstance().getConfigMap());
    }

    @EventListener(listenClass = SearchBarReadyEvent.class)
    private static void onSearchBarReady(Event event) {
        SearchBarReadyEvent searchBarReadyEvent = (SearchBarReadyEvent) event;
        getInstance().onSearchBarVisible(searchBarReadyEvent.showingType);
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void restartEvent(Event event) {
        getInstance().unloadAllPlugins();
    }

    @EventRegister(registerClass = EventProcessedBroadcastEvent.class)
    private static void broadcastEventProcess(Event event) {
        EventProcessedBroadcastEvent eventProcessed = (EventProcessedBroadcastEvent) event;
        PluginService pluginService = getInstance();
        for (PluginInfo each : pluginService.pluginInfoSet) {
            each.plugin.eventProcessed(eventProcessed.c, eventProcessed.eventInstance);
        }
    }

    public static class PluginInfo {
        public final Plugin plugin;
        public final String name;
        public final String identifier;

        private PluginInfo(Plugin plugin, String name, String identifier) {
            this.plugin = plugin;
            this.name = name;
            this.identifier = identifier;
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
