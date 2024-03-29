package file.engine.event.handler.impl.plugin;

import file.engine.event.handler.Event;

public class PluginBaseEvent extends Event {

    // 仅在加载时遍历使用
    public final String pluginDirPath;
    public final String pluginName;

    protected PluginBaseEvent(String pluginDirPath, String pluginName) {
        super();
        this.pluginDirPath = pluginDirPath;
        this.pluginName = pluginName;
    }
}
