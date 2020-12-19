package FileEngine.eventHandler.impl.plugin;

import FileEngine.eventHandler.Event;

public class PluginBaseEvent extends Event {

    public final String pluginDirPath;
    public final String pluginName;

    protected PluginBaseEvent(String pluginDirPath, String pluginName) {
        super();
        this.pluginDirPath = pluginDirPath;
        this.pluginName = pluginName;
    }
}
