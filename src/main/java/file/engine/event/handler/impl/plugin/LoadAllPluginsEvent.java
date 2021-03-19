package file.engine.event.handler.impl.plugin;

public class LoadAllPluginsEvent extends PluginBaseEvent {

    public LoadAllPluginsEvent(String pluginDirPath) {
        super(pluginDirPath, null);
    }
}
