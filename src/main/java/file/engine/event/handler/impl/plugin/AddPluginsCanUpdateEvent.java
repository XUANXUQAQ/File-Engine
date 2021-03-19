package file.engine.event.handler.impl.plugin;

public class AddPluginsCanUpdateEvent extends PluginBaseEvent {

    public AddPluginsCanUpdateEvent(String pluginName) {
        super(null, pluginName);
    }
}
