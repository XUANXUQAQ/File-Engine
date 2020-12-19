package FileEngine.eventHandler.impl.plugin;

public class RemoveFromPluginsCanUpdateEvent extends PluginBaseEvent {

    public RemoveFromPluginsCanUpdateEvent(String pluginName) {
        super(null, pluginName);
    }
}
