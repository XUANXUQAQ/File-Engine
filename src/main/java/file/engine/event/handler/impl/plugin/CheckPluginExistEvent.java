package file.engine.event.handler.impl.plugin;

public class CheckPluginExistEvent extends PluginBaseEvent {
    public CheckPluginExistEvent(String pluginName) {
        super(null, pluginName);
    }
}
