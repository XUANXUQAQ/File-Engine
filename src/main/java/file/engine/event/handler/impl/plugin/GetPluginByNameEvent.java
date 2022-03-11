package file.engine.event.handler.impl.plugin;

public class GetPluginByNameEvent extends PluginBaseEvent {

    public GetPluginByNameEvent(String pluginName) {
        super(null, pluginName);
    }
}
