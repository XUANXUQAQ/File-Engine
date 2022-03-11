package file.engine.event.handler.impl.plugin;

public class GetPluginByIdentifierEvent extends PluginBaseEvent {
    public final String identifier;

    public GetPluginByIdentifierEvent(String pluginIdentifier) {
        super(null, null);
        this.identifier = pluginIdentifier;
    }
}
