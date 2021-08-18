package file.engine.event.handler.impl.plugin;

public class ConfigsChangedEvent extends PluginBaseEvent {
    public final int defaultColor;
    public final int chosenColor;
    public final int borderColor;

    public ConfigsChangedEvent(int defaultColor, int chosenColor, int borderColor) {
        super(null, null);
        this.defaultColor = defaultColor;
        this.chosenColor = chosenColor;
        this.borderColor = borderColor;
    }
}
