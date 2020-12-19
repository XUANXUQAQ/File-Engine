package FileEngine.eventHandler.impl.plugin;

public class SetPluginsCurrentThemeEvent extends PluginBaseEvent {
    public final int defaultColor;
    public final int chosenColor;
    public final int borderColor;

    public SetPluginsCurrentThemeEvent(int defaultColor, int chosenColor, int borderColor) {
        super(null, null);
        this.defaultColor = defaultColor;
        this.chosenColor = chosenColor;
        this.borderColor = borderColor;
    }
}
