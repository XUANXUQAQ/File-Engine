package FileEngine.taskHandler.impl.plugin;

public class SetPluginsCurrentThemeTask extends PluginBaseTask {
    public final int defaultColor;
    public final int chosenColor;
    public final int borderColor;

    public SetPluginsCurrentThemeTask(int defaultColor, int chosenColor, int borderColor) {
        super(null, null);
        this.defaultColor = defaultColor;
        this.chosenColor = chosenColor;
        this.borderColor = borderColor;
    }
}
