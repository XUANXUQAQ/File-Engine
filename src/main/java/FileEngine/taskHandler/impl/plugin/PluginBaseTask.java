package FileEngine.taskHandler.impl.plugin;

import FileEngine.taskHandler.Task;

public class PluginBaseTask extends Task {

    public final String pluginDirPath;
    public final String pluginName;

    protected PluginBaseTask(String pluginDirPath, String pluginName) {
        super();
        this.pluginDirPath = pluginDirPath;
        this.pluginName = pluginName;
    }
}
