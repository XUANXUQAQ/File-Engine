package FileEngine.utils.pluginSystem;

class PluginClassAndInstanceInfo {
    public PluginClassAndInstanceInfo(Class<?> cls, Object instance) {
        this.cls = cls;
        this.clsInstance = instance;
    }

    public final Class<?> cls;
    public final Object clsInstance;
}
