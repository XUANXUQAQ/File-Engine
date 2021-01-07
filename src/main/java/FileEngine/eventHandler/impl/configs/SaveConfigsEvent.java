package FileEngine.eventHandler.impl.configs;

import FileEngine.configs.ConfigEntity;
import FileEngine.eventHandler.Event;

public class SaveConfigsEvent extends Event {
    public final ConfigEntity configEntity;

    public SaveConfigsEvent(ConfigEntity configEntity) {
        super();
        this.configEntity = configEntity;
    }
}
