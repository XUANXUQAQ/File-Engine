package file.engine.event.handler.impl.configs;

import file.engine.configs.ConfigEntity;
import file.engine.event.handler.Event;

public class SaveConfigsEvent extends Event {
    public final ConfigEntity configEntity;

    public SaveConfigsEvent(ConfigEntity configEntity) {
        super();
        this.configEntity = configEntity;
    }
}
