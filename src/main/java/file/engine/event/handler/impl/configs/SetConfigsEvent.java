package file.engine.event.handler.impl.configs;

import file.engine.configs.ConfigEntity;
import file.engine.event.handler.Event;

public class SetConfigsEvent extends Event {

    private ConfigEntity configs;
    public SetConfigsEvent(ConfigEntity configEntity) {
        super();
        this.configs = configEntity;
    }

    public ConfigEntity getConfigs() {
        return configs;
    }

    public void setConfigs(ConfigEntity configs) {
        this.configs = configs;
    }
}
