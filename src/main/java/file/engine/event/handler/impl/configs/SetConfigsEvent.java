package file.engine.event.handler.impl.configs;

import file.engine.configs.ConfigEntity;
import file.engine.event.handler.Event;

public class SetConfigsEvent extends Event {

    private ConfigEntity configs;

    public SetConfigsEvent(ConfigEntity configEntity) {
        super();
        this.setBlock();
        this.configs = configEntity;
    }

    /**
     * 只允许AllConfigs进行调用，因为不能保证配置的正确性
     *
     * @return configs
     */
    public ConfigEntity getConfigs() {
        return configs;
    }

    public void setConfigs(ConfigEntity configs) {
        this.configs = configs;
    }
}
