package file.engine.event.handler.impl.configs;

import file.engine.event.handler.Event;

public class SetSwingLaf extends Event {
    public final String theme;

    /**
     * 改变主题
     * @param theme 主题名
     */
    public SetSwingLaf(String theme) {
        super();
        if (null == theme || theme.isEmpty()) {
            throw new RuntimeException("theme cannot be null or empty");
        }
        this.theme = theme;
    }

    /**
     * 使用当前配置中的主题或默认主题
     */
    public SetSwingLaf() {
        super();
        this.theme = "";
    }
}
