package file.engine.event.handler.impl.plugin;

import file.engine.event.handler.Event;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

import java.util.LinkedHashMap;

@EqualsAndHashCode(callSuper = true)
@Data
@NoArgsConstructor
public class PluginRegisterEvent extends Event {

    private String classFullName;

    private LinkedHashMap<String, Object> params = new LinkedHashMap<>();

}
