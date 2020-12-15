package FileEngine.r;

import java.awt.*;
import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;

public class R {

    private final ConcurrentHashMap<String, Component> NAME_COMPONENT_MAPPER = new ConcurrentHashMap<>();
    private static volatile R instance;

    private R() {}

    public static R getInstance() {
        if (instance == null) {
            synchronized (R.class) {
                if (instance == null) {
                    instance = new R();
                }
            }
        }
        return instance;
    }

    public void addComponent(String name, Component component) {
        NAME_COMPONENT_MAPPER.put(name, component);
    }

    public ArrayList<Component> getAllComponents() {
        return new ArrayList<>(NAME_COMPONENT_MAPPER.values());
    }
}
