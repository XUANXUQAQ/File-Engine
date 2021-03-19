package file.engine.event.handler.impl.configs;

import file.engine.event.handler.Event;

public class CmdModifyEvent extends Event {
    public final String cmd;
    public CmdModifyEvent(String cmd) {
        this.cmd = cmd;
    }
}
