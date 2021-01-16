package FileEngine.eventHandler.impl.configs;

import FileEngine.eventHandler.Event;

public class CmdModifyEvent extends Event {
    public final String cmd;
    public CmdModifyEvent(String cmd) {
        this.cmd = cmd;
    }
}
