package FileEngine.taskHandler.impl.hotkey;

import FileEngine.taskHandler.Task;

public class HotKeyTask extends Task {
    public final String hotkey;

    protected HotKeyTask(String hotkey) {
        super();
        this.hotkey = hotkey;
    }
}
