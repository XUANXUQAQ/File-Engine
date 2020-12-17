package FileEngine.taskHandler.impl.taskbar;

import FileEngine.taskHandler.Task;

public class ShowTaskBarMessageTask extends Task {
    public final String caption;
    public final String message;

    public ShowTaskBarMessageTask(String caption, String message) {
        super();
        this.caption = caption;
        this.message = message;
    }
}
