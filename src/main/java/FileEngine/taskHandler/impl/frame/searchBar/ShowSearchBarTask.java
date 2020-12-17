package FileEngine.taskHandler.impl.frame.searchBar;

import FileEngine.taskHandler.Task;

public class ShowSearchBarTask extends Task {
    public final boolean isGrabFocus;

    public ShowSearchBarTask(boolean isGrabFocus) {
        super();
        this.isGrabFocus = isGrabFocus;
    }
}
