package FileEngine.taskHandler.impl.frame.searchBar;

import FileEngine.taskHandler.Task;

public class SetSearchBarTransparencyTask extends Task {
    public final float trans;

    public SetSearchBarTransparencyTask(float trans) {
        super();
        this.trans = trans;
    }
}
