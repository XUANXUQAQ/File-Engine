package FileEngine.eventHandler.impl.frame.searchBar;

import FileEngine.eventHandler.Event;

public class SetPreviewOrNormalMode extends Event {
    public final boolean isPreview;
    public SetPreviewOrNormalMode(boolean isPreview) {
        super();
        this.isPreview = isPreview;
    }
}
