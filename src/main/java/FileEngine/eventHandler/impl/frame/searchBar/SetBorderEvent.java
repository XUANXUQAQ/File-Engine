package FileEngine.eventHandler.impl.frame.searchBar;

import FileEngine.configs.Enums;
import FileEngine.eventHandler.Event;

public class SetBorderEvent extends Event {

    public final Enums.BorderType borderType;
    public final int borderColor;
    public final int borderThickness;
    public SetBorderEvent(Enums.BorderType borderType, int color, int borderThickness) {
        this.borderType = borderType;
        this.borderColor = color;
        this.borderThickness = borderThickness;
    }
}
