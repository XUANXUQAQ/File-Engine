package file.engine.event.handler.impl.frame.searchBar;

import file.engine.configs.Constants;
import file.engine.event.handler.Event;

public class SetBorderEvent extends Event {

    public final Constants.Enums.BorderType borderType;
    public final int borderColor;
    public final int borderThickness;

    public SetBorderEvent(Constants.Enums.BorderType borderType, int color, int borderThickness) {
        this.borderType = borderType;
        this.borderColor = color;
        this.borderThickness = borderThickness;
    }
}
