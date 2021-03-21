package file.engine.event.handler.impl.frame.searchBar;

import file.engine.configs.Enums;
import file.engine.event.handler.Event;

public class PreviewSearchBarEvent extends Event {
    public final int borderColor;
    public final int searchBarColor;
    public final int searchBarFontColor;
    public final int chosenLabelColor;
    public final int chosenLabelFontColor;
    public final int unchosenLabelFontColor;
    public final int defaultBackgroundColor;
    public final Enums.BorderType borderType;
    public final int borderThickness;

    public PreviewSearchBarEvent(String borderColor,
                                 String searchBarColor,
                                 String searchBarFontColor,
                                 String chosenLabelColor,
                                 String chosenLabelFontColor,
                                 String unchosenLabelFontColor,
                                 String defaultBackgroundColor,
                                 Enums.BorderType borderType,
                                 String borderThickness) {
        super();
        this.borderColor = Integer.parseInt(borderColor, 16);
        this.searchBarColor = Integer.parseInt(searchBarColor, 16);
        this.searchBarFontColor = Integer.parseInt(searchBarFontColor, 16);
        this.chosenLabelColor = Integer.parseInt(chosenLabelColor, 16);
        this.chosenLabelFontColor = Integer.parseInt(chosenLabelFontColor, 16);
        this.unchosenLabelFontColor = Integer.parseInt(unchosenLabelFontColor, 16);
        this.defaultBackgroundColor = Integer.parseInt(defaultBackgroundColor, 16);
        this.borderThickness = Integer.parseInt(borderThickness);
        this.borderType = borderType;
    }
}
