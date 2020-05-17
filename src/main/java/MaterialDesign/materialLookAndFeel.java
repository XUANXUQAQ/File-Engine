package MaterialDesign;

import mdlaf.themes.MaterialLiteTheme;

import java.awt.*;

public class materialLookAndFeel extends MaterialLiteTheme {

    @Override
    protected void installFonts() {
        this.fontBold = new javax.swing.plaf.FontUIResource(Font.SANS_SERIF, Font.BOLD, 12);
        this.fontItalic = new javax.swing.plaf.FontUIResource(Font.SANS_SERIF, Font.ITALIC, 12);
        this.fontMedium = new javax.swing.plaf.FontUIResource(Font.SANS_SERIF, Font.PLAIN, 12);
        this.fontRegular = new javax.swing.plaf.FontUIResource(Font.SANS_SERIF, Font.PLAIN, 12);
    }

}
