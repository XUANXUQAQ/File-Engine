package file.engine.frames.components;

import javax.swing.border.Border;
import java.awt.*;
import java.awt.geom.Path2D;

public class RoundBorder implements Border {
    private final Color color;
    private final int thickness;
    private final int borderRadius;
    private final int roundedCorners;
    private final int showLines;

    public static class ShowLines {
        public static final int TOP = 1;
        public static final int LEFT = 1 << 1;
        public static final int RIGHT = 1 << 2;
        public static final int BOTTOM = 1 << 3;
        public static final int ALL = TOP | LEFT | RIGHT | BOTTOM;
    }

    public static class RoundedCorners {
        public static final int TOP_LEFT = 1;
        public static final int TOP_RIGHT = 1 << 1;
        public static final int BOTTOM_LEFT = 1 << 2;
        public static final int BOTTOM_RIGHT = 1 << 3;
        public static final int ALL = TOP_LEFT | TOP_RIGHT | BOTTOM_LEFT | BOTTOM_RIGHT;

        private RoundedCorners() {
            throw new RuntimeException("should not reach here");
        }
    }

    private static class RoundShape extends Path2D.Double {
        private RoundShape(double width, double height, double radius, int roundedCorners, int showLines) {
            if ((roundedCorners & RoundedCorners.TOP_LEFT) != 0) {
                moveTo(radius, 0);
            } else {
                moveTo(0, 0);
            }
            //上方
            if ((showLines & ShowLines.TOP) != 0) {
                if ((roundedCorners & RoundedCorners.TOP_RIGHT) != 0) {
                    lineTo(width - radius, 0);
                    curveTo(width, 0, width, 0, width, radius);
                } else {
                    lineTo(width, 0);
                }
            } else {
                moveTo(width, 0);
            }
            //右方
            if ((showLines & ShowLines.RIGHT) != 0) {
                if ((roundedCorners & RoundedCorners.BOTTOM_RIGHT) != 0) {
                    lineTo(width, height - radius);
                    curveTo(width, height, width, height, width - radius, height);
                } else {
                    lineTo(width, height);
                }
            } else {
                moveTo(width, height);
            }
            //下方
            if ((showLines & ShowLines.BOTTOM) != 0) {
                if ((roundedCorners & RoundedCorners.BOTTOM_LEFT) != 0) {
                    lineTo(radius, height);
                    curveTo(0, height, 0, height, 0, height - radius);
                } else {
                    lineTo(0, height);
                }
            } else {
                moveTo(0, height);
            }
            //左方
            if ((showLines & ShowLines.LEFT) != 0) {
                if ((roundedCorners & RoundedCorners.TOP_LEFT) != 0) {
                    lineTo(0, radius);
                    curveTo(0, 0, 0, 0, radius, 0);
                } else {
                    lineTo(0, 0);
                }
            }
        }
    }


    public RoundBorder(Color color, int thickness, int roundRadius, int roundedCorners, int showLines) {
        // 有参数的构造方法
        this.color = color;
        this.thickness = thickness;
        this.borderRadius = roundRadius;
        this.roundedCorners = roundedCorners;
        this.showLines = showLines;
    }

    public Insets getBorderInsets(Component c) {
        return new Insets(0, 0, 0, 0);
    }

    @Override
    public void paintBorder(Component c, Graphics g, int x, int y, int width, int height) {
        Graphics2D graphics2d = (Graphics2D) g;
        graphics2d.setPaint(this.color);
//        graphics2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        RoundShape roundShape = new RoundShape(width, height, this.borderRadius, this.roundedCorners, this.showLines);
        BasicStroke basicStroke = new BasicStroke(this.thickness * 2);
        graphics2d.setStroke(basicStroke);
        graphics2d.draw(roundShape);
    }

    public boolean isBorderOpaque() {
        return false;
    }
}
