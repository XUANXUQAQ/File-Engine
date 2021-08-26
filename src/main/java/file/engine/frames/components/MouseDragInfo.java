package file.engine.frames.components;

import file.engine.dllInterface.GetHandle;
import file.engine.utils.CachedThreadPoolUtil;
import file.engine.utils.GetIconUtil;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.RoundRectangle2D;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public class MouseDragInfo extends JFrame {
    private final JLabel iconLabel = new JLabel("icon");
    private final JPanel jPanel = new JPanel();

    public MouseDragInfo() {
        jPanel.setBackground(new Color(193, 251, 255, 100));
        iconLabel.setFocusable(false);
        iconLabel.setForeground(new Color(0, 0, 0, 0));
        jPanel.add(iconLabel);
        jPanel.setLayout(null);
        this.setContentPane(jPanel);
        this.setDefaultCloseOperation(HIDE_ON_CLOSE);
        this.setUndecorated(true);
        this.setType(Type.UTILITY);
        this.setFocusable(false);
        this.setOpacity(0.3f);
    }

    /**
     * 显示鼠标拖动信息
     *
     * @param result        信息
     * @param pointSupplier 位置信息
     * @param isFinished    是否结束
     */
    public void showDragInfo(String result, Supplier<Point> pointSupplier, Supplier<Boolean> isFinished, Runnable callback) {
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize(); // 获取屏幕大小
        double dpi = GetHandle.INSTANCE.getDpi();
        int length = (int) ((screenSize.width / 20.0) * dpi);
        int iconLength = (int) (length * 0.6);
        int offset = (length - iconLength) / 2;
        Point point = pointSupplier.get();
        this.setBounds(point.x - length / 2, point.y - length + offset, length, length);
        jPanel.setSize(length, length);
        ImageIcon bigIcon = GetIconUtil.getInstance().getBigIcon(result, iconLength, iconLength);
        iconLabel.setIcon(bigIcon);
        iconLabel.setBounds(offset, offset, iconLength, iconLength);
        this.setShape(new RoundRectangle2D.Double(0, 0, length, length, 15, 15));
        this.setVisible(true);
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                Point _point;
                while (!isFinished.get()) {
                    TimeUnit.MILLISECONDS.sleep(10);
                    _point = pointSupplier.get();
                    this.setBounds(_point.x - length / 2, _point.y - length + offset, length, length);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            this.setVisible(false);
            this.dispose();
            callback.run();
        });
    }
}
