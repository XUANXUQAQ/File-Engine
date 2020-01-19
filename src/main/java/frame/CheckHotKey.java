package frame;
import javax.swing.JFrame;
import com.melloware.jintellitype.JIntellitype;

public class CheckHotKey extends JFrame{
    public static boolean isShowSearachBar = false;

    /**
     * 利用JIntellitype实现全局热键设置
     *
     */
        private static final long serialVersionUID = 1L;

        //定义热键标识，用于在设置多个热键时，在事件处理中区分用户按下的热键
        public static final int FUNC_KEY_MARK = 1;

        public CheckHotKey() {
            this.setBounds(100, 100, 600, 400);
            this.setLayout(null);
            this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            //第一步：注册热键，第一个参数表示该热键的标识，第二个参数表示组合键，如果没有则为0，第三个参数为定义的主要热键
            JIntellitype.getInstance().registerHotKey(FUNC_KEY_MARK, JIntellitype.MOD_CONTROL + JIntellitype.MOD_ALT, 'J');

            //第二步：添加热键监听器
            JIntellitype.getInstance().addHotKeyListener(markCode -> {
                if (markCode == FUNC_KEY_MARK) {
                    CheckHotKey.this.changeShowSearchBarStatus();
                }
            });
        }
        private void changeShowSearchBarStatus(){
            isShowSearachBar = true;
        }
}
