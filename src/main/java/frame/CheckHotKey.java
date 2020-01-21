package frame;
import javax.swing.JFrame;
import com.melloware.jintellitype.JIntellitype;

public class CheckHotKey extends JFrame{
    public static boolean isShowSearachBar = false;

    /**
     * ����JIntellitypeʵ��ȫ���ȼ�����
     *
     */
        private static final long serialVersionUID = 1L;

        //�����ȼ���ʶ�����������ö���ȼ�ʱ�����¼������������û����µ��ȼ�
        public static final int FUNC_KEY_MARK = 1;

        public CheckHotKey() {
            this.setBounds(100, 100, 600, 400);
            this.setLayout(null);
            this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            //��һ����ע���ȼ�����һ��������ʾ���ȼ��ı�ʶ���ڶ���������ʾ��ϼ������û����Ϊ0������������Ϊ�������Ҫ�ȼ�
            JIntellitype.getInstance().registerHotKey(FUNC_KEY_MARK, JIntellitype.MOD_CONTROL + JIntellitype.MOD_ALT, 'J');

            //�ڶ���������ȼ�������
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
