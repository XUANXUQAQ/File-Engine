package frame;
import javax.swing.JFrame;
import com.melloware.jintellitype.JIntellitype;
import search.Search;

import java.util.HashMap;

public class CheckHotKey extends JFrame{

    private Search search = new Search();

    /**
     * ����JIntellitypeʵ��ȫ���ȼ�����
     *
     */
        private static final long serialVersionUID = 1L;
        private static HashMap<String, Integer> map = new HashMap<>();

        //�����ȼ���ʶ�����������ö���ȼ�ʱ�����¼������������û����µ��ȼ�
        public static final int FUNC_KEY_MARK = 1;

        public void registHotkey(String hotkey){
            //�����ַ���
            String[] hotkeys = hotkey.split(" \\+ ");
            int sum = 0;
            String main = null;
            for (String each:hotkeys){
                if (each.length() != 1){
                    sum += map.get(each);
                }else{
                    main = each;
                }
            }

            //ע���ȼ�
            assert main != null;
            JIntellitype.getInstance().unregisterHotKey(FUNC_KEY_MARK);
            JIntellitype.getInstance().registerHotKey(FUNC_KEY_MARK, sum, main.charAt(0));
        }

        public CheckHotKey() {
            this.setBounds(100, 100, 600, 400);
            this.setLayout(null);
            this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            map.put("Ctrl", JIntellitype.MOD_CONTROL);
            map.put("Alt", JIntellitype.MOD_ALT);
            map.put("Shift", JIntellitype.MOD_ALT);
            map.put("Win", JIntellitype.MOD_WIN);

            //�����ַ���
            String[] hotkeys = SettingsFrame.hotkey.split(" \\+ ");
            int sum = 0;
            String main = null;
            for (String each:hotkeys){
                if (each.length() != 1){
                    sum += map.get(each);
                }else{
                    main = each;
                }
            }

            //ע���ȼ�
            assert main != null;
            JIntellitype.getInstance().registerHotKey(FUNC_KEY_MARK, sum, main.charAt(0));

            //����ȼ�������
            JIntellitype.getInstance().addHotKeyListener(markCode -> {
                if (markCode == FUNC_KEY_MARK) {
                    CheckHotKey.this.changeShowSearchBarStatus();
                }
            });
        }
        private void changeShowSearchBarStatus(){
            search.setFocusLostStatus(false);
        }
}
