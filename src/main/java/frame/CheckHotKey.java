package frame;

import com.melloware.jintellitype.JIntellitype;

import javax.swing.*;
import java.util.HashMap;

public class CheckHotKey extends JFrame {

    /**
     * ����JIntellitypeʵ��ȫ���ȼ�����
     */
    private static final long serialVersionUID = 1L;
    public static boolean isShowSearchBar = false;
    private static CheckHotKey hotKeyListener = new CheckHotKey();
    private HashMap<String, Integer> map = new HashMap<>();

    public static CheckHotKey getInstance() {
        return hotKeyListener;
    }

    //�����ȼ���ʶ�����������ö���ȼ�ʱ�����¼������������û����µ��ȼ�
    public static final int FUNC_KEY_MARK = 1;

    public void setShowSearchBar(boolean b) {
        isShowSearchBar = b;
    }

    public void registerHotkey(String hotkey) {
        //�����ַ���
        String[] hotkeys = hotkey.split(" \\+ ");
        int sum = 0;
        String main = null;
        for (String each : hotkeys) {
            if (each.length() != 1) {
                sum += map.get(each);
            } else {
                main = each;
            }
        }

        //ע���ȼ�
        assert main != null;
        JIntellitype.getInstance().unregisterHotKey(FUNC_KEY_MARK);
        JIntellitype.getInstance().registerHotKey(FUNC_KEY_MARK, sum, main.charAt(0));
    }

    public void unregisterHotkey() {
        JIntellitype.getInstance().unregisterHotKey(FUNC_KEY_MARK);
    }

    private CheckHotKey() {
        this.setBounds(100, 100, 600, 400);
        this.setLayout(null);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        map.put("Ctrl", JIntellitype.MOD_CONTROL);
        map.put("Alt", JIntellitype.MOD_ALT);
        map.put("Shift", JIntellitype.MOD_SHIFT);
        map.put("Win", JIntellitype.MOD_WIN);

        //�����ַ���
        String[] hotkeys = SettingsFrame.hotkey.split(" \\+ ");
        int sum = 0;
        String main = null;
        for (String each : hotkeys) {
            if (each.length() != 1) {
                sum += map.get(each);
            } else {
                main = each;
            }
        }

        //ע���ȼ�
        assert main != null;
        JIntellitype.getInstance().registerHotKey(FUNC_KEY_MARK, sum, main.charAt(0));

        //����ȼ�������
        JIntellitype.getInstance().addHotKeyListener(markCode -> {
            if (markCode == FUNC_KEY_MARK) {
                isShowSearchBar = !isShowSearchBar;
                SearchBar searchBar = SearchBar.getInstance();
                if (isShowSearchBar) {

                    searchBar.showSearchbar();
                } else {
                    searchBar.closedTodo();
                }
            }
        });
    }
}
