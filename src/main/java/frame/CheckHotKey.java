package frame;

import com.melloware.jintellitype.JIntellitype;

import javax.swing.*;
import java.util.HashMap;

public class CheckHotKey extends JFrame {

    /**
     * 利用JIntellitype实现全局热键设置
     */
    private static final long serialVersionUID = 1L;
    public static boolean isShowSearchBar = false;
    private static CheckHotKey hotKeyListener = new CheckHotKey();
    private HashMap<String, Integer> map = new HashMap<>();

    public static CheckHotKey getInstance() {
        return hotKeyListener;
    }

    //定义热键标识，用于在设置多个热键时，在事件处理中区分用户按下的热键
    public static final int FUNC_KEY_MARK = 1;

    public void setShowSearchBar(boolean b) {
        isShowSearchBar = b;
    }

    public void registerHotkey(String hotkey) {
        //解析字符串
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

        //注册热键
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

        //解析字符串
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

        //注册热键
        assert main != null;
        JIntellitype.getInstance().registerHotKey(FUNC_KEY_MARK, sum, main.charAt(0));

        //添加热键监听器
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
