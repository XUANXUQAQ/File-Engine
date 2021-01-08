package FileEngine.utils.pinyin;

import java.util.ArrayList;

public class SeparateTool {

    private static final ArrayList<ArrayList<String>> ym = new ArrayList<>();// 单字集合
    private static final ArrayList<String> sm = new ArrayList<>();// 声母集合
    private static final ArrayList<String> yy = new ArrayList<>();// 单字集合
    private static final ArrayList<String> ym_b = new ArrayList<>();// 声母b的韵母集合
    private static final ArrayList<String> ym_c = new ArrayList<>();// 声母c的韵母集合
    private static final ArrayList<String> ym_d = new ArrayList<>();// 声母d的韵母集合
    private static final ArrayList<String> ym_f = new ArrayList<>();// 声母f的韵母集合
    private static final ArrayList<String> ym_g = new ArrayList<>();// 声母g的韵母集合
    private static final ArrayList<String> ym_h = new ArrayList<>();// 声母h的韵母集合
    private static final ArrayList<String> ym_j = new ArrayList<>();// 声母j的韵母集合
    private static final ArrayList<String> ym_k = new ArrayList<>();// 声母k的韵母集合
    private static final ArrayList<String> ym_l = new ArrayList<>();// 声母l的韵母集合
    private static final ArrayList<String> ym_m = new ArrayList<>();// 声母m的韵母集合
    private static final ArrayList<String> ym_n = new ArrayList<>();// 声母n的韵母集合
    private static final ArrayList<String> ym_p = new ArrayList<>();// 声母p的韵母集合
    private static final ArrayList<String> ym_q = new ArrayList<>();// 声母q的韵母集合
    private static final ArrayList<String> ym_r = new ArrayList<>();// 声母r的韵母集合
    private static final ArrayList<String> ym_s = new ArrayList<>();// 声母s的韵母集合
    private static final ArrayList<String> ym_t = new ArrayList<>();// 声母t的韵母集合
    private static final ArrayList<String> ym_w = new ArrayList<>();// 声母w的韵母集合
    private static final ArrayList<String> ym_x = new ArrayList<>();// 声母x的韵母集合
    private static final ArrayList<String> ym_y = new ArrayList<>();// 声母y的韵母集合
    private static final ArrayList<String> ym_z = new ArrayList<>();// 声母z的韵母集合
    private static final ArrayList<String> ym_sh = new ArrayList<>();// 声母sh的韵母集合
    private static final ArrayList<String> ym_zh = new ArrayList<>();// 声母zh的韵母集合
    private static final ArrayList<String> ym_ch = new ArrayList<>();// 声母ch的韵母集合

    static {
        sm.add("b");
        sm.add("c");
        sm.add("d");
        sm.add("f");
        sm.add("g");
        sm.add("h");
        sm.add("j");
        sm.add("k");
        sm.add("l");
        sm.add("m");
        sm.add("n");
        sm.add("p");
        sm.add("q");
        sm.add("r");
        sm.add("s");
        sm.add("t");
        sm.add("w");
        sm.add("x");
        sm.add("y");
        sm.add("z");
        sm.add("sh");
        sm.add("zh");
        sm.add("ch");

        yy.add("a");//
        yy.add("ai");
        yy.add("an");
        yy.add("ang");
        yy.add("ao");
        yy.add("e");
        yy.add("en");
        yy.add("eng");
        yy.add("er");
        yy.add("o");
        yy.add("ou");
        yy.add("ong");

        ym_b.add("a");//
        ym_b.add("ai");
        ym_b.add("an");
        ym_b.add("ang");
        ym_b.add("ao");
        ym_b.add("ei");
        ym_b.add("en");
        ym_b.add("eng");
        ym_b.add("i");
        ym_b.add("ian");
        ym_b.add("iao");
        ym_b.add("ie");
        ym_b.add("in");
        ym_b.add("ing");
        ym_b.add("o");
        ym_b.add("u");

        ym_c.add("a");// c
        ym_c.add("ai");
        ym_c.add("an");
        ym_c.add("ang");
        ym_c.add("ao");
        ym_c.add("e");
        ym_c.add("en");
        ym_c.add("eng");
        ym_c.add("i");
        ym_c.add("ong");
        ym_c.add("ou");
        ym_c.add("u");
        ym_c.add("uan");
        ym_c.add("ui");
        ym_c.add("un");
        ym_c.add("uo");

        ym_d.add("a");// d
        ym_d.add("ai");
        ym_d.add("an");
        ym_d.add("ang");
        ym_d.add("ao");
        ym_d.add("e");
        ym_d.add("ei");
        ym_d.add("en");
        ym_d.add("eng");
        ym_d.add("i");
        ym_d.add("ia");
        ym_d.add("ian");
        ym_d.add("iao");
        ym_d.add("ie");
        ym_d.add("ing");
        ym_d.add("iu");
        ym_d.add("ong");
        ym_d.add("ou");
        ym_d.add("u");
        ym_d.add("uan");
        ym_d.add("ui");
        ym_d.add("un");
        ym_d.add("uo");

        ym_f.add("a");// f
        ym_f.add("an");
        ym_f.add("ang");
        ym_f.add("ei");
        ym_f.add("en");
        ym_f.add("eng");
        ym_f.add("iao");
        ym_f.add("o");
        ym_f.add("ou");
        ym_f.add("u");

        ym_g.add("a");// g
        ym_g.add("ai");
        ym_g.add("an");
        ym_g.add("ang");
        ym_g.add("ao");
        ym_g.add("e");
        ym_g.add("ei");
        ym_g.add("en");
        ym_g.add("eng");
        ym_g.add("ong");
        ym_g.add("ou");
        ym_g.add("u");
        ym_g.add("uai");
        ym_g.add("uan");
        ym_g.add("uang");
        ym_g.add("ui");
        ym_g.add("un");
        ym_g.add("uo");

        ym_h.add("a");// h
        ym_h.add("ai");
        ym_h.add("an");
        ym_h.add("ang");
        ym_h.add("ao");
        ym_h.add("e");
        ym_h.add("ei");
        ym_h.add("en");
        ym_h.add("eng");
        ym_h.add("ong");
        ym_h.add("ou");
        ym_h.add("u");
        ym_h.add("ua");
        ym_h.add("uai");
        ym_h.add("uan");
        ym_h.add("uang");
        ym_h.add("ui");
        ym_h.add("un");
        ym_h.add("uo");

        ym_j.add("i");// j
        ym_j.add("ia");
        ym_j.add("ian");
        ym_j.add("iang");
        ym_j.add("iao");
        ym_j.add("ie");
        ym_j.add("in");
        ym_j.add("ing");
        ym_j.add("iong");
        ym_j.add("iu");
        ym_j.add("u");
        ym_j.add("uan");
        ym_j.add("ue");
        ym_j.add("un");

        ym_k.add("a");//
        ym_k.add("ai");
        ym_k.add("an");
        ym_k.add("ang");
        ym_k.add("ao");
        ym_k.add("e");
        ym_k.add("en");
        ym_k.add("eng");
        ym_k.add("ong");
        ym_k.add("ou");
        ym_k.add("u");
        ym_k.add("ui");
        ym_k.add("un");
        ym_k.add("uo");

        ym_l.add("a");// l
        ym_l.add("ai");
        ym_l.add("an");
        ym_l.add("ang");
        ym_l.add("ao");
        ym_l.add("e");
        ym_l.add("ei");
        ym_l.add("eng");
        ym_l.add("i");
        ym_l.add("ia");
        ym_l.add("ian");
        ym_l.add("iao");
        ym_l.add("ie");
        ym_l.add("in");
        ym_l.add("ing");
        ym_l.add("iu");
        ym_l.add("o");
        ym_l.add("ong");
        ym_l.add("ou");
        ym_l.add("u");
        ym_l.add("uan");
        ym_l.add("un");
        ym_l.add("uo");
        ym_l.add("v");
        ym_l.add("ve");

        ym_m.add("a");// m
        ym_m.add("ai");
        ym_m.add("an");
        ym_m.add("ang");
        ym_m.add("ao");
        ym_m.add("e");
        ym_m.add("ei");
        ym_m.add("en");
        ym_m.add("eng");
        ym_m.add("i");
        ym_m.add("ian");
        ym_m.add("iao");
        ym_m.add("ie");
        ym_m.add("in");
        ym_m.add("ing");
        ym_m.add("iu");
        ym_m.add("o");
        ym_m.add("ou");
        ym_m.add("u");

        ym_n.add("a");// n
        ym_n.add("ai");
        ym_n.add("an");
        ym_n.add("ang");
        ym_n.add("ao");
        ym_n.add("e");
        ym_n.add("ei");
        ym_n.add("en");
        ym_n.add("eng");
        ym_n.add("i");
        ym_n.add("ian");
        ym_n.add("iang");
        ym_n.add("iao");
        ym_n.add("ie");
        ym_n.add("in");
        ym_n.add("ing");
        ym_n.add("iu");
        ym_n.add("ong");
        ym_n.add("ou");
        ym_n.add("u");
        ym_n.add("uan");
        ym_n.add("un");
        ym_n.add("uo");
        ym_n.add("v");
        ym_n.add("ve");

        ym_p.add("a");// p
        ym_p.add("ai");
        ym_p.add("an");
        ym_p.add("ang");
        ym_p.add("ao");
        ym_p.add("e");
        ym_p.add("ei");
        ym_p.add("en");
        ym_p.add("eng");
        ym_p.add("i");
        ym_p.add("ian");
        ym_p.add("iao");
        ym_p.add("ie");
        ym_p.add("in");
        ym_p.add("ing");
        ym_p.add("o");
        ym_p.add("ou");
        ym_p.add("u");

        ym_q.add("i");// q
        ym_q.add("ia");
        ym_q.add("ian");
        ym_q.add("iang");
        ym_q.add("iao");
        ym_q.add("ie");
        ym_q.add("in");
        ym_q.add("ing");
        ym_q.add("iong");
        ym_q.add("iu");
        ym_q.add("u");
        ym_q.add("uan");
        ym_q.add("ue");
        ym_q.add("un");

        ym_r.add("an");// r
        ym_r.add("ang");
        ym_r.add("ao");
        ym_r.add("e");
        ym_r.add("en");
        ym_r.add("eng");
        ym_r.add("i");
        ym_r.add("ong");
        ym_r.add("ou");
        ym_r.add("u");
        ym_r.add("ua");
        ym_r.add("uan");
        ym_r.add("ui");
        ym_r.add("un");
        ym_r.add("uo");

        ym_s.add("a");// s
        ym_s.add("ai");
        ym_s.add("an");
        ym_s.add("ang");
        ym_s.add("ao");
        ym_s.add("e");
        ym_s.add("en");
        ym_s.add("eng");
        ym_s.add("i");
        ym_s.add("ong");
        ym_s.add("ou");
        ym_s.add("u");
        ym_s.add("uan");
        ym_s.add("ui");
        ym_s.add("un");
        ym_s.add("uo");

        ym_t.add("a");// t
        ym_t.add("ai");
        ym_t.add("an");
        ym_t.add("ang");
        ym_t.add("ao");
        ym_t.add("e");
        ym_t.add("ei");
        ym_t.add("eng");
        ym_t.add("i");
        ym_t.add("ian");
        ym_t.add("iao");
        ym_t.add("ie");
        ym_t.add("ing");
        ym_t.add("ong");
        ym_t.add("ou");
        ym_t.add("u");
        ym_t.add("uan");
        ym_t.add("ui");
        ym_t.add("un");
        ym_t.add("uo");

        ym_w.add("a");// w
        ym_w.add("ai");
        ym_w.add("an");
        ym_w.add("ang");
        ym_w.add("ei");
        ym_w.add("en");
        ym_w.add("eng");
        ym_w.add("o");
        ym_w.add("u");

        ym_x.add("i");// x
        ym_x.add("ia");
        ym_x.add("ian");
        ym_x.add("iang");
        ym_x.add("iao");
        ym_x.add("ie");
        ym_x.add("in");
        ym_x.add("ing");
        ym_x.add("iong");
        ym_x.add("iu");
        ym_x.add("u");
        ym_x.add("uan");
        ym_x.add("ue");
        ym_x.add("un");

        ym_y.add("a");// y
        ym_y.add("an");
        ym_y.add("ang");
        ym_y.add("ao");
        ym_y.add("e");
        ym_y.add("i");
        ym_y.add("in");
        ym_y.add("ing");
        ym_y.add("o");
        ym_y.add("ong");
        ym_y.add("ou");
        ym_y.add("u");
        ym_y.add("uan");
        ym_y.add("ue");
        ym_y.add("un");

        ym_z.add("a");//
        ym_z.add("ai");
        ym_z.add("an");
        ym_z.add("ang");
        ym_z.add("ao");
        ym_z.add("e");
        ym_z.add("ei");
        ym_z.add("en");
        ym_z.add("eng");
        ym_z.add("i");
        ym_z.add("ong");
        ym_z.add("ou");
        ym_z.add("u");
        ym_z.add("uan");
        ym_z.add("ui");
        ym_z.add("un");
        ym_z.add("uo");

        ym_ch.add("a");// ch
        ym_ch.add("ai");
        ym_ch.add("an");
        ym_ch.add("ang");
        ym_ch.add("ao");
        ym_ch.add("e");
        ym_ch.add("en");
        ym_ch.add("eng");
        ym_ch.add("i");
        ym_ch.add("ong");
        ym_ch.add("ou");
        ym_ch.add("u");
        ym_ch.add("ua");
        ym_ch.add("uai");
        ym_ch.add("uan");
        ym_ch.add("uang");
        ym_ch.add("ui");
        ym_ch.add("un");
        ym_ch.add("uo");

        ym_sh.add("a");// sh
        ym_sh.add("ai");
        ym_sh.add("an");
        ym_sh.add("ang");
        ym_sh.add("ao");
        ym_sh.add("e");
        ym_sh.add("ei");
        ym_sh.add("en");
        ym_sh.add("eng");
        ym_sh.add("i");
        ym_sh.add("ou");
        ym_sh.add("u");
        ym_sh.add("ua");
        ym_sh.add("uai");
        ym_sh.add("uan");
        ym_sh.add("uang");
        ym_sh.add("ui");
        ym_sh.add("un");
        ym_sh.add("uo");

        ym_zh.add("a");// zh
        ym_zh.add("ai");
        ym_zh.add("an");
        ym_zh.add("ang");
        ym_zh.add("ao");
        ym_zh.add("e");
        ym_zh.add("ei");
        ym_zh.add("en");
        ym_zh.add("eng");
        ym_zh.add("i");
        ym_zh.add("ong");
        ym_zh.add("ou");
        ym_zh.add("u");
        ym_zh.add("ua");
        ym_zh.add("uai");
        ym_zh.add("uan");
        ym_zh.add("uang");
        ym_zh.add("ui");
        ym_zh.add("un");
        ym_zh.add("uo");

        ym.add(yy);
        ym.add(ym_b);
        ym.add(ym_c);
        ym.add(ym_d);
        ym.add(ym_f);
        ym.add(ym_g);
        ym.add(ym_h);
        ym.add(ym_j);
        ym.add(ym_k);
        ym.add(ym_l);
        ym.add(ym_m);
        ym.add(ym_n);
        ym.add(ym_p);
        ym.add(ym_q);
        ym.add(ym_r);
        ym.add(ym_s);
        ym.add(ym_t);
        ym.add(ym_w);
        ym.add(ym_x);
        ym.add(ym_y);
        ym.add(ym_z);
        ym.add(ym_ch);
        ym.add(ym_sh);
        ym.add(ym_zh);

    }

    private static String findsm(String py, StringBuilder result) {
        char[] py2 = py.toCharArray();// 将读入字符转为char数组
        int temp = 0;
        int index = 0;//
        // 遍历声母集合，匹对
        for (int i = 0; i < sm.size(); i++) {
            for (int j = 1; j <= py2.length; j++) {
                String py3 = String.copyValueOf(py2, 0, j);// 截取从0开始到j结束的字符串
                if (py3.equals(sm.get(i))) {
                    temp = sm.get(i).length();// 对应的声母的长度
                    index = i + 1;
                    break;
                }
            }
        }
        if (temp != 0) {
            result.append(String.copyValueOf(py2, 0, temp));// 将匹对出来的声母赋给result
            py = py.substring(temp);
        }
        // 匹对完声母后如果字符串还么结束继续匹对韵母
        if (py.length() != 0) {
            return findym(py, index, result);
        } else {
            return py;
        }
    }

    private static String findym(String py, int index, StringBuilder result) {
        int temp = 0;
        char[] py2 = py.toCharArray();
        for (int i = 0; i < ym.get(index).size(); i++) {
            for (int j = 1; j <= py2.length; j++) {
                String py3 = String.copyValueOf(py2, 0, j);
                if (py3.equals(ym.get(index).get(i))) {
                    temp = ym.get(index).get(i).length();
                    break;
                }
            }
        }
        if (temp != 0) {
            result.append(String.copyValueOf(py2, 0, temp)).append(" ");
            py = py.substring(temp);
        }
        return py;
    }

    /**
     * 拼音全拼分词工具
     *
     * @param spell 拼音
     * @return 以空格区分的分词后的字符串
     */
    protected static String trimSpell(String spell) {
        StringBuilder result = new StringBuilder();
        String s = spell;
        for (int i = 0; i < spell.length() * 2; i++) {
            if (s.length() == 0) {
                break;
            }
            s = findym(s, 0, result);
            s = findsm(s, result);
        }
        return result.toString();
    }
}
