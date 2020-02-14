package pinyin;

import net.sourceforge.pinyin4j.PinyinHelper;
import net.sourceforge.pinyin4j.format.HanyuPinyinCaseType;
import net.sourceforge.pinyin4j.format.HanyuPinyinOutputFormat;
import net.sourceforge.pinyin4j.format.HanyuPinyinToneType;
import net.sourceforge.pinyin4j.format.HanyuPinyinVCharType;
import net.sourceforge.pinyin4j.format.exception.BadHanyuPinyinOutputFormatCombination;

public class PinYinConverter {
    public static String getPinYin(String src) {
        char[] t1;
        t1 = src.toCharArray();
        // System.out.println(t1.length);
        String[] t2;
        // System.out.println(t2.length);
        // ���ú���ƴ������ĸ�ʽ
        HanyuPinyinOutputFormat t3 = new HanyuPinyinOutputFormat();
        t3.setCaseType(HanyuPinyinCaseType.LOWERCASE);
        t3.setToneType(HanyuPinyinToneType.WITHOUT_TONE);
        t3.setVCharType(HanyuPinyinVCharType.WITH_V);
        StringBuilder t4 = new StringBuilder();
        int t0 = t1.length;
        try {
            for (char c : t1) {
                // �ж��ܷ�Ϊ�����ַ�
                // System.out.println(t1[i]);
                if (Character.toString(c).matches("[\\u4E00-\\u9FA5]+")) {
                    t2 = PinyinHelper.toHanyuPinyinStringArray(c, t3);// �����ֵļ���ȫƴ���浽t2������
                    t4.append(t2[0]);// ȡ���ú���ȫƴ�ĵ�һ�ֶ��������ӵ��ַ���t4��
                } else {
                    // ������Ǻ����ַ������ȡ���ַ������ӵ��ַ���t4��
                    t4.append(c);
                }
            }
        } catch (BadHanyuPinyinOutputFormatCombination e) {
            e.printStackTrace();
        }
        return t4.toString();
    }

    /**
     * ��ȡÿ�����ֵ�����ĸ
     *
     * @param str
     * @return String
     */
    public static String getPinYinHeadChar(String str) {
        StringBuilder convert = new StringBuilder();
        for (int j = 0; j < str.length(); j++) {
            char word = str.charAt(j);
            // ��ȡ���ֵ�����ĸ
            String[] pinyinArray = PinyinHelper.toHanyuPinyinStringArray(word);
            if (pinyinArray != null) {
                convert.append(pinyinArray[0].charAt(0));
            } else {
                convert.append(word);
            }
        }
        return convert.toString();
    }

    /**
     * ���ַ���ת����ASCII��
     *
     * @param cnStr
     * @return String
     */
    public static String getCnASCII(String cnStr) {
        StringBuilder strBuf = new StringBuilder();
        // ���ַ���ת�����ֽ�����
        byte[] bGBK = cnStr.getBytes();
        for (byte b : bGBK) {
            // System.out.println(Integer.toHexString(bGBK[i] & 0xff));
            // ��ÿ���ַ�ת����ASCII��
            strBuf.append(Integer.toHexString(b & 0xff));
        }
        return strBuf.toString();
    }
}
