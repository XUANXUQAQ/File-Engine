package file.engine.utils.bit;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 大数位运算模块
 */
public class Bit {

    private final AtomicReference<byte[]> bytes = new AtomicReference<>();

    public Bit(byte[] init) {
        if (init != null && init.length > 0) {
            this.bytes.set(init);
        } else {
            throw new RuntimeException("the bytes could not be empty");
        }
    }

    public Bit(Bit bit) {
        if (bit != null && bit.bytes.get().length > 0) {
            byte[] bytes = Arrays.copyOf(bit.bytes.get(), bit.bytes.get().length);
            this.bytes.set(bytes);
        } else {
            throw new RuntimeException("the bytes could not be empty");
        }
    }

    /**
     * 左移1位
     * @return 当前bit对象
     */
    public Bit shiftLeft() {
        return shiftLeft(1);
    }

    /**
     * 右移一位
     * @return 当前bit对象
     */
    public Bit shiftRight() {
        return shiftRight(1);
    }

    /**
     * 左移count位
     * @param count 次数
     * @return 当前bit对象
     */
    public Bit shiftLeft(int count) {
        byte[] newBytes = Arrays.copyOf(bytes.get(), this.bytes.get().length + count);
        bytes.set(newBytes);
        return this;
    }

    /**
     * 右移count位
     * @param count 次数
     * @return 当前bit对象
     */
    public Bit shiftRight(int count) {
        byte[] newBytes = Arrays.copyOfRange(bytes.get(), 0, bytes.get().length - count);
        bytes.set(newBytes);
        return this;
    }

    /**
     * 与运算
     * @param bit bit
     * @return 结果
     */
    public Bit and(Bit bit) {
        boolean isThisBigger = this.bytes.get().length > bit.bytes.get().length;
        byte[] bigger = isThisBigger ? this.bytes.get() : bit.bytes.get();
        byte[] smaller = isThisBigger ? bit.bytes.get() : this.bytes.get();
        int offset = Math.abs(this.bytes.get().length - bit.bytes.get().length);
        int minLength = smaller.length;
        byte[] res = new byte[minLength];
        for (int i = minLength - 1; i >= 0; i--) {
            byte b1, b2;
            int index;
            b1 = smaller[i];
            if ((index = i + offset) >= 0) {
                b2 = bigger[index];
            } else {
                b2 = 0;
            }
            res[i] = (byte) (b1 & b2);
        }
        return new Bit(res);
    }

    /**
     * 或运算
     * @param bit bit
     * @return 结果
     */
    public Bit or(Bit bit) {
        boolean isThisBigger = this.bytes.get().length > bit.bytes.get().length;
        byte[] bigger = isThisBigger ? this.bytes.get() : bit.bytes.get();
        byte[] smaller = isThisBigger ? bit.bytes.get() : this.bytes.get();
        int offset = Math.abs(this.bytes.get().length - bit.bytes.get().length);
        int maxLength = bigger.length;
        byte[] res = new byte[maxLength];
        for (int i = maxLength - 1; i >= 0 ; i--) {
            byte b1, b2;
            int index;
            b1 = bigger[i];
            if ((index = i - offset) >= 0) {
                b2 = smaller[index];
            } else {
                b2 = 0;
            }
            res[i] = (byte) (b1 | b2);
        }
        return new Bit(res);
    }

    /**
     * 非运算
     * @return 结果
     */
    public Bit not() {
        byte[] res = new byte[this.bytes.get().length];
        for (int i = 0; i < this.bytes.get().length; i++) {
            res[i] = (byte) (this.bytes.get()[i] == 0 ? 1 : 0);
        }
        return new Bit(res);
    }

    /**
     * 更新值
     * @param bit bit
     * @return this
     */
    public Bit set(Bit bit) {
        byte[] bytes = Arrays.copyOf(bit.bytes.get(), bit.bytes.get().length);
        this.bytes.set(bytes);
        return this;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < this.bytes.get().length; i++) {
            builder.append(this.bytes.get()[i]);
        }
        return builder.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Bit) {
            Bit tmp = (Bit) obj;
            return Arrays.equals(this.bytes.get(), tmp.bytes.get());
        }
        return false;
    }

    public static void main(String[] args) {
        Bit bit = new Bit(new byte[] {1, 0, 1, 0});
        Bit or = bit.or(new Bit(new byte[]{0, 1, 1}));
        assert or.equals(new Bit(new byte[] {1, 0, 1, 1}));

        Bit and = bit.and(new Bit(new byte[] {1, 0, 0}));
        assert and.equals(new Bit(new byte[]{0}));

        Bit not = bit.not();
        assert not.equals(new Bit(new byte[] {0, 1, 0, 1}));
    }
}
