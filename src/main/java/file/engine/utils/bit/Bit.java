package file.engine.utils.bit;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 大数位运算模块
 */
public class Bit {

    private final AtomicReference<byte[]> bytes = new AtomicReference<>();
    private final byte[] zero = new byte[]{0};

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
            this.bytes.compareAndSet(this.bytes.get(), bytes);
        } else {
            throw new RuntimeException("the bytes could not be empty");
        }
    }

    /**
     * 左移count位
     *
     * @param count 次数
     * @return 当前bit对象
     */
    public Bit shiftLeft(int count) {
        byte[] newBytes = Arrays.copyOf(bytes.get(), this.bytes.get().length + count);
        bytes.compareAndSet(bytes.get(), newBytes);
        return this;
    }

    /**
     * 右移count位
     *
     * @param count 次数
     * @return 当前bit对象
     */
    public Bit shiftRight(int count) {
        if (bytes.get().length <= count) {
            bytes.compareAndSet(bytes.get(), zero);
        } else {
            byte[] newBytes = Arrays.copyOfRange(bytes.get(), 0, bytes.get().length - count);
            bytes.compareAndSet(bytes.get(), newBytes);
        }
        return this;
    }

    /**
     * 与运算
     *
     * @param bit bit
     * @return 结果
     */
    public Bit and(Bit bit) {
        boolean isThisBigger = this.bytes.get().length > bit.bytes.get().length;
        AtomicReference<byte[]> bigger = isThisBigger ? this.bytes : bit.bytes;
        AtomicReference<byte[]> smaller = isThisBigger ? bit.bytes : this.bytes;
        int offset = Math.abs(this.bytes.get().length - bit.bytes.get().length);
        int minLength = smaller.get().length;
        byte[] res = new byte[minLength];
        for (int i = minLength - 1; i >= 0; i--) {
            byte b1, b2;
            b1 = smaller.get()[i];
            int index = i + offset;
            if (index < bigger.get().length) {
                b2 = bigger.get()[index];
            } else {
                b2 = 0;
            }
            res[i] = (byte) (b1 & b2);
        }
        return new Bit(res);
    }

    /**
     * 或运算
     *
     * @param bit bit
     * @return 结果
     */
    public Bit or(Bit bit) {
        boolean isThisBigger = this.bytes.get().length > bit.bytes.get().length;
        AtomicReference<byte[]> bigger = isThisBigger ? this.bytes : bit.bytes;
        AtomicReference<byte[]> smaller = isThisBigger ? bit.bytes : this.bytes;
        int offset = Math.abs(this.bytes.get().length - bit.bytes.get().length);
        int maxLength = bigger.get().length;
        byte[] res = new byte[maxLength];
        for (int i = maxLength - 1; i >= 0; i--) {
            byte b1, b2;
            int index;
            b1 = bigger.get()[i];
            if ((index = i - offset) >= 0 && index < bigger.get().length) {
                b2 = smaller.get()[index];
            } else {
                b2 = 0;
            }
            res[i] = (byte) (b1 | b2);
        }
        return new Bit(res);
    }

    /**
     * 非运算
     *
     * @return 结果
     */
    public Bit not() {
        int index = 0;
        for (int i = 0; i < this.bytes.get().length; i++) {
            if (this.bytes.get()[i] == 0) {
                index = i;
                break;
            }
        }
        byte[] res = Arrays.copyOfRange(this.bytes.get(), index, this.bytes.get().length);
        for (int i = 0; i < res.length; i++) {
            res[i] = (byte) (res[i] == 0 ? 1 : 0);
        }
        return new Bit(res);
    }

    /**
     * 更新值
     *
     * @param bit bit
     * @return this
     */
    public Bit set(Bit bit) {
        byte[] bytes = Arrays.copyOf(bit.bytes.get(), bit.bytes.get().length);
        this.bytes.compareAndSet(this.bytes.get(), bytes);
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

    public int length() {
        return this.bytes.get().length;
    }
}
