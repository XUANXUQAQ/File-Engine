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
            byte[] newVal = Arrays.copyOf(bit.bytes.get(), bit.bytes.get().length);
            this.bytes.set(newVal);
        } else {
            throw new RuntimeException("the bytes could not be empty");
        }
    }

    public byte[] getBytes() {
        return this.bytes.get();
    }

    /**
     * 左移count位
     *
     * @param count 次数
     * @return 当前bit对象
     */
    public Bit shiftLeft(int count) {
        byte[] originBytes;
        while ((originBytes = bytes.get()) != null) {
            byte[] newBytes = Arrays.copyOf(originBytes, originBytes.length + count);
            if (bytes.compareAndSet(originBytes, newBytes)) {
                return this;
            }
        }
        throw new RuntimeException("bit value is null");
    }

    /**
     * 右移count位
     *
     * @param count 次数
     * @return 当前bit对象
     */
    public Bit shiftRight(int count) {
        if (bytes.get().length <= count) {
            bytes.set(zero);
            return this;
        } else {
            byte[] originBytes;
            while ((originBytes = bytes.get()) != null) {
                byte[] newBytes = Arrays.copyOfRange(originBytes, 0, originBytes.length - count);
                if (bytes.compareAndSet(originBytes, newBytes)) {
                    return this;
                }
            }
        }
        throw new RuntimeException("bit value is null");
    }

    /**
     * 与运算
     *
     * @param bytes1 bytes1
     * @param bytes2 bytes2
     * @return 结果 Bit
     */
    public static Bit and(byte[] bytes1, byte[] bytes2) {
        boolean isBytes1Bigger = bytes1.length > bytes2.length;
        byte[] bigger = isBytes1Bigger ? bytes1 : bytes2;
        byte[] smaller = isBytes1Bigger ? bytes2 : bytes1;
        int offset = Math.abs(bytes1.length - bytes2.length);
        int minLength = smaller.length;
        byte[] res = new byte[minLength];
        for (int i = minLength - 1; i >= 0; i--) {
            byte b1, b2;
            b1 = smaller[i];
            int index = i + offset;
            if (index < bigger.length) {
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
     *
     * @param bytes1 bytes1
     * @param bytes2 bytes2
     * @return 结果 Bit
     */
    public static Bit or(byte[] bytes1, byte[] bytes2) {
        boolean isBytes1Bigger = bytes1.length > bytes2.length;
        byte[] bigger = isBytes1Bigger ? bytes1 : bytes2;
        byte[] smaller = isBytes1Bigger ? bytes2 : bytes1;
        int offset = Math.abs(bytes1.length - bytes2.length);
        int maxLength = bigger.length;
        byte[] res = new byte[maxLength];
        for (int i = maxLength - 1; i >= 0; i--) {
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
     * 更新值，使用cas算法
     * @param expect 之前值
     * @param bit 更新的值
     * @return 是否成功
     */
    public boolean set(byte[] expect, Bit bit) {
        byte[] bytes = Arrays.copyOf(bit.bytes.get(), bit.bytes.get().length);
        return this.bytes.compareAndSet(expect, bytes);
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
