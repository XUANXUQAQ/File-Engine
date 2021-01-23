package FileEngine.eventHandler;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class Event {
    private final AtomicBoolean isFinished = new AtomicBoolean(false);
    private final AtomicBoolean isFailed = new AtomicBoolean(false);
    private final AtomicInteger executeTimes = new AtomicInteger(0);
    private final AtomicBoolean isBlock = new AtomicBoolean(false);
    private Object returnValue;

    public void incrementExecuteTimes() {
        executeTimes.incrementAndGet();
    }

    public int getExecuteTimes() {
        return executeTimes.get();
    }

    public boolean isFinished() {
        return isFinished.get();
    }

    public void setBlock() {
        isBlock.set(true);
    }

    public boolean isBlock() {
        return isBlock.get();
    }

    public boolean isFailed() {
        return isFailed.get();
    }

    public void setFailed() {
        isFailed.set(true);
    }

    public void setFinished() {
        isFinished.set(true);
    }

    public void setReturnValue(Object obj) {
        returnValue = obj;
    }

    protected Object getReturnValue() {
        return returnValue;
    }

    @Override
    public String toString() {
        return "{task >>> " + this.getClass() + "}";
    }
}
