package file.engine.event.handler;

import lombok.Setter;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class Event {
    private final AtomicBoolean isFinished = new AtomicBoolean(false);
    private final AtomicBoolean isFailed = new AtomicBoolean(false);
    private final AtomicInteger executeTimes = new AtomicInteger(0);
    private final AtomicBoolean isBlock = new AtomicBoolean(false);
    private int maxRetryTimes = 5;
    private @Setter
    Object returnValue;
    private Consumer<Event> callback;
    private Consumer<Event> errorHandler;

    protected void incrementExecuteTimes() {
        executeTimes.incrementAndGet();
    }

    protected int getExecuteTimes() {
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

    protected void setFailed() {
        if (this.errorHandler != null) {
            this.errorHandler.accept(this);
        }
        isFailed.set(true);
    }

    protected void setFinished() {
        if (this.callback != null) {
            this.callback.accept(this);
        }
        isFinished.set(true);
    }

    @SuppressWarnings("unchecked")
    public <T> Optional<T> getReturnValue() {
        return Optional.ofNullable((T) returnValue);
    }

    public void setCallback(Consumer<Event> callback) {
        this.callback = callback;
    }

    public void setErrorHandler(Consumer<Event> errorHandler) {
        this.errorHandler = errorHandler;
    }

    public int getMaxRetryTimes() {
        return maxRetryTimes;
    }

    public void setMaxRetryTimes(int maxRetryTimes) {
        this.maxRetryTimes = maxRetryTimes;
    }

    @Override
    public String toString() {
        return "{event >>> " + this.getClass() + "}";
    }
}
