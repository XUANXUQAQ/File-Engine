package file.engine.event.handler;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class Event {
    private final AtomicBoolean isFinished = new AtomicBoolean(false);
    private final AtomicInteger executeTimes = new AtomicInteger(0);
    private final AtomicBoolean isBlock = new AtomicBoolean(false);
    private int maxRetryTimes = 5;
    private volatile Object returnValue;
    private Consumer<Event> callback;
    private Consumer<Event> errorHandler;

    protected void incrementExecuteTimes() {
        executeTimes.incrementAndGet();
    }

    protected boolean allRetryFailed() {
        return executeTimes.get() > maxRetryTimes;
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

    protected void execErrorHandler() {
        if (this.errorHandler != null) {
            this.errorHandler.accept(this);
        }
    }

    protected void setFinishedAndExecCallback() {
        if (this.callback != null) {
            this.callback.accept(this);
        }
        isFinished.set(true);
    }

    public void setReturnValue(Object obj) {
        this.returnValue = obj;
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

    public void setMaxRetryTimes(int maxRetryTimes) {
        this.maxRetryTimes = maxRetryTimes;
    }

    @Override
    public String toString() {
        return "{event >>> " + this.getClass() + "}";
    }
}
