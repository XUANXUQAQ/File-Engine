package FileEngine.eventHandler;

public abstract class EventHandler {
    public abstract void todo(Event event);

    protected void doTask(Event event) {
        todo(event);
        event.setFinished();
    }
}
