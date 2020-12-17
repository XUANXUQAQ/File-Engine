package FileEngine.taskHandler;

public abstract class TaskHandler {
    public abstract void todo(Task task);

    protected void doTask(Task task) {
        todo(task);
        task.setFinished();
    }
}
