package FileEngine.taskHandler.impl.download;

import FileEngine.taskHandler.Task;

public class DownloadTask extends Task {
    public final String url;
    public final String fileName;
    public final String savePath;

    protected DownloadTask(String url, String fileName, String savePath) {
        super();
        this.url = url;
        this.fileName = fileName;
        this.savePath = savePath;
    }
}
