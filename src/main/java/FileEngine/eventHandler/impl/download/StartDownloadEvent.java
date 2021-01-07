package FileEngine.eventHandler.impl.download;

import java.io.File;
import java.io.IOException;

public class StartDownloadEvent extends DownloadEvent {

    public StartDownloadEvent(String url, String fileName, String savePath) {
        super(url, fileName, savePath);
        try {
            if (!new File(savePath, fileName).createNewFile()) {
                throw new IOException();
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("创建下载任务文件" + fileName + File.pathSeparatorChar + savePath + "失败");
        }
    }
}
