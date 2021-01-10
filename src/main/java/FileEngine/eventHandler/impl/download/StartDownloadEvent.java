package FileEngine.eventHandler.impl.download;

import FileEngine.utils.download.DownloadManager;

import java.io.File;
import java.io.IOException;

public class StartDownloadEvent extends DownloadEvent {

    public StartDownloadEvent(DownloadManager downloadManager) {
        super(downloadManager);
        try {
            boolean isFileExist;
            File dir = new File(downloadManager.savePath);
            File f = new File(dir, downloadManager.fileName);
            if (f.exists()) {
                isFileExist = true;
            } else {
                if (dir.exists()) {
                    isFileExist = f.createNewFile();
                } else {
                    if ((isFileExist = dir.mkdirs())) {
                        isFileExist = f.createNewFile();
                    }
                }
            }
            if (!isFileExist) {
                throw new IOException("创建下载任务文件" +
                        downloadManager.fileName + File.pathSeparatorChar + downloadManager.savePath + "失败");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
