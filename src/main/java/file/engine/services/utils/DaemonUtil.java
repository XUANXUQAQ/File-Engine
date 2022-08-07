package file.engine.services.utils;

import java.io.File;
import java.io.IOException;

public class DaemonUtil {
    /**
     * 关闭守护进程
     */
    public static void stopDaemon() {
        File closeSignal = new File("tmp/closeDaemon");
        if (!closeSignal.exists()) {
            boolean isCreated = false;
            try {
                isCreated = closeSignal.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (!isCreated) {
                System.err.println("创建守护进程关闭标志文件失败");
            }
        }
    }
}
