package file.engine.services.utils;

import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.io.IOException;

@Slf4j
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
                log.error("error: {}", e.getMessage(), e);
            }
            if (!isCreated) {
                log.error("创建守护进程关闭标志文件失败");
            }
        }
    }
}
