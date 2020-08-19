package FileEngine.daemon;

import java.io.File;
import java.io.IOException;

public class DaemonUtil {
    public static void startDaemon(String currentWorkingDir) {
        File daemonProcess = new File("user/daemonProcess.exe");
        if (daemonProcess.canExecute()) {
            try {
                String command = daemonProcess.getAbsolutePath();
                String start = "cmd.exe /c " + command.substring(0, 2);
                String end = "\"" + command.substring(2) + "\"";
                String finalCommand = start + end + " \"" + currentWorkingDir + "\"";
                Runtime.getRuntime().exec(finalCommand, null, daemonProcess.getParentFile());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void stopDaemon() {
        File closeSignal = new File("tmp/closeDaemon");
        try {
            closeSignal.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
