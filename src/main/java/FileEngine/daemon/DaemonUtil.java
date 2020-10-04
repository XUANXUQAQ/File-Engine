package FileEngine.daemon;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

public class DaemonUtil {

    /**
     * 开启守护进程
     * @param currentWorkingDir 当前进程工作环境位置
     */
    public static void startDaemon(String currentWorkingDir) {
        try {
            if (!isDaemonExist()) {
                File daemonProcess = new File("user/daemonProcess.exe");
                if (daemonProcess.canExecute()) {
                    String command = daemonProcess.getAbsolutePath();
                    String start = "cmd.exe /c " + command.substring(0, 2);
                    String end = "\"" + command.substring(2) + "\"";
                    String finalCommand = start + end + " \"" + currentWorkingDir + "\"";
                    Runtime.getRuntime().exec(finalCommand, null, daemonProcess.getParentFile());
                }
            }
        } catch (InterruptedException | IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 关闭守护进程
     */
    public static void stopDaemon() {
        File closeSignal = new File("tmp/closeDaemon");
        try {
            closeSignal.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 检测进程是否存在
     * @return true如果进程以存在
     */
    private static boolean isDaemonExist() throws IOException, InterruptedException {
        StringBuilder strBuilder = new StringBuilder();
        Process p = Runtime.getRuntime().exec("tasklist /FI \"IMAGENAME eq " + "daemonProcess.exe" + "\"");
        p.waitFor();
        String eachLine;
        try (BufferedReader buffr = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            while ((eachLine = buffr.readLine()) != null) {
                strBuilder.append(eachLine);
            }
        }
        return strBuilder.toString().contains("daemonProcess.exe");
    }
}
