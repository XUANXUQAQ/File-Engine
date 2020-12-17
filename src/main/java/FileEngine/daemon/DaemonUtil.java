package FileEngine.daemon;

import FileEngine.taskHandler.TaskUtil;
import FileEngine.taskHandler.Task;
import FileEngine.taskHandler.TaskHandler;
import FileEngine.taskHandler.impl.daemon.StartDaemonTask;
import FileEngine.taskHandler.impl.daemon.StopDaemonTask;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

public class DaemonUtil {
    private static volatile DaemonUtil instance = null;

    private DaemonUtil() {}

    public static DaemonUtil getInstance() {
        if (instance == null) {
            synchronized (DaemonUtil.class) {
                if (instance == null) {
                    instance = new DaemonUtil();
                }
            }
        }
        return instance;
    }

    public static void registerTaskHandler() {
        TaskUtil.getInstance().registerTaskHandler(StartDaemonTask.class, new TaskHandler() {
            @Override
            public void todo(Task task) {
                getInstance().startDaemon(((StartDaemonTask) task).currentWorkingDir);
            }
        });

        TaskUtil.getInstance().registerTaskHandler(StopDaemonTask.class, new TaskHandler() {
            @Override
            public void todo(Task task) {
                getInstance().stopDaemon();
            }
        });
    }


    /**
     * 开启守护进程
     * @param currentWorkingDir 当前进程工作环境位置
     */
    private void startDaemon(String currentWorkingDir) {
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
    private void stopDaemon() {
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

    /**
     * 检测进程是否存在
     * @return true如果进程以存在
     */
    private boolean isDaemonExist() throws IOException, InterruptedException {
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
