package FileEngine.daemon;

import FileEngine.eventHandler.EventUtil;
import FileEngine.eventHandler.Event;
import FileEngine.eventHandler.EventHandler;
import FileEngine.eventHandler.impl.daemon.StartDaemonEvent;
import FileEngine.eventHandler.impl.daemon.StopDaemonEvent;

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

    public static void registerEventHandler() {
        EventUtil.getInstance().register(StartDaemonEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
                getInstance().startDaemon(((StartDaemonEvent) event).currentWorkingDir);
            }
        });

        EventUtil.getInstance().register(StopDaemonEvent.class, new EventHandler() {
            @Override
            public void todo(Event event) {
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
