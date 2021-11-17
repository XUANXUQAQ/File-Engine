package file.engine.utils;

import file.engine.utils.system.properties.IsDebug;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.concurrent.TimeUnit;

public class ProcessUtil {

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

    /**
     * 进程是否存在
     *
     * @param procName 进程名
     * @return boolean
     * @throws IOException          失败
     * @throws InterruptedException 失败
     */
    @SuppressWarnings("IndexOfReplaceableByContains")
    public static boolean isProcessExist(String procName) throws IOException, InterruptedException {
        StringBuilder strBuilder = new StringBuilder();
        if (!procName.isEmpty()) {
            Process p = Runtime.getRuntime().exec("tasklist /FI \"IMAGENAME eq " + procName + "\"");
            p.waitFor();
            String eachLine;
            try (BufferedReader buffr = new BufferedReader(new InputStreamReader(p.getInputStream(), Charset.defaultCharset()))) {
                while ((eachLine = buffr.readLine()) != null) {
                    strBuilder.append(eachLine);
                }
            }
            return strBuilder.toString().indexOf(procName) != -1;
        }
        return false;
    }

    /**
     * 等待进程
     *
     * @param procName 进程名
     * @throws IOException          失败
     * @throws InterruptedException 失败
     */
    public static void waitForProcess(@SuppressWarnings("SameParameterValue") String procName) throws IOException, InterruptedException {
        long start = System.currentTimeMillis();
        long timeLimit = 10 * 60 * 1000;
        if (IsDebug.isDebug()) {
            timeLimit = Long.MAX_VALUE;
        }
        while (isProcessExist(procName)) {
            TimeUnit.MILLISECONDS.sleep(10);
            if (System.currentTimeMillis() - start > timeLimit) {
                System.err.printf("等待进程%s超时\n", procName);
                String command = String.format("taskkill /im %s /f", procName);
                Process exec = Runtime.getRuntime().exec(command);
                exec.waitFor();
                break;
            }
        }
    }

    /**
     * 添加等待进程并当进程完成后执行回调
     *
     * @param procName 进程名
     * @param callback 回调
     */
    public static void waitForProcessAsync(@SuppressWarnings("SameParameterValue") String procName, Runnable callback) {
        CachedThreadPoolUtil.getInstance().executeTask(() -> {
            try {
                waitForProcess(procName);
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            } finally {
                callback.run();
            }
        });
    }
}
