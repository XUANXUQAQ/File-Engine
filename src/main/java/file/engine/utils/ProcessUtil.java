package file.engine.utils;

import file.engine.utils.system.properties.IsDebug;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.concurrent.TimeUnit;

public class ProcessUtil {

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
    public static void waitForProcess(@SuppressWarnings("SameParameterValue") String procName, long checkInterval) throws IOException, InterruptedException {
        long start = System.currentTimeMillis();
        long timeLimit = 10 * 60 * 1000;
        if (IsDebug.isDebug()) {
            timeLimit = Long.MAX_VALUE;
        }
        while (isProcessExist(procName)) {
            TimeUnit.MILLISECONDS.sleep(checkInterval);
            if (System.currentTimeMillis() - start > timeLimit) {
                System.err.printf("等待进程%s超时%n", procName);
                String command = String.format("taskkill /im %s /f", procName);
                Process exec = Runtime.getRuntime().exec(command);
                exec.waitFor();
                break;
            }
        }
    }
}
