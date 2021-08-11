package file.engine.utils;

import file.engine.configs.Constants;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

public class StartupUtil {

    /**
     * 检查是否又开机启动
     * @return boolean
     */
    public static boolean hasStartup() {
        String command = "cmd.exe /c chcp 65001 & schtasks /query /tn \"File-Engine\"";
        Process p = null;
        try {
            p = Runtime.getRuntime().exec(command);
            StringBuilder strBuilder = new StringBuilder();
            String line;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                while ((line = reader.readLine()) != null) {
                    strBuilder.append(line);
                }
            }
            return strBuilder.toString().contains("File-Engine");
        } catch (IOException e) {
            return false;
        } finally {
            if (p != null) {
                p.destroy();
            }
        }
    }

    /**
     * 添加到开机启动
     * @return Process
     * @throws IOException exception
     * @throws InterruptedException exception
     */
    public static Process addStartup() throws IOException, InterruptedException {
        String command = "cmd.exe /c schtasks /create /ru \"administrators\" /rl HIGHEST /sc ONLOGON /tn \"File-Engine\" /tr ";
        File FileEngine = new File(Constants.FILE_NAME);
        String absolutePath = "\"\"" + FileEngine.getAbsolutePath() + "\"\" /f";
        command += absolutePath;
        Process p;
        p = Runtime.getRuntime().exec(command);
        p.waitFor();
        return p;
    }

    /**
     * 删除开机启动
     * @return Process
     * @throws InterruptedException exception
     * @throws IOException exception
     */
    public static Process deleteStartup() throws InterruptedException, IOException {
        String command = "cmd.exe /c schtasks /delete /tn \"File-Engine\" /f";
        Process p;
        p = Runtime.getRuntime().exec(command);
        p.waitFor();
        return p;
    }
}
