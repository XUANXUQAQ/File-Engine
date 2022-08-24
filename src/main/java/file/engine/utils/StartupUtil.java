package file.engine.utils;

import file.engine.configs.Constants;
import file.engine.utils.file.FileUtil;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class StartupUtil {

    /**
     * 检查是否开机启动
     *
     * @return 全为零代表包含开机启动且启动项有效，如果第1位为1则为包含启动项但启动项无效，如果第2位为1则为不存在启动项或检查失败，
     */
    public static int hasStartup() {
        String command = "cmd.exe /c chcp 65001 & schtasks /query /V /tn \"File-Engine\"";
        Process p = null;
        try {
            p = Runtime.getRuntime().exec(command);
            p.waitFor();
            String keys = "";
            String results = "";
            String separator = "";
            String line;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                // 前三行无用
                reader.readLine();
                reader.readLine();
                reader.readLine();
                if ((line = reader.readLine()) != null) {
                    keys = line;
                }
                if ((line = reader.readLine()) != null) {
                    separator = line;
                }
                if ((line = reader.readLine()) != null) {
                    results = line;
                }
            }
            //noinspection IndexOfReplaceableByContains
            if (results.indexOf("File-Engine") == -1) {
                return 2;
            }
            Map<String, String> infoMap = parseResults(separator, keys, results);
            String taskToRun = infoMap.get("Task To Run");
            Pattern pattern = RegexUtil.getPattern("\"", 0);
            taskToRun = pattern.matcher(taskToRun).replaceAll("");
            if (Files.exists(Path.of(taskToRun))) {
                return 0;
            }
            deleteStartup();
            return 1;
        } catch (Exception e) {
            e.printStackTrace();
            return 2;
        } finally {
            if (p != null) {
                p.destroy();
            }
        }
    }

    /*
      添加到开机启动

      @return Process
     * @throws IOException          exception
     * @throws InterruptedException exception
     */
//    public static Process addStartup() throws IOException, InterruptedException {
//        String command = "cmd.exe /c schtasks /create /ru \"administrators\" /rl HIGHEST /sc ONLOGON /tn \"File-Engine\" /tr ";
//        String parentPath = FileUtil.getParentPath(new File("").getAbsolutePath());
//        File fileEngine = new File(parentPath + File.separator + Constants.LAUNCH_WRAPPER_NAME);
//        String absolutePath = fileEngine.getAbsolutePath();
//        command += "\"'" + absolutePath + "'\"";
//        Process p;
//        p = Runtime.getRuntime().exec(command);
//        p.waitFor();
//        return p;
//    }

    /**
     * 添加到开机启动
     *
     * @return Process
     * @throws IOException          文件创建失败
     * @throws InterruptedException exception
     */
    public static Process addStartupByXml() throws IOException, InterruptedException {
        String parentPath = FileUtil.getParentPath(new File("").getAbsolutePath());
        File fileEngine = new File(parentPath + File.separator + Constants.LAUNCH_WRAPPER_NAME);
        String absolutePath = fileEngine.getAbsolutePath();
        StringBuilder builder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(Objects.requireNonNull(StartupUtil.class.getResourceAsStream("/schtasks.xml")),
                StandardCharsets.UTF_16LE))) {
            String line;
            while ((line = reader.readLine()) != null) {
                builder.append(line);
            }
        }
        String xmlContent = builder.toString();
        xmlContent = String.format(xmlContent, absolutePath);
        String tmpDir = new File("").getAbsolutePath().contains(" ") ? System.getProperty("java.io.tmpdir") : new File("tmp").getAbsolutePath();
        Path xmlPath = Path.of(tmpDir, "schtasks-File-Engine.xml");
        Files.writeString(xmlPath, xmlContent, StandardCharsets.UTF_16LE);
        String command = String.format("cmd.exe /c schtasks /create /xml %s /tn \"File-Engine\"", xmlPath.getFileName());
        Process exec = Runtime.getRuntime().exec(command, null, new File(tmpDir));
        exec.waitFor();
        return exec;
    }

    /**
     * 删除开机启动
     *
     * @return Process
     * @throws InterruptedException exception
     * @throws IOException          exception
     */
    public static Process deleteStartup() throws InterruptedException, IOException {
        String command = "cmd.exe /c schtasks /delete /tn \"File-Engine\" /f";
        Process p;
        p = Runtime.getRuntime().exec(command);
        p.waitFor();
        return p;
    }

    /**
     * 解析schtasks返回的结果字符串为Map
     *
     * @param separator 分隔符
     * @param keys      所有的key，用分隔符连接在一起
     * @param results   所有的result，用分隔符连接在一起
     * @return Map
     */
    private static Map<String, String> parseResults(String separator, String keys, String results) {
        String[] separatorArray = RegexUtil.blank.split(separator);
        int size = separatorArray.length;
        int keyIndex = 0;
        int valIndex = 0;
        String[] keysArray = new String[size];
        String[] resultsArray = new String[size];
        for (int i = 0; i < size; i++) {
            int strLength = separatorArray[i].length();
            keysArray[i] = keys.substring(keyIndex, keyIndex + strLength);
            keyIndex += strLength + 1;
            resultsArray[i] = results.substring(valIndex, valIndex + strLength);
            valIndex += strLength + 1;
        }
        List<String> keyList = Arrays.stream(keysArray).filter(each -> !each.isBlank()).map(String::trim).collect(Collectors.toList());
        List<String> resultList = Arrays.stream(resultsArray).filter(each -> !each.isBlank()).map(String::trim).collect(Collectors.toList());
        int keySize = keyList.size();
        if (keySize == resultList.size()) {
            return IntStream.range(0, keySize).boxed().collect(Collectors.toMap(keyList::get, resultList::get));
        }
        throw new RuntimeException("parse schtasks information failed.");
    }
}
