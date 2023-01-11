package file.engine.services.utils;

import java.io.IOException;
import java.io.PrintStream;
import java.util.Scanner;

public class AdminUtil {

    /**
     * 检查是否拥有管理员权限
     *
     * @return boolean
     */
    public static boolean isAdmin() {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("cmd.exe");
            Process process = processBuilder.start();
            try (PrintStream printStream = new PrintStream(process.getOutputStream(), true)) {
                try (Scanner scanner = new Scanner(process.getInputStream())) {
                    printStream.println("@echo off");
                    printStream.println(">nul 2>&1 \"%SYSTEMROOT%\\system32\\cacls.exe\" \"%SYSTEMROOT%\\system32\\config\\system\"");
                    printStream.println("echo %errorlevel%");
                    boolean printedErrorLevel = false;
                    while (true) {
                        String nextLine = scanner.nextLine();
                        if (printedErrorLevel) {
                            int errorLevel = Integer.parseInt(nextLine);
                            scanner.close();
                            return errorLevel == 0;
                        } else if ("echo %errorlevel%".equals(nextLine)) {
                            printedErrorLevel = true;
                        }
                    }
                }
            }
        } catch (IOException e) {
            return false;
        }
    }
}
