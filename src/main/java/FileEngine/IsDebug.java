package FileEngine;

public class IsDebug {
    private static volatile boolean isDebugVar = false;
    private static volatile boolean isExecuted = false;

    public static boolean isDebug() {
        if (!isExecuted) {
            isExecuted = true;
            try {
                String res = System.getProperty("File_Engine_Debug");
                isDebugVar =  "true".equalsIgnoreCase(res);
            } catch (Exception e) {
                isDebugVar = false;
            }
        }
        return isDebugVar;
    }
}
