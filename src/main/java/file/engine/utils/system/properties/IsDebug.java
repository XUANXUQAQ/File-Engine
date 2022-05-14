package file.engine.utils.system.properties;

public class IsDebug {
    private static final boolean isDebugVar;

    static {
        String res = System.getProperty("File_Engine_Debug");
        isDebugVar = "true".equalsIgnoreCase(res);
    }

    public static boolean isDebug() {
        return isDebugVar;
    }
}
