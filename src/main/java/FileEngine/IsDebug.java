package FileEngine;

import java.util.concurrent.atomic.AtomicBoolean;

public class IsDebug {
    private static final AtomicBoolean isDebugVar = new AtomicBoolean(false);
    private static final AtomicBoolean isExecuted = new AtomicBoolean(false);

    public static boolean isDebug() {
        if (isExecuted.get()) {
            return isDebugVar.get();
        }
        isExecuted.set(true);
        try {
            String res = System.getProperty("File_Engine_Debug");
            isDebugVar.set("true".equalsIgnoreCase(res));
        } catch (Exception e) {
            isDebugVar.set(false);
        }
        return isDebug();
    }
}
