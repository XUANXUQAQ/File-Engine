package file.engine.utils.system.properties;

import java.util.concurrent.atomic.AtomicBoolean;

public class IsPreview {
    private static final AtomicBoolean isPreviewVar = new AtomicBoolean(false);
    private static final AtomicBoolean isExecuted = new AtomicBoolean(false);

    public static boolean isPreview() {
        if (isExecuted.get()) {
            return isPreviewVar.get();
        }
        isExecuted.set(true);
        try {
            String res = System.getProperty("File_Engine_Preview");
            isPreviewVar.set("true".equalsIgnoreCase(res));
        } catch (Exception e) {
            isPreviewVar.set(false);
        }
        return isPreview();
    }
}
