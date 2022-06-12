package file.engine.utils.system.properties;

public class IsPreview {
    private static final boolean isPreviewVar;

    static {
        String res = System.getProperty("File_Engine_Preview");
        isPreviewVar = "true".equalsIgnoreCase(res);
    }

    public static boolean isPreview() {
        return isPreviewVar;
    }
}
