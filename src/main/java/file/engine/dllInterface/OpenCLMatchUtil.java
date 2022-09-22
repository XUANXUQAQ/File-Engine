package file.engine.dllInterface;

import java.util.function.Consumer;

public class OpenCLMatchUtil {

    private static final boolean isAvailable;

    static {
//        boolean isLoaded;
//        try {
//            System.load(Path.of("user/openclMatchUtil.dll").toAbsolutePath().toString());
//            isLoaded = true;
//        } catch (UnsatisfiedLinkError e) {
//            isLoaded = false;
//        }
//        if (isLoaded) {
//            isAvailable = isOpenCLAvailable();
//        } else {
//            isAvailable = false;
//        }
        isAvailable = false;
    }

    public static boolean isOpenCLAvailableOnSystem() {
        return isAvailable;
    }

    private static native boolean isOpenCLAvailable();

    public static native int check(Object[] recordsToCheck,
                                   int size,
                                   String[] searchCase,
                                   boolean isIgnoreCase,
                                   String searchText,
                                   String[] keywords,
                                   String[] keywordsLowerCase,
                                   boolean[] isKeywordPath,
                                   Consumer<String> resultCollector);
}