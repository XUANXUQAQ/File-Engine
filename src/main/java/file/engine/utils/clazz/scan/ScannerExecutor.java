package file.engine.utils.clazz.scan;

import java.util.Set;

public class ScannerExecutor implements Scan {
    private volatile static ScannerExecutor instance;

    @Override
    public Set<String> search(String packageName) {
        Scan fileSc = new FileScanner();
        Set<String> fileSearch = fileSc.search(packageName);
        Scan jarScanner = new JarScanner();
        Set<String> jarSearch = jarScanner.search(packageName);
        fileSearch.addAll(jarSearch);
        return fileSearch;
    }

    private ScannerExecutor() {
    }

    public static ScannerExecutor getInstance() {
        if (instance == null) {
            synchronized (ScannerExecutor.class) {
                if (instance == null) {
                    instance = new ScannerExecutor();
                }
            }
        }
        return instance;
    }
}
