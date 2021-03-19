package file.engine.utils.clazz.scan;

import java.util.Set;

public interface Scan {

    String CLASS_SUFFIX = ".class";

    Set<String> search(String packageName);
}
