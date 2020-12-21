package FileEngine.utils.classScan;

import java.util.Set;

public interface Scan {

    String CLASS_SUFFIX = ".class";

    Set<String> search(String packageName);
}
