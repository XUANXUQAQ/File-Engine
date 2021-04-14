package file.engine.utils.clazz.scan;

import java.lang.annotation.Annotation;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Set;
import java.util.function.BiConsumer;

public class ClassScannerUtil {

    private static Set<String> searchClasses(String packageName) {
        return ScannerExecutor.getInstance().search(packageName);
    }

    public static void searchAndRun(Class<? extends Annotation> cl, BiConsumer<Class<? extends Annotation>, Method> doFunction) throws ClassNotFoundException, InvocationTargetException, IllegalAccessException {
        String packageName = "file.engine";
        Set<String> classNames = searchClasses(packageName);
        Class<?> c;
        Method[] methods;
        if (classNames == null || classNames.isEmpty()) {
            return;
        }
        for (String className : classNames) {
            c = Class.forName(className);
            methods = c.getDeclaredMethods();
            for (Method eachMethod : methods) {
                eachMethod.setAccessible(true);
                if (eachMethod.isAnnotationPresent(cl)) {
                    doFunction.accept(cl, eachMethod);
                }
            }
        }
    }
}
