package file.engine.utils.clazz.scan;

import java.lang.annotation.Annotation;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Set;

public class ClassScannerUtil {

    private static Set<String> searchClasses(String packageName) {
        return ScannerExecutor.getInstance().search(packageName);
    }

    public static void executeMethodByAnnotation(Class<? extends Annotation> annotationClass, Object instance) throws InvocationTargetException, IllegalAccessException, ClassNotFoundException {
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
                if (eachMethod.isAnnotationPresent(annotationClass)) {
                    eachMethod.invoke(instance);
                }
            }
        }
    }
}
