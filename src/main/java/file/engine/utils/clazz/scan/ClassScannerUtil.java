package file.engine.utils.clazz.scan;

import file.engine.utils.system.properties.IsDebug;

import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.util.HashSet;
import java.util.Set;
import java.util.function.BiConsumer;

public class ClassScannerUtil {

    private static HashSet<String> classWithAnnotation = null;
    private static final boolean isDebug = IsDebug.isDebug();

    private static Set<String> searchClasses(String packageName) {
        return ScannerExecutor.getInstance().search(packageName);
    }

    /**
     * 查找所有含有注解的方法，每找到一个就调用一次doFunction
     *
     * @param cl         注解类
     * @param doFunction 方法
     * @throws ClassNotFoundException 未找到类
     */
    public static void searchAndRun(Class<? extends Annotation> cl, BiConsumer<Class<? extends Annotation>, Method> doFunction) throws ClassNotFoundException {
        String packageName = "file.engine";
        Set<String> classNames = searchClasses(packageName);
        Class<?> c;
        Method[] methods;
        if (isDebug) {
            if (classWithAnnotation == null) {
                classWithAnnotation = new HashSet<>();
            }
        }
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
                    if (isDebug) {
                        classWithAnnotation.add(className);
                    }
                }
            }
        }
    }

    public static void printClassesWithAnnotation() {
        System.out.println(classWithAnnotation);
    }
}
