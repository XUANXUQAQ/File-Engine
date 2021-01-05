package FileEngine.utils.classScan;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Set;

public class ClassScannerUtil {

    private static Set<String> searchClasses(String packageName){
        return ScannerExecutor.getInstance().search(packageName);
    }

    public static void executeStaticMethodByName(String methodName) throws InvocationTargetException, IllegalAccessException, ClassNotFoundException {
        String packageName = "FileEngine";
        Set<String> classNames = searchClasses(packageName);
        if (classNames != null) {
            for (String className : classNames) {
                Class<?> c = Class.forName(className);
                Method method;
                try {
                    method = c.getMethod(methodName);
                    method.invoke(null);
                } catch (NoSuchMethodException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
