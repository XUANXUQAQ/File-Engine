package file.engine.utils;

import java.lang.reflect.Field;
import java.util.Date;

public class BeanUtil {

    public static void copyMatchingFields(Object fromObj, Object toObj) {
        if (fromObj == null || toObj == null)
            throw new NullPointerException("Source and destination objects must be non-null");

        Class<?> fromClass = fromObj.getClass();
        Class<?> toClass = toObj.getClass();

        Field[] fields = fromClass.getDeclaredFields();
        for (Field f : fields) {
            try {
                Field t = toClass.getDeclaredField(f.getName());
                if (t.getType() == f.getType()) {
                    // extend this if to copy more immutable types if interested
                    if (t.getType() == String.class
                            || t.getType() == int.class || t.getType() == Integer.class
                            || t.getType() == char.class || t.getType() == Character.class) {
                        f.setAccessible(true);
                        t.setAccessible(true);
                        t.set(toObj, f.get(fromObj));
                    } else if (t.getType() == Date.class) {
                        // dates are not immutable, so clone non-null dates into the destination object
                        Date d = (Date) f.get(fromObj);
                        f.setAccessible(true);
                        t.setAccessible(true);
                        t.set(toObj, d != null ? d.clone() : null);
                    }
                }
            } catch (NoSuchFieldException ex) {
                // skip it
            } catch (IllegalAccessException ex) {
                ex.printStackTrace();
            }
        }
    }
}
