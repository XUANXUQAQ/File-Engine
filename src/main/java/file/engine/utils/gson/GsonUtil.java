package file.engine.utils.gson;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.util.Map;

public class GsonUtil {
    private static volatile GsonUtil INSTANCE = null;
    private final GsonBuilder gsonBuilder = new GsonBuilder();
    private final DataDataTypeAdaptor dataDataTypeAdaptor = new DataDataTypeAdaptor();

    private GsonUtil() {
    }

    public static GsonUtil getInstance() {
        if (INSTANCE == null) {
            synchronized (GsonUtil.class) {
                if (INSTANCE == null) {
                    INSTANCE = new GsonUtil();
                }
            }
        }
        return INSTANCE;
    }

    @SuppressWarnings("rawtypes")
    public Gson getGson() {
        return gsonBuilder.setPrettyPrinting().registerTypeAdapter(new TypeToken<Map>(){}.getType(), dataDataTypeAdaptor).create();
    }
}
