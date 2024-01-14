package file.engine.services;

import cn.hutool.http.HttpGlobalConfig;
import cn.hutool.http.HttpRequest;
import cn.hutool.http.HttpUtil;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
import file.engine.configs.AllConfigs;
import file.engine.configs.ConfigEntity;
import file.engine.configs.Constants;
import file.engine.configs.core.CoreConfigEntity;
import file.engine.configs.core.ResultEntity;
import file.engine.event.handler.Event;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.database.*;
import file.engine.event.handler.impl.frame.searchBar.SearchBarCloseEvent;
import file.engine.event.handler.impl.frame.searchBar.SearchBarReadyEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.services.utils.OpenFileUtil;
import file.engine.utils.ProcessUtil;
import file.engine.utils.file.FileUtil;
import file.engine.utils.gson.GsonUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.net.BindException;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.TimeUnit;

@Slf4j
@SuppressWarnings("unchecked")
public class DatabaseNativeService {
    private static final String coreFile = Path.of(Constants.FILE_ENGINE_CORE_DIR + Constants.FILE_ENGINE_CORE_NAME).toAbsolutePath().toString();
    private static int port = 50721;
    private static final String coreUrl = "http://127.0.0.1:%d";
    private static final String coreConfigFile = Constants.FILE_ENGINE_CORE_DIR + "user/settings.json";

    public static List<String> getTop8Caches() {
        String url = getUrl() + "/frequentResult";
        HashMap<String, Object> params = new HashMap<>();
        params.put("num", 8);
        String res = HttpUtil.get(url, params);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, List.class);
    }

    public static Set<String> getCache() {
        String url = getUrl() + "/cache";
        String res = HttpUtil.get(url);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, Set.class);
    }

    public static Map<String, Object> getPriorityMap() {
        String url = getUrl() + "/suffixPriority";
        Gson gson = GsonUtil.getInstance().getGson();
        String res = HttpUtil.get(url);
        return gson.fromJson(res, Map.class);
    }

    public static ResultEntity getCacheAndPriorityResults(int startIndex) {
        String cacheResults = getUrl() + "/cacheResult";
        return getResultEntity(startIndex, cacheResults);
    }

    public static ResultEntity getResults(int startIndex) {
        String url = getUrl() + "/result";
        return getResultEntity(startIndex, url);
    }

    public static Map<String, String> getGpuDevices() {
        String s = getUrl() + "/gpu";
        String res = HttpUtil.get(s);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, Map.class);
    }

    public static CoreConfigEntity getCoreConfigs() {
        String url = getUrl() + "/config";
        String res = HttpUtil.get(url);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, CoreConfigEntity.class);
    }

    private static ResultEntity getResultEntity(int startIndex, String url) {
        HashMap<String, Object> params = new HashMap<>();
        params.put("startIndex", startIndex);
        String res = HttpUtil.get(url, params);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, ResultEntity.class);
    }

    /**
     * 获取数据库状态
     *
     * @return 数据库状态
     */
    public static Constants.Enums.DatabaseStatus getStatus() {
        String url = getUrl() + "/status";
        String status = HttpUtil.get(url);
        return Constants.Enums.DatabaseStatus.valueOf(status);
    }

    private static int getAvailablePort() {
        if (IsDebug.isDebug()) {
            return 50721;
        }
        int port;
        do {
            Random random = new Random();
            port = random.nextInt(20000, 65535);
        } while (!isAvailable(port));
        return port;
    }

    private static String getUrl() {
        return String.format(coreUrl, port);
    }

    private static boolean isAvailable(int port) {
        try {
            new ServerSocket(port).close();
        } catch (IOException e) {
            if (e instanceof BindException) {
                return false;
            }
        }
        return true;
    }

    @SneakyThrows
    private static void startCore() {
        Path coreSettingsPath = Path.of(coreConfigFile);
        if (FileUtil.isFileExist(coreConfigFile)) {
            String coreConfigs = Files.readString(coreSettingsPath, StandardCharsets.UTF_8);
            Gson gson = GsonUtil.getInstance().getGson();
            Map<String, Object> coreConfigMap = gson.fromJson(coreConfigs, Map.class);
            port = getAvailablePort();
            coreConfigMap.put("port", port);
            coreConfigs = gson.toJson(coreConfigMap);
            Files.writeString(coreSettingsPath, coreConfigs);
        }
        OpenFileUtil.openWithAdmin(coreFile, true);
    }

    @EventRegister(registerClass = StartCoreEvent.class)
    @SneakyThrows
    private static void initCore(Event event) {
        startCore();
        while (!ProcessUtil.isProcessExist(Constants.FILE_ENGINE_CORE_NAME)) {
            TimeUnit.MILLISECONDS.sleep(100);
        }
        Constants.Enums.DatabaseStatus status = null;
        final long startTime = System.currentTimeMillis();
        do {
            try {
                status = getStatus();
                TimeUnit.MILLISECONDS.sleep(250);
            } catch (Exception ignored) {
            }
        } while (status != Constants.Enums.DatabaseStatus.NORMAL && System.currentTimeMillis() - startTime < 10_000);
    }

    @EventListener(listenClass = SearchBarReadyEvent.class)
    private static void searchBarVisibleListener(Event event) {
        String url = getUrl() + "/flushFileChanges";
        HttpUtil.post(url, Collections.emptyMap());
    }

    @EventListener(listenClass = SetConfigsEvent.class)
    private static void setConfigs(Event event) {
        SetConfigsEvent setConfigsEvent = (SetConfigsEvent) event;
        ConfigEntity configEntity = setConfigsEvent.getConfigs();
        if (configEntity != null) {
            configEntity = AllConfigs.getInstance().getConfigEntity();
            Gson gson = GsonUtil.getInstance().getGson();
            JsonElement jsonTree = gson.toJsonTree(configEntity.getCoreConfigEntity());
            Map<String, Object> configMap = gson.fromJson(jsonTree, Map.class);
            configMap.put("port", port);
            String url = getUrl() + "/config";
            HttpUtil.post(url, gson.toJson(configMap));
        }
    }

    @EventRegister(registerClass = PrepareSearchEvent.class)
    private static void prepareSearchEvent(Event event) {
        String url = getUrl() + "/prepareSearch";
        HashMap<String, Object> params = new HashMap<>();
        params.put("searchText", ((PrepareSearchEvent) event).searchText.get());
        params.put("maxResultNum", 200);
        String paramsStr = HttpUtil.toParams(params);
        String taskUUID = HttpUtil.post(url + "?" + paramsStr, Collections.emptyMap());
        event.setReturnValue(taskUUID);
    }

    @EventRegister(registerClass = StartSearchEvent.class)
    private static void startSearchEvent(Event event) {
        String url = getUrl() + "/searchAsync";
        HashMap<String, Object> params = new HashMap<>();
        params.put("searchText", ((StartSearchEvent) event).searchText.get());
        params.put("maxResultNum", 200);
        String paramsStr = HttpUtil.toParams(params);
        String taskUUID = HttpUtil.post(url + "?" + paramsStr, Collections.emptyMap());
        event.setReturnValue(taskUUID);
    }

    @EventListener(listenClass = SearchBarCloseEvent.class)
    private static void stopSearchEvent(Event event) {
        String url = getUrl() + "/search";
        HttpRequest.delete(url).timeout(HttpGlobalConfig.getTimeout()).execute().close();
    }

    @EventRegister(registerClass = AddToCacheEvent.class)
    private static void addToCacheEvent(Event event) {
        String url = getUrl() + "/cache";
        HashMap<String, Object> params = new HashMap<>();
        params.put("path", ((AddToCacheEvent) event).path);
        String paramsStr = HttpUtil.toParams(params);
        HttpUtil.post(url + "?" + paramsStr, Collections.emptyMap());
    }

    @EventRegister(registerClass = DeleteFromCacheEvent.class)
    private static void deleteFromCacheEvent(Event event) {
        String url = getUrl() + "/cache";
        HashMap<String, Object> params = new HashMap<>();
        params.put("path", ((DeleteFromCacheEvent) event).path);
        String paramsStr = HttpUtil.toParams(params);
        HttpRequest.delete(url + "?" + paramsStr).timeout(HttpGlobalConfig.getTimeout()).execute().close();
    }

    @EventRegister(registerClass = UpdateDatabaseEvent.class)
    private static void updateDatabaseEvent(Event event) {
        String url = getUrl() + "/update";
        UpdateDatabaseEvent updateDatabaseEvent = (UpdateDatabaseEvent) event;
        HashMap<String, Object> params = new HashMap<>();
        params.put("isDropPrevious", updateDatabaseEvent.isDropPrevious);
        String paramsStr = HttpUtil.toParams(params);
        HttpUtil.post(url + "?" + paramsStr, Collections.emptyMap());
        do {
            try {
                TimeUnit.SECONDS.sleep(1);
                if (!ProcessUtil.isProcessExist("fileSearcherUSN.exe")) {
                    break;
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } while (true);
    }

    @EventRegister(registerClass = OptimiseDatabaseEvent.class)
    private static void optimizeDatabaseEvent(Event event) {
        String url = getUrl() + "/optimize";
        HttpUtil.post(url, Collections.emptyMap());
    }

    @EventRegister(registerClass = AddToSuffixPriorityMapEvent.class)
    private static void addToSuffixPriorityMapEvent(Event event) {
        AddToSuffixPriorityMapEvent addToSuffixPriorityMapEvent = (AddToSuffixPriorityMapEvent) event;
        String url = getUrl() + "/suffixPriority";
        HashMap<String, Object> params = new HashMap<>();
        params.put("suffix", addToSuffixPriorityMapEvent.suffix);
        params.put("priority", addToSuffixPriorityMapEvent.priority);
        String paramsStr = HttpUtil.toParams(params);
        HttpUtil.post(url + "?" + paramsStr, Collections.emptyMap());
    }

    @EventRegister(registerClass = ClearSuffixPriorityMapEvent.class)
    private static void clearSuffixPriorityMapEvent(Event event) {
        String url = getUrl() + "/clearSuffixPriority";
        HttpRequest.delete(url).timeout(HttpGlobalConfig.getTimeout()).execute().close();
    }

    @EventRegister(registerClass = DeleteFromSuffixPriorityMapEvent.class)
    private static void deleteFromSuffixPriorityMapEvent(Event event) {
        String url = getUrl() + "/suffixPriority";
        DeleteFromSuffixPriorityMapEvent deleteFromSuffixPriorityMapEvent = (DeleteFromSuffixPriorityMapEvent) event;
        HashMap<String, Object> params = new HashMap<>();
        params.put("suffix", deleteFromSuffixPriorityMapEvent.suffix);
        String paramsStr = HttpUtil.toParams(params);
        HttpRequest.delete(url + "?" + paramsStr).timeout(HttpGlobalConfig.getTimeout()).execute().close();
    }

    @EventRegister(registerClass = UpdateSuffixPriorityEvent.class)
    private static void updateSuffixPriorityEvent(Event event) {
        UpdateSuffixPriorityEvent updateSuffixPriorityEvent = (UpdateSuffixPriorityEvent) event;
        String url = getUrl() + "/suffixPriority";
        HashMap<String, Object> params = new HashMap<>();
        params.put("oldSuffix", updateSuffixPriorityEvent.originSuffix);
        params.put("newSuffix", updateSuffixPriorityEvent.newSuffix);
        params.put("priority", updateSuffixPriorityEvent.newPriority);
        String paramsStr = HttpUtil.toParams(params);
        HttpRequest.put(url + "?" + paramsStr).timeout(HttpGlobalConfig.getTimeout()).execute().close();
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void restartEvent(Event event) {
        String url = getUrl() + "/close";
        HttpUtil.post(url, Collections.emptyMap());
        try {
            ProcessUtil.waitForProcess(Constants.FILE_ENGINE_CORE_NAME, 500, 5000);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
