package file.engine.services;

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
import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.configs.SetConfigsEvent;
import file.engine.event.handler.impl.database.*;
import file.engine.event.handler.impl.frame.searchBar.SearchBarCloseEvent;
import file.engine.event.handler.impl.frame.searchBar.SearchBarReadyEvent;
import file.engine.event.handler.impl.stop.RestartEvent;
import file.engine.services.utils.CoreUtil;
import file.engine.utils.ProcessUtil;
import file.engine.utils.ThreadPoolUtil;
import file.engine.utils.file.FileUtil;
import file.engine.utils.gson.GsonUtil;
import file.engine.utils.system.properties.IsDebug;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.io.File;
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
    private static final String CORE_START_CMD = Path.of(Constants.FILE_ENGINE_CORE_DIR + Constants.FILE_ENGINE_CORE_CMD_NAME).toAbsolutePath().toString();
    private static int port = 50721;
    private static final int MAX_RESULT_NUMBER = 200;
    private static final String CORE_CONFIG_FILE = Constants.FILE_ENGINE_CORE_DIR + "user/settings.json";
    private static volatile long connectionEstablishedTime = 0;

    public static void closeConnections() {
        String url = "/closeConnections";
        CoreUtil.getCoreResult("DELETE", url, port);
    }

    public static List<String> getTop8Caches() {
        String url = "/frequentResult";
        HashMap<String, Object> params = new HashMap<>();
        params.put("num", 8);
        String paramsStr = HttpUtil.toParams(params);
        String res = CoreUtil.get(url + "?" + paramsStr, port);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, List.class);
    }

    public static Set<String> getCache() {
        String url = "/cache";
        String res = CoreUtil.get(url, port);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, Set.class);
    }

    public static Map<String, Object> getPriorityMap() {
        String url = "/suffixPriority";
        Gson gson = GsonUtil.getInstance().getGson();
        String res = CoreUtil.get(url, port);
        return gson.fromJson(res, Map.class);
    }

    public static ResultEntity getCacheAndPriorityResults(int startIndex) {
        String cacheResults = "/cacheResult";
        return getResultEntity(startIndex, cacheResults);
    }

    public static ResultEntity getResults(int startIndex) {
        String url = "/result";
        return getResultEntity(startIndex, url);
    }

    public static Map<String, String> getGpuDevices() {
        String s = "/gpu";
        String res = CoreUtil.get(s, port);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, Map.class);
    }

    public static CoreConfigEntity getCoreConfigs() {
        String url = "/config";
        String res = CoreUtil.get(url, port);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, CoreConfigEntity.class);
    }

    private static ResultEntity getResultEntity(int startIndex, String url) {
        HashMap<String, Object> params = new HashMap<>();
        params.put("startIndex", startIndex);
        String paramsStr = HttpUtil.toParams(params);
        String res = CoreUtil.get(url + "?" + paramsStr, port);
        Gson gson = GsonUtil.getInstance().getGson();
        return gson.fromJson(res, ResultEntity.class);
    }

    /**
     * 获取数据库状态
     *
     * @return 数据库状态
     */
    public static Constants.Enums.DatabaseStatus getStatus() {
        String url = "/status";
        String status = CoreUtil.get(url, port);
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
        Path coreSettingsPath = Path.of(CORE_CONFIG_FILE);
        if (FileUtil.isFileExist(CORE_CONFIG_FILE)) {
            String coreConfigs = Files.readString(coreSettingsPath, StandardCharsets.UTF_8);
            Gson gson = GsonUtil.getInstance().getGson();
            Map<String, Object> coreConfigMap = gson.fromJson(coreConfigs, Map.class);
            port = getAvailablePort();
            coreConfigMap.put("port", port);
            coreConfigs = gson.toJson(coreConfigMap);
            Files.writeString(coreSettingsPath, coreConfigs);
        }
        String startCmd = Files.readString(Path.of(CORE_START_CMD));
        Runtime.getRuntime().exec(new String[]{"cmd.exe", "/c", startCmd}, null, new File(Constants.FILE_ENGINE_CORE_DIR));
    }

    @EventRegister(registerClass = StartCoreEvent.class)
    @SneakyThrows
    private static void initCore(Event event) {
        startCore();
        Constants.Enums.DatabaseStatus status = null;
        final long startTime = System.currentTimeMillis();
        do {
            try {
                status = getStatus();
                TimeUnit.MILLISECONDS.sleep(250);
            } catch (Exception ignored) {
            }
        } while (status != Constants.Enums.DatabaseStatus.NORMAL && System.currentTimeMillis() - startTime < 10_000);
        log.info("File-Engine-Core启动完成");
        ThreadPoolUtil.getInstance().executeTask(() -> {
            EventManagement eventManagement = EventManagement.getInstance();
            while (eventManagement.notMainExit()) {
                //数据库连接超过3分钟未使用
                if (connectionEstablishedTime != 0 && System.currentTimeMillis() - connectionEstablishedTime > 3 * 60 * 1000) {
                    connectionEstablishedTime = 0;
                    synchronized (DatabaseNativeService.class) {
                        closeConnections();
                    }
                }
                try {
                    TimeUnit.MILLISECONDS.sleep(100);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        });
    }

    @EventListener(listenClass = SearchBarReadyEvent.class)
    private static void searchBarVisibleListener(Event event) {
        String url = "/flushFileChanges";
        CoreUtil.post(url, port);
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
            String url = "/config";
            String configJson = gson.toJson(configMap);
            HashMap<String, String> paramMap = new HashMap<>();
            paramMap.put("config", configJson);
            String paramsStr = HttpUtil.toParams(paramMap);
            CoreUtil.post(url + "?" + paramsStr, port);
        }
    }

    @EventRegister(registerClass = PrepareSearchEvent.class)
    private static void prepareSearchEvent(Event event) {
        String url = "/prepareSearch";
        sendSearchToCore((PrepareSearchEvent) event, url);
    }

    @EventRegister(registerClass = StartSearchEvent.class)
    private static void startSearchEvent(Event event) {
        connectionEstablishedTime = System.currentTimeMillis();
        synchronized (DatabaseNativeService.class) {
            String url = "/searchAsync";
            sendSearchToCore((StartSearchEvent) event, url);
        }
    }

    private static void sendSearchToCore(StartSearchEvent startSearchEvent, String url) {
        HashMap<String, Object> params = new HashMap<>();
        String[] searchCase = startSearchEvent.searchCase.get();
        if (searchCase != null) {
            params.put("searchText", startSearchEvent.searchText.get() + "|" + String.join(",", searchCase));
        } else {
            params.put("searchText", startSearchEvent.searchText.get());
        }
        params.put("maxResultNum", MAX_RESULT_NUMBER);
        String paramsStr = HttpUtil.toParams(params);
        String taskUUID = CoreUtil.post(url + "?" + paramsStr, port);
        startSearchEvent.setReturnValue(taskUUID);
    }

    @EventListener(listenClass = SearchBarCloseEvent.class)
    private static void stopSearchEvent(Event event) {
        String url = "/search";
        CoreUtil.getCoreResult("DELETE", url, port);
    }

    @EventRegister(registerClass = AddToCacheEvent.class)
    private static void addToCacheEvent(Event event) {
        String url = "/cache";
        HashMap<String, Object> params = new HashMap<>();
        params.put("path", ((AddToCacheEvent) event).path);
        String paramsStr = HttpUtil.toParams(params);
        CoreUtil.post(url + "?" + paramsStr, port);
    }

    @EventRegister(registerClass = DeleteFromCacheEvent.class)
    private static void deleteFromCacheEvent(Event event) {
        String url = "/cache";
        HashMap<String, Object> params = new HashMap<>();
        params.put("path", ((DeleteFromCacheEvent) event).path);
        String paramsStr = HttpUtil.toParams(params);
        CoreUtil.getCoreResult("DELETE", url + "?" + paramsStr, port);
    }

    @EventRegister(registerClass = UpdateDatabaseEvent.class)
    private static void updateDatabaseEvent(Event event) {
        String url = "/update";
        UpdateDatabaseEvent updateDatabaseEvent = (UpdateDatabaseEvent) event;
        HashMap<String, Object> params = new HashMap<>();
        params.put("isDropPrevious", updateDatabaseEvent.isDropPrevious);
        String paramsStr = HttpUtil.toParams(params);
        CoreUtil.post(url + "?" + paramsStr, port);
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
        String url = "/optimize";
        CoreUtil.post(url, port);
    }

    @EventRegister(registerClass = AddToSuffixPriorityMapEvent.class)
    private static void addToSuffixPriorityMapEvent(Event event) {
        AddToSuffixPriorityMapEvent addToSuffixPriorityMapEvent = (AddToSuffixPriorityMapEvent) event;
        String url = "/suffixPriority";
        HashMap<String, Object> params = new HashMap<>();
        params.put("suffix", addToSuffixPriorityMapEvent.suffix);
        params.put("priority", addToSuffixPriorityMapEvent.priority);
        String paramsStr = HttpUtil.toParams(params);
        CoreUtil.post(url + "?" + paramsStr, port);
    }

    @EventRegister(registerClass = ClearSuffixPriorityMapEvent.class)
    private static void clearSuffixPriorityMapEvent(Event event) {
        String url = "/clearSuffixPriority";
        CoreUtil.getCoreResult("DELETE", url, port);
    }

    @EventRegister(registerClass = DeleteFromSuffixPriorityMapEvent.class)
    private static void deleteFromSuffixPriorityMapEvent(Event event) {
        String url = "/suffixPriority";
        DeleteFromSuffixPriorityMapEvent deleteFromSuffixPriorityMapEvent = (DeleteFromSuffixPriorityMapEvent) event;
        HashMap<String, Object> params = new HashMap<>();
        params.put("suffix", deleteFromSuffixPriorityMapEvent.suffix);
        String paramsStr = HttpUtil.toParams(params);
        CoreUtil.getCoreResult("DELETE", url + "?" + paramsStr, port);
    }

    @EventRegister(registerClass = UpdateSuffixPriorityEvent.class)
    private static void updateSuffixPriorityEvent(Event event) {
        UpdateSuffixPriorityEvent updateSuffixPriorityEvent = (UpdateSuffixPriorityEvent) event;
        String url = "/suffixPriority";
        HashMap<String, Object> params = new HashMap<>();
        params.put("oldSuffix", updateSuffixPriorityEvent.originSuffix);
        params.put("newSuffix", updateSuffixPriorityEvent.newSuffix);
        params.put("priority", updateSuffixPriorityEvent.newPriority);
        String paramsStr = HttpUtil.toParams(params);
        CoreUtil.getCoreResult("PUT", url + "?" + paramsStr, port);
    }

    @EventListener(listenClass = RestartEvent.class)
    private static void restartEvent(Event event) {
        String url = "/close";
        CoreUtil.post(url, port);
    }
}
