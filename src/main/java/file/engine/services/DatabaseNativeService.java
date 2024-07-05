package file.engine.services;

import cn.hutool.http.HttpGlobalConfig;
import cn.hutool.http.HttpRequest;
import cn.hutool.http.HttpUtil;
import com.google.gson.Gson;
import file.engine.annotation.EventListener;
import file.engine.annotation.EventRegister;
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
import file.engine.utils.ProcessUtil;
import file.engine.utils.ThreadPoolUtil;
import file.engine.utils.file.FileUtil;
import file.engine.utils.gson.GsonUtil;
import lombok.Getter;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

@Slf4j
@SuppressWarnings("unchecked")
public class DatabaseNativeService {
    private static final String CORE_START_CMD = Path.of(Constants.FILE_ENGINE_CORE_DIR + Constants.FILE_ENGINE_CORE_CMD_NAME).toAbsolutePath().toString();
    @Getter
    private static int port = 50721;
    private static final int MAX_RESULT_NUMBER = 200;
    private static final String CORE_URL = "http://127.0.0.1:%d";
    private static final String PORT_CMD_PLACEHOLDER = "${PORT}";
    private static final String CORE_CONFIG = Constants.FILE_ENGINE_CORE_DIR + "user" + File.separator + "settings.json";
    private static volatile long connectionEstablishedTime = 0;
    private static final Set<String> searchTaskUUIDSet = ConcurrentHashMap.newKeySet();

    public static void closeConnections() {
        String url = getUrl() + "/closeConnections";
        HttpRequest.delete(url).timeout(HttpGlobalConfig.getTimeout()).execute().close();
    }

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

    private static String getUrl() {
        return String.format(CORE_URL, port);
    }

    private static int getRandomPort() {
        try (ServerSocket serverSocket = new ServerSocket(0)) {
            return serverSocket.getLocalPort();
        } catch (Exception e) {
            log.error(e.getMessage(), e);
            throw new RuntimeException(e);
        }
    }

    @SneakyThrows
    private static void startCore() {
        String startCmd = Files.readString(Path.of(CORE_START_CMD));
        port = getRandomPort();
        if (!startCmd.contains(PORT_CMD_PLACEHOLDER)) {
            throw new RuntimeException("Port placeholder not found");
        }
        startCmd = startCmd.replace(PORT_CMD_PLACEHOLDER, String.valueOf(port));
        Runtime.getRuntime().exec(new String[]{"cmd.exe", "/c", startCmd}, null, new File(Constants.FILE_ENGINE_CORE_DIR));
        final long startTime = System.currentTimeMillis();
        while (!FileUtil.isFileExist(CORE_CONFIG)) {
            TimeUnit.SECONDS.sleep(1);
            if (System.currentTimeMillis() - startTime > 10_000) {
                throw new RuntimeException("Start File-Engine-Core failed");
            }
        }
    }

    @EventRegister(registerClass = StartCoreEvent.class)
    private static void initCore(Event event) {
        try {
            startCore();
            Constants.Enums.DatabaseStatus status = null;
            final long startTime = System.currentTimeMillis();
            do {
                try {
                    status = getStatus();
                    TimeUnit.MILLISECONDS.sleep(250);
                } catch (Exception ignored) {
                    // ignored
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
        } catch (Exception e) {
            log.error(e.getMessage(), e);
            throw new RuntimeException(e);
        }
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
            Gson gson = GsonUtil.getInstance().getGson();
            String coreJson = gson.toJson(configEntity.getCoreConfigEntity());
            String url = getUrl() + "/config";
            HttpUtil.post(url, coreJson);
        }
    }

    @EventRegister(registerClass = PrepareSearchEvent.class)
    private static void prepareSearchEvent(Event event) {
        String url = getUrl() + "/prepareSearch";
        sendSearchToCore(url, (PrepareSearchEvent) event);
    }

    @EventRegister(registerClass = StartSearchEvent.class)
    private static void startSearchEvent(Event event) {
        connectionEstablishedTime = System.currentTimeMillis();
        synchronized (DatabaseNativeService.class) {
            String url = getUrl() + "/searchAsync";
            sendSearchToCore(url, (StartSearchEvent) event);
        }
    }

    private static void sendSearchToCore(String url, StartSearchEvent startSearchEvent) {
        String removeTaskUrl = getUrl() + "/result";
        searchTaskUUIDSet.forEach(uuid -> HttpRequest.delete(removeTaskUrl + "?uuid=" + uuid).execute().close());
        searchTaskUUIDSet.clear();
        HashMap<String, Object> params = new HashMap<>();
        String[] searchCase = startSearchEvent.searchCase.get();
        if (searchCase != null) {
            params.put("searchText", String.join(";", startSearchEvent.keywords.get()) + "|" + String.join(";", searchCase));
        } else {
            params.put("searchText", startSearchEvent.searchText.get());
        }
        params.put("maxResultNum", MAX_RESULT_NUMBER);
        String paramsStr = HttpUtil.toParams(params);
        String taskUUID = HttpUtil.post(url + "?" + paramsStr, Collections.emptyMap());
        searchTaskUUIDSet.add(taskUUID);
        startSearchEvent.setReturnValue(taskUUID);
    }

    @EventListener(listenClass = SearchBarCloseEvent.class)
    private static void stopSearchEvent(Event event) {
        String url = getUrl() + "/search";
        HttpRequest.delete(url).timeout(HttpGlobalConfig.getTimeout()).execute().close();
        String removeTaskUrl = getUrl() + "/result";
        searchTaskUUIDSet.forEach(uuid -> HttpRequest.delete(removeTaskUrl + "?uuid=" + uuid).execute().close());
        searchTaskUUIDSet.clear();
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
        try {
            TimeUnit.SECONDS.sleep(5);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        do {
            final long start = System.currentTimeMillis();
            try {
                TimeUnit.SECONDS.sleep(1);
                if (!ProcessUtil.isProcessExist("fileSearcherUSN.exe") &&
                        getStatus() == Constants.Enums.DatabaseStatus.NORMAL) {
                    break;
                }
                // 超时时间10分钟
                if (System.currentTimeMillis() - start > 10 * 60 * 1000) {
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
    }
}
