# 项目结构

## 项目结构为分层结构。

    顶层是UI层，放在frames文件夹下。

    中层是服务层，放在services文件夹下。

    下层是jvm运行环境以及C++库提供的基础支持。C++库在dllInterface文件夹下。

    由服务层调用C++库并注册事件响应，提供基本功能，UI层响应用户的操作并发送事件进行调用。

```
├─MainClass.java                    主启动类，初始化依赖并发出启动事件   
├─utils                             公用工具类   
|    ├─......   
├─services   
| ├─CheckHotKeyService.java         全局快捷键服务，注册和检测键盘快捷键   
| ├─DaemonService.java              守护进程服务，开启和停止守护进程   
| ├─DatabaseService.java            数据库维护服务，负责文件搜索和同步   
| ├─OpenFileService.java            打开可执行文件服务   
| ├─TranslateService.java           翻译服务   
| ├─plugin   
| | ├─system   
| | | ├─Plugin.java                 插件对象   
| | | └PluginService.java           插件服务，提供插件的查询和方法调用   
| ├─download   
| | ├─BasicAuthenticator.java   
| | ├─DownloadManager.java   
| | └DownloadService.java           下载服务，负责下载文件   
| ├─utils                           仅供service调用工具类   
| | ├─......   
├─frames   
| ├─PluginMarket.form   
| ├─PluginMarket.java               插件市场UI界面   
| ├─SearchBar.java                  搜索框界面   
| ├─SetDownloadProgress.java    
| ├─SettingsFrame.form   
| ├─SettingsFrame.java              设置UI界面   
| ├─TaskBar.java                    Windows任务栏图标   
| ├─components                      通用Swing组件   
| | ├─LoadingPanel.java   
| | ├─MouseDragInfo.java   
| | └RoundBorder.java   
├─event   
| ├─handler   
| | ├─Event.java                    事件基类   
| | ├─EventManagement.java          事件处理工具   
| | ├─impl                          具体事件实现   
| | | ├─......   
├─dllInterface   
| ├─gpu   
| | ├─GPUAccelerator.java           GPU加速工具包装类   
| | ├─GPUApiCategory.java           GPU加速API框架枚举类   
| | ├─IGPUAccelerator.java          GPU加速框架接口   
| | ├─CudaAccelerator.java          NVIDIA CUDA加速工具   
| | └OpenclAccelerator.java         OpenCL加速工具   
| ├─EmptyRecycleBin.java            清空回收站工具   
| ├─FileMonitor.java                文件改动监测工具   
| ├─GetHandle.java                  检测windows资源管理器并进行互操作工具   
| ├─GetWindowsKnownFolder.java      获取Windows默认文件夹，如开始菜单，详见MSDN(SHGetKnownFolderPath)
| ├─HotkeyListener.java             Windows全局快捷键注册工具   
| ├─IsLocalDisk.java                检测硬盘是否为本地硬盘以及NTFS文件系统工具   
├─configs   
| ├─AllConfigs.java                 全局配置中心   
| ├─ConfigEntity.java               配置对象   
| ├─AdvancedConfigEntity.java       高级配置对象
| ├─Constants.java                  全局常量   
| └ProxyInfo.java                   网络代理对象，如http socks5   
├─annotation   
| ├─EventListener.java              事件监听注解，添加该注解可以将函数注册为对应事件的回调函数   
| └EventRegister.java               事件处理注解，添加该注解可以将函数注册为相应事件的处理函数。注意：一个事件只能有一个处理函数，可以有多个回调函数。
```

事件处理系统详见[Event_Management](https://github.com/XUANXUQAQ/File-Engine/blob/master/Event_Mangement.md)

## 下面是各个包以及各个依赖的作用。

## Java包部分

- ### frames
  
  - UI层的实现。包含搜索框，设置窗口，任务栏，插件市场窗口的实现。以及一些通用控件。
  
  - UI层使用Java Swing实现，以及Intellij idea中的GUI Designer来构建。
  
  - #### SearchBar
    
    - 搜索框的具体实现，搜索框输入后将会先发出*PrepareSearchTaskEvent*准备搜索任务并进行预搜索，等待用户输入超时后将会发出*StartSearchEvent*开始进行搜索，并开启*mergeResultsThread*线程不断从*DatabaseService.tempResults*获取结果并显示。
    
    - mergeResultsThread同时会检查插件有没有发来结果，如果有也会一并合并。    
  
  - #### TaskBar
    
    - 任务栏的具体实现，有打开设置，重启，退出三个选项。
  
  - #### PluginMarket
    
    - 插件市场窗口的具体实现，可下载插件。
  
  - #### SettingsFrame
    
    - 设置窗口的具体实现，当窗口关闭时将会检查各项设置是否有效并发出SetConfigsEvent，随后由AllConfigs配置中心响应。
  
  - #### components：
    
    - LoadingPanel：通用组件，加载窗口的实现。
    
    - MouseDragInfo：通用组件，鼠标从搜索框拖动到资源管理器时显示的窗口。
    
    - RoundBorder：通用组件，圆角边框的具体实现。

- ### service
  
  - 服务层的实现。包含下载服务，插件加载服务，键盘快捷键监听服务，数据库服务，多语言UI翻译服务。
  
  - **所有的服务启动需要监听一个BootSystemEvent事件**，在该事件发出前各服务不应该互相调用或者被上层调用。
  
  - **所有的服务都应该为单例模式**，在程序中只拥有一个实例。
  
  - #### DownloadService
    
    - 下载服务的具体实现，下载文件并在下载完成后执行回调。
    
    - DownloadManager：下载文件信息的封装。作为发送StartDownloadEvent事件的参数。
    
    - 注册和监听的事件：
      
      - ```java
        @EventRegister(registerClass = StartDownloadEvent.class)
        private static void startDownloadEvent(Event event) {
            StartDownloadEvent startDownloadTask = (StartDownloadEvent) event;
            getInstance().downLoadFromUrl(startDownloadTask.downloadManager);
        }
        
        @EventRegister(registerClass = StartDownloadEvent2.class)
        private static void startDownloadEvent2(Event event) {
            StartDownloadEvent2 startDownloadEvent2 = (StartDownloadEvent2) event;
            getInstance().downLoadFromUrl(startDownloadEvent2.downloadManager);
        }
        
        @EventRegister(registerClass = StopDownloadEvent.class)
        private static void stopDownloadEvent(Event event) {
            StopDownloadEvent stopDownloadTask = (StopDownloadEvent) event;
            getInstance().cancelDownload(stopDownloadTask.downloadManager);
        }
        
        @EventRegister(registerClass = StopDownloadEvent2.class)
        private static void stopDownloadEvent2(Event event) {
            StopDownloadEvent2 stopDownloadEvent2 = (StopDownloadEvent2) event;
            getInstance().cancelDownload(stopDownloadEvent2.downloadManager);
        }
        ```
        
        StartDownloadEvent和StartDownloadEvent2并无本质区别，区别在于构造函数，StartDownloadEvent2不需要直接传入DownloadManager，方便插件进行调用，StopDownloadEvent和StopDownloadEvent2同理。
        
        传入StartDownloadEvent需要使用DownloadManager类描述下载文件信息。
        
        ```java
        public class DownloadManager {
            public final String url;
            public final String savePath;
            public final String fileName;
            private volatile double progress = 0.0;
            private volatile boolean isUserInterrupted = false;
            private volatile Constants.Enums.DownloadStatus downloadStatus;
            private Proxy proxy = null;
            private Authenticator authenticator = null;
        
            public DownloadManager(String url, String fileName, String savePath) {
                this.url = url;
                this.fileName = fileName;
                this.savePath = savePath;
                this.downloadStatus = Constants.Enums.DownloadStatus.DOWNLOAD_NO_TASK;
                ProxyInfo proxyInfo = AllConfigs.getInstance().getProxy();
                setProxy(proxyInfo.type, proxyInfo.address, proxyInfo.port, proxyInfo.userName, proxyInfo.password);
            }
        }
        ```
        
        DownloadManager类中包含**下载地址url**，**保存路径savePath**，**文件名fileName**，**下载进度progress**，**下载状态downloadStatus**
        
        通过DownloadManager可以获取下载的基本信息。
  
  - #### PluginService
    
    - 插件服务，为项目提供插件加载卸载以及基本调用的接口。获取插件的一些基本信息
    
    - Plugin：插件对外暴露的接口。
    
    - 注册和监听的事件
      
      ```java
      @EventRegister(registerClass = GetPluginByNameEvent.class)
      private static void getPluginByNameEvent(Event event) {
          GetPluginByNameEvent event1 = (GetPluginByNameEvent) event;
          PluginInfo pluginInfoByName = getInstance().getPluginInfoByName(event1.pluginName);
          event1.setReturnValue(pluginInfoByName);
      }
      
      @EventRegister(registerClass = GetPluginByIdentifierEvent.class)
      private static void getPluginByIdentifier(Event event) {
          GetPluginByIdentifierEvent event1 = (GetPluginByIdentifierEvent) event;
          PluginInfo pluginInfoByIdentifier = getInstance().getPluginInfoByIdentifier(event1.identifier);
          event1.setReturnValue(pluginInfoByIdentifier);
      }
      
      @EventRegister(registerClass = AddPluginsCanUpdateEvent.class)
      private static void addPluginsCanUpdateEvent(Event event) {
          getInstance().addPluginsCanUpdate(((AddPluginsCanUpdateEvent) event).pluginName);
      }
      
      @EventRegister(registerClass = LoadAllPluginsEvent.class)
      private static void loadAllPluginsEvent(Event event) {
          getInstance().loadAllPlugins(((LoadAllPluginsEvent) event).pluginDirPath);
          checkPluginInfo();
      }
      
      @EventRegister(registerClass = RemoveFromPluginsCanUpdateEvent.class)
      private static void removeFromPluginsCanUpdateEvent(Event event) {
          getInstance().removeFromPluginsCanUpdate(((RemoveFromPluginsCanUpdateEvent) event).pluginName);
      }
      
      @EventListener(listenClass = SetConfigsEvent.class)
      private static void setPluginsCurrentThemeEvent(Event event) {
          var configs = AllConfigs.getInstance().getConfigMap();
          getInstance().configsChanged((Integer) configs.get("defaultBackground"), (Integer) configs.get("labelColor"), (Integer) configs.get("borderColor"), configs);
      }
      
      @EventListener(listenClass = SearchBarReadyEvent.class)
      private static void onSearchBarReady(Event event) {
          SearchBarReadyEvent searchBarReadyEvent = (SearchBarReadyEvent) event;
          getInstance().onSearchBarVisible(searchBarReadyEvent.showingType);
      }
      
      @EventListener(listenClass = RestartEvent.class)
      private static void restartEvent(Event event) {
          getInstance().unloadAllPlugins();
      }
      
      @EventRegister(registerClass = EventProcessedBroadcastEvent.class)
      private static void broadcastEventProcess(Event event) {
          EventProcessedBroadcastEvent eventProcessed = (EventProcessedBroadcastEvent) event;
          PluginService pluginService = getInstance();
          for (PluginInfo each : pluginService.pluginInfoSet) {
              each.plugin.eventProcessed(eventProcessed.c, eventProcessed.eventInstance);
          }
      }
      ```
      
      PluginService中主要对外的事件为GetPluginByNameEvent和GetPluginByIdentifierEvent事件，这两个事件可以获取插件名和插件关键字获取对应的插件对象。
      
      ```java
      public static class PluginInfo {
          public final Plugin plugin;
          public final String name;
          public final String identifier;
      
          private PluginInfo(Plugin plugin, String name, String identifier) {
              this.plugin = plugin;
              this.name = name;
              this.identifier = identifier;
          }
      
          @Override
          public String toString() {
              return name;
          }
      }
      ```
      
      ```java
      public class Plugin {
          public final String name;
          public final String identifier;
          private final Object instance;
          private final ConcurrentHashMap<String, Method> methodHashMap = new ConcurrentHashMap<>();
          private static final HashSet<String> methodList = new HashSet<>();
      }
      ```
      
      Plugin类中包含所有插件的方法，通过反射进行调用。
      
      ```java
      @SuppressWarnings("unchecked")
      private <T> T invokeByKey(String key, Object... args) {
          if (methodHashMap.containsKey(key)) {
              try {
                  return (T) methodHashMap.get(key).invoke(instance, args);
              } catch (Exception e) {
                  throw new RuntimeException(e);
              }
          }
          return null;
      }
      
      @SuppressWarnings("unchecked")
      private <T> T invokeByKeyNoExcept(String key, Object... args) {
          if (methodHashMap.containsKey(key)) {
              try {
                  return (T) methodHashMap.get(key).invoke(instance, args);
              } catch (Exception e) {
                  e.printStackTrace();
              }
          }
          return null;
      }
      
      public void loadPlugin(Map<String, Object> configs) {
          String key = "loadPlugin" + Arrays.toString(new Class<?>[]{Map.class});
          invokeByKey(key, configs);
      }
      ...
      ```
      
      每个函数的意义请参考插件开发文档或插件模板API上的函数注释。
      
      [XUANXUQAQ/File-Engine-Plugin-Template: A File-Engine plugin template (github.com)](https://github.com/XUANXUQAQ/File-Engine-Plugin-Template)

- #### CheckHotKeyService
  
  - 键盘快捷键的监听服务，当键盘快捷键点击后将会发出ShowSearchBarEvent打开搜索框。
  
  - 注册和监听的事件
    
    ```java
    @EventRegister(registerClass = CheckHotKeyAvailableEvent.class)
    private static void checkHotKeyAvailableEvent(Event event) {
        var event1 = (CheckHotKeyAvailableEvent) event;
        event1.setReturnValue(getInstance().isHotkeyAvailable(event1.hotkey));
    }
    
    @EventListener(listenClass = SetConfigsEvent.class)
    private static void registerHotKeyEvent(Event event) {
        getInstance().registerHotkey(AllConfigs.getInstance().getConfigEntity().getHotkey());
    }
    
    @EventListener(listenClass = RestartEvent.class)
    private static void stopListen(Event event) {
        getInstance().stopListen();
    }
    ```
    
    CheckHotKeyService是通过配置中心AllConfigs发出的配置更新事件来进行响应。CheckHotKeyAvailableEvent事件用于检测快捷键是否有效，用于设置界面保存设置时发出检查快捷键是否有效。

- #### DatabaseService
  
  - 数据库服务，提供数据库的搜索，添加，删除等基本操作。当搜索框发出StartSearchEvent后，数据库将会通过关键字进行搜索，并将结果返回给搜索框进行显示。
  
  - 注册和监听的主要事件：
    
    ```java
    @EventRegister(registerClass = PrepareSearchEvent.class)
    private static void prepareSearchEvent(Event event) {
        if (IsDebug.isDebug()) {
            System.out.println("进行预搜索并添加搜索任务");
        }
        var startWaiting = System.currentTimeMillis();
        final long timeout = 3000;
        var databaseService = getInstance();
        while (databaseService.getStatus() != Constants.Enums.DatabaseStatus.NORMAL) {
            if (System.currentTimeMillis() - startWaiting > timeout) {
                System.err.println("prepareSearch，等待数据库状态超时");
                break;
            }
            Thread.onSpinWait();
        }
    
        // 检查prepareTaskMap中是否有过期任务
        for (var eachTask : prepareTasksMap.entrySet()) {
            if (System.currentTimeMillis() - eachTask.getValue().taskCreateTimeMills > SearchTask.maxTaskValidThreshold) {
                prepareTasksMap.remove(eachTask.getKey());
            }
        }
    
        var prepareSearchEvent = (PrepareSearchEvent) event;
        var searchInfo = prepareSearchKeywords(prepareSearchEvent.searchText, prepareSearchEvent.searchCase, prepareSearchEvent.keywords);
        var searchTask = prepareSearch(searchInfo);
        prepareTasksMap.put(searchInfo, searchTask);
        event.setReturnValue(searchTask);
    }
    
    @EventRegister(registerClass = StartSearchEvent.class)
    private static void startSearchEvent(Event event) {
        if (((StartSearchEvent) event).searchText.get().length() > Constants.MAX_SEARCH_TEXT_LENGTH) {
            System.err.println("关键字太长，取消搜索");
            return;
        }
        DatabaseService databaseService = getInstance();
        final long startWaiting = System.currentTimeMillis();
        final long timeout = 3000;
        while (databaseService.getStatus() != Constants.Enums.DatabaseStatus.NORMAL) {
            if (System.currentTimeMillis() - startWaiting > timeout) {
                System.out.println("等待数据库状态为NORMAL超时");
                return;
            }
            Thread.onSpinWait();
        }
    
        var startSearchEvent = (StartSearchEvent) event;
        var searchInfo = prepareSearchKeywords(startSearchEvent.searchText, startSearchEvent.searchCase, startSearchEvent.keywords);
        var searchTask = prepareTasksMap.get(searchInfo);
        if (searchTask == null) {
            searchTask = prepareSearch(searchInfo);
        }
        databaseService.startSearchInThreadPool(searchTask);
        event.setReturnValue(searchTask);
    }
    
    @EventRegister(registerClass = InitializeDatabaseEvent.class)
    private static void initAllDatabases(Event event) {
        SQLiteUtil.initAllConnections();
    }
    
    @EventRegister(registerClass = InitializeDatabaseEvent.class)
    private static void initAllDatabases(Event event) {
        SQLiteUtil.initAllConnections();
    }
    
    @EventRegister(registerClass = StartMonitorDiskEvent.class)
    private static void startMonitorDiskEvent(Event event) {
        startMonitorDisks();
    }
    
    @EventListener(listenClass = SetConfigsEvent.class)
    private static void setGpuDevice(Event event) {
        isEnableGPUAccelerate = AllConfigs.getInstance().getConfigEntity().isEnableGpuAccelerate();
        if (isEnableGPUAccelerate) {
            synchronized (DatabaseService.class) {
                var device = AllConfigs.getInstance().getConfigEntity().getGpuDevice();
                if (!GPUAccelerator.INSTANCE.setDevice(device)) {
                    System.err.println("gpu设备" + device + "无效");
                }
            }
        }
    }
    ```
    
    核心事件为PrepareSearchEvent和StartSearchEvent，用于启动搜索。PrepareSearchEvent用于预搜索，提前于StartSearchEvent发出，用于提前搜索桌面，开始菜单快捷方式，以及GPU加速启用时的搜索。
    
    StartMonitorDiskEvent事件发出后将会开启文件监控，记录文件的新增和删除的变化，并同步到数据库以及缓存中。
    
    InitializeDatabaseEvent事件发出后将会初始化数据库，打开所有的数据库连接并创建需要的表。

- ### utils
  
  - 基本工具类
  
  - #### Bit
    
    - 大数位运算模块，用于运算超过long位数的位运算。由于File-Engine使用异步搜索来进行数据库的查询，通过多个表以及优先级的任务划分后会产生几百个小任务，所以通过该类来进行任务的完成标记。
  
  - #### clazz.scan
    
    - ClassScannerUtil：注解扫描工具，扫描带有@EventRegister和@EventListener注解的方法，并进行注册。
  
  - #### connection
    
    - PreparedStatementWrapper
      
      - 用于包裹PreparedStatement，通过继承JDBC4PreparedStatement并重写AutoCloseable的close方法，实现引用计数的功能，File-Engine拥有闲时自动关闭数据库的功能。通过引用计数来实现对数据库的使用进行监控，防止外部还有数据库使用时数据库被关闭导致崩溃。
        
        ```java
        /**
         * 通过复写AutoCloseable接口的close方法实现引用计数，确保在关闭数据库时没有被使用
         * 必须使用 try-with-source语法
         */
        class PreparedStatementWrapper extends JDBC4PreparedStatement {
            private final AtomicInteger connectionUsingCounter;
        
            public PreparedStatementWrapper(SQLiteConnection conn, String sql, AtomicInteger connectionUsingCounter) throws SQLException {
                super(conn, sql);
                this.connectionUsingCounter = connectionUsingCounter;
                this.connectionUsingCounter.incrementAndGet();
            }
        
            @Override
            public void close() throws SQLException {
                super.close();
                connectionUsingCounter.decrementAndGet();
            }
        }
        ```
    
    - StatementWrapper
      
      - 用于包裹Statement，通过继承JDBC4Statement并重写AutoCloseable的close方法实现引用计数的功能。
        
        ```java
        /**
         * 通过复写AutoCloseable接口的close方法实现引用计数，确保在关闭数据库时没有被使用
         * 必须使用 try-with-source语法
         */
        class StatementWrapper extends JDBC4Statement {
            private final AtomicInteger connectionUsingCounter;
        
            public StatementWrapper(SQLiteConnection conn, AtomicInteger connectionUsingCounter) {
                super(conn);
                this.connectionUsingCounter = connectionUsingCounter;
                this.connectionUsingCounter.incrementAndGet();
            }
        
            @Override
            public void close() throws SQLException {
                super.close();
                connectionUsingCounter.decrementAndGet();
            }
        }
        ```
    
    - SQLiteUtil
      
      - sqlite数据库的管理工具，实现数据库的打开关闭基本功能，以及闲时关闭数据库的功能。
      
      - 对外提供的函数：
        
        ```java
        /**
         * 打开所有连接
         */
        public static void openAllConnection() {
            ...
        }
        
        /**
         * 不要用于大量数据的select查询，否则可能会占用大量内存
         *
         * @param sql select语句
         * @return 已编译的PreparedStatement
         * @throws SQLException 失败
         */
        public static PreparedStatement getPreparedStatement(String sql, String key) throws SQLException {
            if (isConnectionNotInitialized(key)) {
                var root = key + ":\\";
                if (FileUtil.isFileNotExist(root) || !IsLocalDisk.INSTANCE.isDiskNTFS(root)) {
                    throw new SQLException(root + " disk is invalid.");
                } else {
                    File data = new File(currentDatabaseDir, key + ".db");
                    initConnection("jdbc:sqlite:" + data.getAbsolutePath(), key);
                }
            }
            ConnectionWrapper connectionWrapper = getFromConnectionPool(key);
            return new PreparedStatementWrapper((SQLiteConnection) connectionWrapper.connection, sql, connectionWrapper.connectionUsingCounter);
        }
        
        /**
         * 用于需要重复运行多次指令的地方
         *
         * @return Statement
         * @throws SQLException 失败
         */
        public static Statement getStatement(String key) throws SQLException {
            if (isConnectionNotInitialized(key)) {
                var root = key + ":\\";
                if (FileUtil.isFileNotExist(root) || !IsLocalDisk.INSTANCE.isDiskNTFS(root)) {
                    throw new SQLException(root + " disk is invalid.");
                } else {
                    File data = new File(currentDatabaseDir, key + ".db");
                    initConnection("jdbc:sqlite:" + data.getAbsolutePath(), key);
                }
            }
            ConnectionWrapper wrapper = getFromConnectionPool(key);
            return new StatementWrapper((SQLiteConnection) wrapper.connection, wrapper.connectionUsingCounter);
        }
        
        public static void initAllConnections() {
            initAllConnections("data");
        }
        
        public static void initAllConnections(String dir) {
            ...
        }
        ```
        
        SQLite对外提供Statement和PreparedStatement，DatabaseService使用SQLite进行sql操作。

- #### file
  
  - FileUtil：文件处理工具类，清空文件夹，获取上级目录，判断是否为文件等基础功能。
  
  - MoveDesktopFiles：移动桌面文件到File-Engine的Files文件夹下。

- #### gson
  
  - GsonUtil：google json处理工具类。

- #### system.properties
  
  - IsDebug
    
    - 判断File-Engine是否处于debug模式下，当jvm启动参数中包含 **-DFile_Engine_Debug=true** 时返回true。当File-Engine处于debug模式下会输出很多调试信息。
  
  - IsPreview
    
    - 判断File-Engine是否处于preview模式下，当处于preview模式时，将会忽略版本信息，始终判断稳定版为最新版本，在发布不稳定的新特性版本时使用。

- #### CachedThreadPoolUtil
  
  - 缓存线程池工具类，用于启动和管理线程。目前线程池拥有两个分支，主分支master中只有一个线程池platformThreadPool，virtual-thread-feature分支实现了虚拟线程池，未来可能会将virtual-thread-feature分支合并到主分支。

- #### ColorUtil
  
  - 颜色工具类，判断字符串hex值是否能转换到RGB颜色，以及获取高亮颜色，高对比度颜色，判断颜色是亮色还是暗色，颜色和字符串转换功能。

- #### DpiUtil
  
  - 获取windows系统的缩放级别（dpi）。

- #### GetIconUtil
  
  - 获取图标工具类，当搜索框显示文件时，通过GetIconUtil获取文件的图标，并显示在结果左方。

- #### Md5Util
  
  - 获取文件的MD5值，用于File-Engine更新资源和依赖，当版本更新后将会通过比对user/文件夹中现有的依赖文件和File-Engine中保存的依赖文件来更新资源。

- #### OpenFileUtil
  
  - 打开文件工具类，当在搜索框上点击Enter或者双击鼠标左键后将会调用OpenFileUtil中的方法来打开文件。

- #### PathMatchUtil
  
  - 文件名关键字匹配工具类，由databaseService进行调用。

- #### PinyinUtil
  
  - File-Engine支持拼音搜索，通过将文字转换为拼音来进行搜索。

- #### ProcessUtil
  
  - 进程控制工具类，判断进程是否存在，等待进程。

- #### RegexUtil
  
  - 正则表达式工具类，获取并缓存正则表达式。

- #### RobotUtil
  
  - 鼠标键盘控制工具类。

- #### StartupUtil
  
  - 开机启动工具类，用于判断开机启动是否生效，添加和删除开机启动。

- #### SystemInfoUtil
  
  - 系统信息工具类，用于获取系统的内存信息，判断系统内存用量。

## C++库部分

- ### fileMonitor
  
  - 监控文件的变化，文件删除或是增加会被记录，然后添加进数据库。

- ### fileSearcherUSN
  
  - 搜索磁盘上的文件，创建索引。 搜索时同时会创建共享内存，如果硬盘速度太慢导致存储时间太长，~~可以通过resultPipe先读取共享内存，同时等待数据库保存完成~~，共享内存功能已弃用，新版本将会在创建索引时切换到临时数据库，继续提供搜索功能。

- ### getAscII（已弃用）
  
  - **新版本通过Java实现，不再使用JNI调用该原生库。**
    
    ```java
    public class StringUtf8SumUtil {
    
        public static int getStringSum(String fileName) {
            if (fileName == null || fileName.isEmpty()) {
                return 0;
            }
            var bytes = fileName.getBytes(StandardCharsets.UTF_8);
            int sum = 0;
            for (byte aByte : bytes) {
                if (aByte > 0) {
                    sum += aByte;
                }
            }
            return sum;
        }
    }
    ```
  
  - 获取文件名的每个字符，将char值相加并返回。
  
  - File-Engine的数据库通过41个表保存文件数据。分别是list0-list40，list0保存文件名ascii和在0-100范围内的文件。list1保存100-200的范围内的文件，以此类推。

- ### getDpi
  
  - 获取windows系统的缩放级别，适配系统的分辨率和DPI，使程序在高分辨率屏幕下显示不模糊和错位。

- ### getHandle
  
  - 实现与explorer.exe相关的操作，贴靠在explorer.exe下方，让explorer跳转到其他路径，File-Engine普通模式与贴靠模式的切换，搜索框拖动到文件夹以创建快捷方式，以及检测窗口是否全屏等与Windows 交互功能的实现。

- ### getWindowsKnownFolder
  
  - 获取Windows默认文件夹的路径，如开始菜单，因为大部分程序都在开始菜单，因此作为优先搜索的路径。该函数为下方Windows API接口的封装。
  - [SHGetKnownFolderPath function (shlobj_core.h) - Win32 apps | Microsoft Learn](https://learn.microsoft.com/en-us/windows/win32/api/shlobj_core/nf-shlobj_core-shgetknownfolderpath)

- ### hotkeyListener
  
  - 键盘快捷键监听，实现点击键盘快捷键后打开搜索框，以及对Ctrl双击，Shift双击的全局检测。

- ### isLocalDisk
  
  - 检查磁盘是不是本地磁盘或者U盘，以及检测文件系统是否为NTFS。

- ### launcherWrap
  
  - File-Engine的启动器以及守护进程。

- ### resultPipe（已弃用）
  
  - **新版本采用临时数据库的方式实现索引时提供搜索服务，不再采用共享内存。**
  
  - 读取fileSearcherUSN创建的共享内存，实现在创建索引时也能进行搜索。
  
  - 共享内存将会在索引创建完成后关闭。

- ### cudaAccelerator
  
  - NVIDIA GPU CUDA加速引擎，通过并行计算实现高速搜索字符串。

- ### openclAccelerator
  
  - 基于opencl框架实现的通用GPU加速引擎。

- ### sqliteJDBC
  
  - sqlite的jni接口，实现java调用sqlite.dll。

## MainClass

主启动类，对系统进行初始化，释放资源，设置系统属性，发出系统启动事件BootSystemEvent
