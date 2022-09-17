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
| ├─CudaAccelerator.java            NVIDIA显卡加速工具   
| ├─EmptyRecycleBin.java            清空回收站工具   
| ├─FileMonitor.java                文件改动监测工具   
| ├─GetAscII.java                   获取String UTF8值工具   
| ├─GetHandle.java                  检测windows资源管理器并进行互操作工具   
| ├─GetWindowsKnownFolder.java      获取Windows默认文件夹，如开始菜单，详见MSDN(SHGetKnownFolderPath)
| ├─HotkeyListener.java             Windows全局快捷键注册工具   
| ├─IsLocalDisk.java                检测硬盘是否为本地硬盘以及NTFS文件系统工具   
| └ResultPipe.java                  共享内存读取工具，当创建索引时fileSearcherUSN.exe会先创建共享内存，然后再写入硬盘   
├─configs   
| ├─AllConfigs.java                 全局配置中心   
| ├─ConfigEntity.java               配置对象   
| ├─Constants.java                  全局常量   
| └ProxyInfo.java                   网络代理对象，如http socks5   
├─annotation   
| ├─EventListener.java              事件监听注解，添加该注解可以将函数注册为对应事件的回调函数   
| └EventRegister.java               事件处理注解，添加该注解可以将函数注册为相应事件的处理函数。注意：一个事件只能有一个处理函数，可以有多个回调函数。
```
事件处理系统详见[Event_Management](https://github.com/XUANXUQAQ/File-Engine/blob/master/Event_Mangement.md)

## 下面是各个包以及各个依赖的作用。

## Java包部分

- #### frames
  
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

### service

    服务层的实现。包含下载服务，插件加载服务，键盘快捷键监听服务，数据库服务，多语言UI翻译服务。

- #### DownloadService
  
  - 下载服务的具体实现，下载文件并在下载完成后执行回调。
  
  - DownloadManager：下载文件信息的封装。作为发送StartDownloadEvent事件的参数。

- #### PluginService
  
  - 插件服务，为项目提供插件加载卸载以及基本调用的接口。获取插件的一些基本信息
  
  - Plugin：插件对外暴露的接口。

- #### CheckHotKeyService
  
  - 键盘快捷键的监听服务，当键盘快捷键点击后将会发出ShowSearchBarEvent打开搜索框。

- #### DatabaseService
  
  - 数据库服务，提供数据库的搜索，添加，删除等基本操作。当搜索框发出StartSearchEvent后，数据库将会通过关键字进行搜索，并将结果返回给搜索框进行显示。

### utils

基本工具类

- ### Bit
  
  - 大数位运算模块，用于运算超过long位数的位运算。由于File-Engine使用异步搜索来进行数据库的查询，通过多个表以及优先级的任务划分后会产生几百个小任务，所以通过该类来进行任务的完成标记。

- ### clazz.scan
  
  - ClassScannerUtil：注解扫描工具，扫描带有@EventRegister和@EventListener注解的方法，并进行注册。

- ### connection

- PreparedStatementWrapper
  
  - 用于包裹PreparedStatement，通过继承JDBC4PreparedStatement并重写AutoCloseable的close方法，实现引用计数的功能，File-Engine拥有闲时自动关闭数据库的功能。通过引用计数来实现对数据库的使用进行监控，防止外部还有数据库使用时数据库被关闭导致崩溃。

- StatementWrapper
  
  - 用于包裹Statement，通过继承JDBC4Statement并重写AutoCloseable的close方法实现引用计数的功能。

- SQLiteUtil
  
  - sqlite数据库的管理工具，实现数据库的打开关闭基本功能，以及闲时关闭数据库的功能。

- ### file
  
  - FileUtil：文件处理工具类，清空文件夹，获取上级目录，判断是否为文件等基础功能。
  
  - MoveDesktopFiles：移动桌面文件到File-Engine的Files文件夹下。

- ### gson
  
  - GsonUtil：google json处理工具类。

- ### system.properties

- IsDebug
  
  - 判断File-Engine是否处于debug模式下，当jvm启动参数中包含 **-DFile_Engine_Debug=true** 时返回true。当File-Engine处于debug模式下会输出很多调试信息。

- IsPreview
  
  - 判断File-Engine是否处于preview模式下，当处于preview模式时，将会忽略版本信息，始终判断稳定版为最新版本，在发布不稳定的新特性版本时使用。

- ### CachedThreadPoolUtil
  
  - 缓存线程池工具类，用于启动和管理线程。目前线程池拥有两个分支，主分支master中只有一个线程池platformThreadPool，virtual-thread-feature分支实现了虚拟线程池，未来可能会将virtual-thread-feature分支合并到主分支。

- ### ColorUtil
  
  - 颜色工具类，判断字符串hex值是否能转换到RGB颜色，以及获取高亮颜色，高对比度颜色，判断颜色是亮色还是暗色，颜色和字符串转换功能。

- ### DpiUtil
  
  - 获取windows系统的缩放级别（dpi）。

- ### GetIconUtil
  
  - 获取图标工具类，当搜索框显示文件时，通过GetIconUtil获取文件的图标，并显示在结果左方。

- ### Md5Util
  
  - 获取文件的MD5值，用于File-Engine更新资源和依赖，当版本更新后将会通过比对user/文件夹中现有的依赖文件和File-Engine中保存的依赖文件来更新资源。

- ### OpenFileUtil
  
  - 打开文件工具类，当在搜索框上点击Enter或者双击鼠标左键后将会调用OpenFileUtil中的方法来打开文件。

- ### PathMatchUtil
  
  - 文件名关键字匹配工具类，由databaseService进行调用。

- ### PinyinUtil
  
  - File-Engine支持拼音搜索，通过将文字转换为拼音来进行搜索。

- ### ProcessUtil
  
  - 进程控制工具类，判断进程是否存在，等待进程。

- ### RegexUtil
  
  - 正则表达式工具类，获取并缓存正则表达式。

- ### RobotUtil
  
  - 鼠标键盘控制工具类。

- ### StartupUtil
  
  - 开机启动工具类，用于判断开机启动是否生效，添加和删除开机启动。

- ### SystemInfoUtil
  
  - 系统信息工具类，用于获取系统的内存信息，判断系统内存用量。

## C++库部分

│ ├─fileMonitor  
│ ├─fileSearcherUSN  
│ ├─getAscII  
│ ├─getDpi  
│ ├─getHandle  
│ ├─getWindowsKnownFolder  
│ ├─hotkeyListener  
│ ├─isLocalDisk  
│ ├─launcherWrap  
│ ├─resultPipe  
│ ├─cudaAccelerator  
│ └─sqliteJDBC

- ### fileMonitor
  
  - 监控文件的变化，文件删除或是增加会被记录，然后添加进数据库。

- ### fileSearcherUSN
  
  - 搜索磁盘上的文件，创建索引。 搜索时同时会创建共享内存，如果硬盘速度太慢导致存储时间太长，可以通过resultPipe先读取共享内存，同时等待数据库保存完成。

- ### getAscII
  
  - 获取文件名的每个字符，将char值相加并返回。
  
  - File-Engine的数据库通过41个表保存文件数据。分别是list0-list40，list0保存文件名ascii和在0-100范围内的文件。list1保存100-200的范围内的文件，以此类推。

- ### getDpi
  
  - 获取windows系统的缩放级别，适配系统的分辨率和DPI，使程序在高分辨率屏幕下显示不模糊和错位。

- ### getHandle
  
  - 实现与explorer.exe相关的操作，贴靠在explorer.exe下方，让explorer跳转到其他路径，File-Engine普通模式与贴靠模式的切换。

- ### getWindowsKnownFolder
  
  - 获取Windows默认文件夹的路径，如开始菜单，因为大部分程序都在开始菜单，因此作为优先搜索的路径。

- ### hotkeyListener
  
  - 键盘快捷键监听，实现点击键盘快捷键后打开搜索框。

- ### isLocalDisk
  
  - 检查磁盘是不是本地磁盘或者U盘，以及检测文件系统是否为NTFS。

- ### launcherWrap
  
  - File-Engine的启动器以及守护进程。

- ### resultPipe
  
  - 读取fileSearcherUSN创建的共享内存，实现在创建索引时也能进行搜索。
  
  - 共享内存将会在索引创建完成后关闭。

- ### cudaAccelerator
  
  - NVIDIA GPU CUDA加速引擎，通过并行计算实现高速搜索字符串。

- ### sqliteJDBC
  
  - sqlite的jni接口，实现java调用sqlite.dll。

## MainClass

主启动类，对系统进行初始化，释放资源，设置系统属性，发出系统启动事件
