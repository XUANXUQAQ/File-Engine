# 项目结构

## 本项目架构总体分为两层。

上层是UI层，放在frames文件夹下，下层是服务层，放在services文件夹下。

由服务层提供基本功能，UI层响应用户的操作并进行调用。



本项目拥有一个事件处理系统。在event/handler文件夹下。

事件的基类为Event。下方为Event拥有的public方法以及作用

### isFinished

```java
    public boolean isFinished() {
        return isFinished.get();
    }
```

返回任务是否完成。

### setBlock

```java
    public void setBlock() {
        isBlock.set(true);
    }
```

设置事件为阻塞事件，同样被设置为阻塞的事件会通过单线程阻塞执行。如果有多个事件，各个事件需要互相等待，但是整体相对于其他事件可以异步，则可以将这几个事件设置为阻塞事件，根据事件传入的顺序进行执行。一般不需要调用。

### isBlock

```java
    public boolean isBlock() {
        return isBlock.get();
    }
```

返回事件是否为阻塞事件。默认事件为乱序异步执行。

### isFailed

```java
    public boolean isFailed() {
        return isFailed.get();
    }
```

返回事件是否失败。

### getReturnValue

```java
    public <T> Optional<T> getReturnValue() {
        return Optional.ofNullable((T) returnValue);
    }
```

获得事件执行完成的返回值，事件被响应后可以设置返回值传给发送事件方。

### setCallback

```java
    public void setCallback(Consumer<Event> callback) {
        this.callback = callback;
    }
```

设置事件执行完成后执行的回调方法。回调方法参数为事件对象。

### setErrorHandler

```java
    public void setErrorHandler(Consumer<Event> errorHandler) {
        this.errorHandler = errorHandler;
    }
```

设置事件执行失败的错误处理方法，回调方法参数为事件对象。

### setMaxRetryTimes

```java
    public void setMaxRetryTimes(int maxRetryTimes) {
        this.maxRetryTimes = maxRetryTimes;
    }
```

设置事件最大重试次数，默认最大尝试次数为5，一般不需要修改。

### 事件处理器通过EventManagement进行管理。

EventManagement中可调用的方法如下。

### waitForEvent

```java
     /**
     * 等待任务
     *
     * @param event 任务实例
     * @return true如果任务执行失败， false如果执行正常完成
     */
    public boolean waitForEvent(Event event);
```

等待任务执行完成或者执行失败，该方法将会阻塞当前线程直到事件处理完成。超时时间20秒。

### putEvent

该方法有两个重载

```java
    /**
     * 发送任务
     *
     * @param event 任务
     */
    public void putEvent(Event event);
    
     /**
     * 异步回调方法发送任务
     *
     * @param event        任务
     * @param callback     回调函数
     * @param errorHandler 错误处理
     */
    public void putEvent(Event event, 
                         Consumer<Event> callback, 
                         Consumer<Event> errorHandler) {
        event.setCallback(callback);
        event.setErrorHandler(errorHandler);
        putEvent(event);
    }
```

向事件处理中心发送任务。任务会被送到处理队列，然后找到对应的事件处理器进行处理。

### notMainExit

```java
    public boolean notMainExit() {
        return !exit.get();
    }
```

程序退出标志，可以用做进行死循环判断退出的标志。

### registerAllHandler & registerAllListener

```java
    /**
     * 注册所有事件处理器
     */
    public void registerAllHandler();
    
    /**
     * 注册所有时间监听器
     */
    public void registerAllListener()
```

注册事件处理器。File-Engine中有两种对事件的处理方式。一种是事件处理器，一种是事件监听器。事件处理器一个事件只能有一个，事件监听器一个事件可以有多个。

事件处理器通过annotation文件夹中的 **@EventRegister** 和 **@EventListener** 来指定。

```java
/**
 * 注册事件处理器
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface EventListener {
    Class<? extends Event>[] listenClass();
}
```

```java
/**
 * 注册事件处理器，该注解可以保证方法被第一个执行，且一个事件只能有一个Register，可以有多个Listener
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface EventRegister {
    Class<? extends Event> registerClass();
}
```

**需要注意的是**

注册事件处理器和监听器的方法必须是static方法，且只能有一个参数Event

```java
    @EventRegister(registerClass = SomeEvent.class)
    private static void someEventHandler(Event event) {
        
    }
    
    @EventListener(listenClass = SomeEvent.class)
    private static void someEventListener1(Event event) {
        
    }
    
    @EventListener(listenClass = SomeEvent.class)
    private static void someEventListener2(Event event) {
        
    }
```

UI层和service层都可以通过事件的注册和处理来实现互相调用。各层之间也可以通过事件来进行调用。



下面是各个包以及各个依赖的作用。

## C++库部分

│ ├─fileMonitor   
│ ├─fileSearcherUSN   
│ ├─getAscII   
│ ├─getDpi   
│ ├─getHandle   
│ ├─getStartMenu   
│ ├─hotkeyListener   
│ ├─isLocalDisk   
│ ├─launcherWrap   
│ ├─resultPipe   
│ └─sqliteJDBC   

## fileMonitor

监控文件的变化，文件删除或是增加会被记录，然后添加进数据库。

## fileSearcherUSN

搜索磁盘上的文件，创建索引。 搜索时同时会创建共享内存，如果硬盘速度太慢导致存储时间太长，可以通过resultPipe先读取共享内存，同时等待数据库保存完成。

## getAscII

获取文件名的每个字符，将char值相加并返回。

File-Engine的数据库通过41个表保存文件数据。分别是list0-list40，list0保存文件名ascii和在0-100范围内的文件。list1保存100-200的范围内的文件，以此类推。

## getDpi

获取windows系统的缩放级别，适配系统的分辨率和DPI，使程序在高分辨率屏幕下显示不模糊和错位。

## getHandle

实现与explorer.exe相关的操作，贴靠在explorer.exe下方，让explorer跳转到其他路径，File-Engine普通模式与贴靠模式的切换。

## getStartMenu

获取开始菜单的路径，因为大部分程序都在开始菜单，因此作为优先搜索的路径。

## hotkeyListener

键盘快捷键监听，实现点击键盘快捷键后打开搜索框。

## isLocalDisk

检查磁盘是不是本地磁盘或者U盘，以及检测文件系统是否为NTFS。

## launcherWrap

File-Engine的启动器以及守护进程。

## resultPipe

读取fileSearcherUSN创建的共享内存，实现在创建索引时也能进行搜索。

共享内存将会在索引创建完成后关闭。

## sqliteJDBC

sqlite的jni接口，实现java调用sqlite.dll。



## Java包部分

### annotation

1. EventRegister，事件处理器注册注解，详细使用方法请见上方事件处理部分。

2. EventListener，事件监听器注册注解，详细使用方法请见上方事件处理部分。

### configs

配置保存和加载中心，保存FIle-Engine所有的配置信息，以及对配置相关事件的处理。

AllConfigs：读取和处理所有的配置信息。

ConfigEntity：配置信息实体类。

Constants：File-Engine使用的常量信息。

### dllInterface

jni实现调用上方C++库的接口。

### Event & EventManagement

事件处理中心，详细信息见上方。

### frames

UI层的实现。包含搜索框，设置窗口，任务栏，插件市场窗口的实现。以及一些通用控件。

UI层使用Java Swing实现，以及Intellij idea中的GUI Designer来构建。

SearchBar：搜索框的具体实现。

TaskBar：任务栏的具体实现。

PluginMarket：插件市场窗口的具体实现。

SettingsFrame：设置窗口的具体实现。

components：

1. LoadingPanel：通用组件，加载窗口的实现。

2. MouseDragInfo：通用组件，鼠标从搜索框拖动到资源管理器时显示的窗口。

3. RoundBorder：通用组件，圆角边框的具体实现。

### service

服务层的实现。包含下载服务，插件加载服务，键盘快捷键监听服务，数据库服务，多语言UI翻译服务。

DownloadService：下载服务的具体实现。

DownloadManager：下载文件信息的封装。作为发送StartDownloadEvent事件的参数。

PluginService：插件服务，为项目提供插件加载卸载以及基本调用的接口。获取插件的一些基本信息

Plugin：插件对外暴露的接口。

CheckHotKeyService：键盘快捷键的监听服务，当键盘快捷键点击后将会发出ShowSearchBarEvent打开搜索框。

DatabaseService：数据库服务，提供数据库的搜索，添加，删除等基本操作。当搜索框发出StartSearchEvent后，数据库将会通过关键字进行搜索，并将结果返回给搜索框进行显示。

### utils

基本工具类

#### Bit

大数位运算模块，用于运算超过long位数的位运算。由于File-Engine使用异步搜索来进行数据库的查询，通过多个表以及优先级的任务划分后会产生几百个小任务，所以通过该类来进行任务的完成标记。

#### clazz.scan

ClassScannerUtil：注解扫描工具，扫描带有@EventRegister和@EventListener注解的方法，并进行注册。

#### connection

PreparedStatementWrapper：用于包裹PreparedStatement，通过继承JDBC4PreparedStatement并重写AutoCloseable的close方法，实现引用计数的功能，File-Engine拥有闲时自动关闭数据库的功能。通过引用计数来实现对数据库的使用进行监控，防止外部还有数据库使用时数据库被关闭导致崩溃。

StatementWrapper：用于包裹Statement，通过继承JDBC4Statement并重写AutoCloseable的close方法实现引用计数的功能。

SQLiteUtil：sqlite数据库的管理工具，实现数据库的打开关闭基本功能，以及闲时关闭数据库的功能。

#### file

FileUtil：文件处理工具类，清空文件夹，获取上级目录，判断是否为文件等基础功能。

MoveDesktopFiles：移动桌面文件到File-Engine的Files文件夹下。

#### gson

GsonUtil：google json处理工具类。

#### system.properties

IsDebug：判断File-Engine是否处于debug模式下，当jvm启动参数中包含 **-DFile_Engine_Debug=true** 时返回true。当File-Engine处于debug模式下会输出很多调试信息。

IsPreview：判断File-Engine是否处于preview模式下，当处于preview模式时，将会忽略版本信息，始终判断稳定版为最新版本，在发布不稳定的新特性版本时使用。

#### CachedThreadPoolUtil

缓存线程池工具类，用于启动和管理线程。目前线程池拥有两个分支，主分支master中只有一个线程池platformThreadPool，virtual-thread-feature分支实现了虚拟线程池，未来可能会将virtual-thread-feature分支合并到主分支。

#### ColorUtil

颜色工具类，判断字符串hex值是否能转换到RGB颜色，以及获取高亮颜色，高对比度颜色，判断颜色是亮色还是暗色，颜色和字符串转换功能。

#### DpiUtil

获取windows系统的缩放级别（dpi）。

#### GetIconUtil

获取图标工具类，当搜索框显示文件时，通过GetIconUtil获取文件的图标，并显示在结果左方。

#### Md5Util

获取文件的MD5值，用于File-Engine更新资源和依赖，当版本更新后将会通过比对user/文件夹中现有的依赖文件和File-Engine中保存的依赖文件来更新资源。

#### OpenFileUtil

打开文件工具类，当在搜索框上点击Enter或者双击鼠标左键后将会调用OpenFileUtil中的方法来打开文件。

#### PathMatchUtil

文件名关键字匹配工具类，由databaseService进行调用。

#### PinyinUtil

File-Engine支持拼音搜索，通过将文字转换为拼音来进行搜索。

#### ProcessUtil

进程控制工具类，判断进程是否存在，等待进程。

#### RegexUtil

正则表达式工具类，获取并缓存正则表达式。

#### RobotUtil

鼠标键盘控制工具类。

#### StartupUtil

开机启动工具类，用于判断开机启动是否生效，添加和删除开机启动。

#### SystemInfoUtil

系统信息工具类，用于获取系统的内存信息，判断系统内存用量。



### MainClass

主启动类，对系统进行初始化，释放资源，设置系统属性，发出系统启动事件。


