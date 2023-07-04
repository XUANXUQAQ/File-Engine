# 事件处理系统

本项目的事件处理系统。在event/handler文件夹下。

事件的基类为Event。下方为Event拥有的public方法以及作用

## Event.java

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





## 事件处理器通过EventManagement进行管理。

EventManagement中可调用的方法如下。

## EventManagement.java

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
     * @param event任务
     * @param callback     回调函数
     * @param errorHandler 错误处理
     */
    public void putEvent(Event event,
                         Consumer<Event> callback, 
                         Consumer<Event> errorHandler)
```

向事件处理中心发送任务。任务会被送到处理队列，然后找到对应的事件处理器进行处理。

### notMainExit

```java
    public boolean notMainExit() {
        return !exit.get();
    }
```

程序全局退出标志，可以用做进行死循环判断退出的标志。

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





## 注册事件处理器和监听器

使用上方的两个注解就可以实现注册事件的处理器和监听器

**需要注意的是**

**注册事件处理器和监听器的方法必须是static方法**，且只能有一个参数Event

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

UI层和service层都可以通过事件的注册和处理来实现互相调用。

各层之间也可以通过事件来进行调用。
