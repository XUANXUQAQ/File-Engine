package file.engine.annotation;

import file.engine.event.handler.Event;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * 注册事件处理器，该注解可以保证方法被第一个执行，且一个事件只能有一个Register，可以有多个Listener
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface EventRegister {
    Class<? extends Event> registerClass();
}
