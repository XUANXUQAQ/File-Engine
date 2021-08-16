module File.Engine {
    requires java.desktop;
    requires java.sql;
    requires TinyPinyin;
    requires com.formdev.flatlaf;
    requires com.formdev.flatlaf.intellijthemes;
    requires com.sun.jna;
    requires org.xerial.sqlitejdbc;
    requires fastjson;

    requires static lombok;

    exports file.engine.configs;
    exports file.engine.frames;
    exports file.engine.services;
    exports file.engine.services.plugin.system;
    exports file.engine.services.download;
    exports file.engine.utils;
    exports file.engine.utils.file;
    exports file.engine.utils.bit;
    exports file.engine.utils.clazz.scan;
    exports file.engine.utils.system.properties;

    opens file.engine.configs;
    opens file.engine.frames;
    opens file.engine.services;
    opens file.engine.services.plugin.system;
    opens file.engine.services.download;
    opens file.engine.utils;
    opens file.engine.utils.file;
    opens file.engine.utils.bit;
    opens file.engine.utils.clazz.scan;
    opens file.engine.utils.system.properties;
}