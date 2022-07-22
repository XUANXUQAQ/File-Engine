# 如何编译构建本项目

## 该项目仅支持windows 7以上系统。

### 1. 将本项目源码下载到本地。

```bash
git clone https://github.com/XUANXUQAQ/File-Engine.git
```

本项目需要jdk11及以上jdk才能编译。

### 2. 编译项目

项目中的gui界面构建使用了intellij idea的gui designer。依赖放在 **libs/forms_rt.jar** ，需要先使用maven安装到本地仓库。

在项目根目录下运行以下命令安装forms_rt.jar到本地maven仓库。

```bash
mvn install:install-file -Dfile=libs/forms_rt.jar -DgroupId=com.intellij -DartifactId=forms_rt -Dversion=1.0 -Dpackaging=jar
```

通过修改pom.xml中的<version>标签修改版本号。

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    ...
    <groupId>github.fileengine</groupId>
    <artifactId>File-Engine</artifactId>
    <version>3.5</version>
    ...
</project>
```

通过maven进行编译构建

```bash
maven clean compile package
```

编译完成后在target目录下会生成 **File-Engine-(版本).jar** 以及**File-Engine.jar**。

其中File-Engine-(版本).jar是不包含依赖的jar包，而File-Engine.jar是包含所有依赖的jar包。

![](https://i.bmp.ovh/imgs/2022/07/22/dadb33cef92f0657.png)

### 4. 构建启动器

使用Visual Studio打开**C++/launcherWrap**，进入launcherWrap文件夹源码根目录。创建**File-Engine.zip**文件。

File-Engine.zip压缩包中需要放入**jre运行环境**和刚才maven生成的**File-Engine.jar**

![](https://i.bmp.ovh/imgs/2022/07/22/c82f676a3c7c6782.png)

jre运行环境可以由jlink工具进行生成。

首先使用jdeps分析刚才maven生成的无依赖的jar包（File-Engine-(版本).jar)

```bash
jdeps --ignore-missing-deps --list-deps .\File-Engine-3.5.jar
```

![](https://i.bmp.ovh/imgs/2022/07/22/5f28ebdce13ee327.png)

分析完成后再使用jlink工具生成精简后的jre运行环境。

```bash
jlink --no-header-files --no-man-pages --compress=2 --module-path jmods --add-modules java.base,java.datatransfer,java.desktop,java.sql --output 输出文件夹位置
```

jlink生成完成之后，将生成的jre运行环境文件夹重命名为jre，和File-Engine.jar(带有依赖的jar包)一起压缩成File-Engine.zip然后放入launcherWrap源码根目录。

然后编译launcherWrap项目即可。使用release模式编译生成launcherWrap.exe

重命名为**File-Engine.exe**即可。
