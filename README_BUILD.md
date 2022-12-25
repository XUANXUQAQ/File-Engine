# 如何编译构建本项目

## 该项目仅支持windows 7以上系统。

### 1. 将本项目源码下载到本地。

```bash
git clone https://github.com/XUANXUQAQ/File-Engine.git
```

本项目需要jdk11及以上jdk，以及visual studio安装才能编译。

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

进入项目根目录，打开cmd或者powershell，直接使用python运行build.py即可，后一个参数可以指定jdk目录位置。

![Dd7jp.jpeg](https://i.328888.xyz/2022/12/25/Dd7jp.jpeg)

或指定使用的jdk位置，在后面增加参数即可。

![DdEFU.jpeg](https://i.328888.xyz/2022/12/25/DdEFU.jpeg)

# 以下内容已过时

通过maven进行编译构建

```bash
mvn clean compile package
```

编译完成后在target目录下会生成 **File-Engine-(版本).jar** 以及**File-Engine.jar**。

其中File-Engine-(版本).jar是不包含依赖的jar包，而File-Engine.jar是包含所有依赖的jar包。

![](https://p0.meituan.net/dpplatform/1d87be4c66fc8882ccf742a8b7022fb924871.png)

### 3. 构建启动器

### (1) 生成jre

首先使用jdeps分析刚才maven生成的无依赖的jar包（File-Engine-(版本).jar)

```bash
jdeps --ignore-missing-deps --list-deps .\File-Engine-${version}.jar
```

![](https://p0.meituan.net/dpplatform/3d243f78a9c3f7536e4bcea8377e1a526459.png)

分析完成后再使用jlink工具生成精简后的jre运行环境。

```bash
jlink --no-header-files --no-man-pages --compress=2 --module-path jmods --add-modules java.base,java.datatransfer,java.desktop,java.sql --output jre
```

然后在根目录下创建build文件夹，将生成的jre放入build文件夹中。

![DzFAL.jpeg](https://i.328888.xyz/2022/12/25/DzFAL.jpeg)

### (2) 创建File-Engine.zip

jlink生成完成之后，将生成的jre运行环境文件夹重命名为jre，和File-Engine.jar(带有依赖的jar包)一起压缩成File-Engine.zip然后放入launcherWrap源码根目录。

使用Visual Studio打开**C++/launcherWrap/launcherWrap.sln**，进入launcherWrap文件夹源码根目录。创建**File-Engine.zip**文件。

![Dd5cZ.png](https://i.328888.xyz/2022/12/25/Dd5cZ.png)

File-Engine.zip压缩包中需要放入**jre运行环境**和刚才maven生成的**File-Engine.jar**

![](https://p1.meituan.net/dpplatform/6b3c8049ab49dac3a18560ebefd9275546273.png)

### (3) 更新MD5

启动器通过检查File-Engine.jar的MD5值来更新资源，编译时需要计算**File-Engine.jar**的MD5值，并找到**launcherWrap.cpp**更新MD5

进入target文件夹

```batch
certutil -hashfile File-Engine.jar md5
```

![DdsfH.png](https://i.328888.xyz/2022/12/25/DdsfH.png)

[![xbzxVP.jpg](https://s1.ax1x.com/2022/11/03/xbzxVP.jpg)](https://imgse.com/i/xbzxVP)

然后编译launcherWrap项目即可。使用release模式编译生成launcherWrap.exe

重命名为**File-Engine.exe**即可。
