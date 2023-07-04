<h1 align="center">File-Engine</h1>

<div align="center">
  <strong>A tool can make you work more efficiently.</strong><br>
  <sub>不必再在杂乱的文件夹中寻找文件。提高工作效率。这是一个绿色软件，不必安装，随时使用。</sub>
</div>
<br>
<div align="center">
  <img src="https://img.shields.io/badge/license-MIT-yellow"/>
  <img src="https://img.shields.io/badge/language-c++-brightgreen"/>
  <img src="https://img.shields.io/badge/language-java-brightgreen" />
  <img src="https://img.shields.io/badge/language-cuda-brightgreen"/>
  <img src="https://img.shields.io/badge/documentation-yes-brightgreen"/>
</div>

## 🏠如何编译构建本项目

---

- JDK >= 17
- Python >= 3.7
- Visual Studio 2022（C++ 生成工具 >= v143）
- maven >= 3.6.1
- [如何编译构建项目](https://github.com/XUANXUQAQ/File-Engine/blob/master/README_BUILD.md)

---

## [🧱项目结构](https://github.com/XUANXUQAQ/File-Engine/blob/master/README_PROJECT.md)

---

## 🛠️制作自己的插件

- [File-Engine-Plugin-Template](https://github.com/XUANXUQAQ/File-Engine-Plugin-Template)   
  
  ### 目前已经可以使用的插件：

- [File-Engine-UDisk-Plugin](https://github.com/XUANXUQAQ/File-Engine-UDisk-Plugin) 使File-Engine支持搜索U盘

- [File-Engine-UWP-Plugin](https://github.com/XUANXUQAQ/File-Engine-UWP-Plugin) 使File-Engine支持搜索UWP应用程序

- [File-Engine-Remote-Plugin](https://github.com/XUANXUQAQ/File-Engine-Remote-Plugin) 通过远程搜索本机软件并下载

- [File-Engine-Cmd-Plugin](https://github.com/XUANXUQAQ/File-Engine-Cmd-Plugin) 使用搜索框快速输入cmd命令并执行

- [File-Engine-Translate-Plugin](https://github.com/XUANXUQAQ/File-Engine-Translate-Plugin) 在输入框中输入单词并进行翻译

- [File-Engine-NoUAC-Plugin](https://github.com/XUANXUQAQ/File-Engine-NoUAC-Plugin) 利用File-Engine的管理员权限实现开机启动而不出现UAC弹窗

- [File-Engine-Web-Plugin](https://github.com/XUANXUQAQ/File-Engine-Web-Plugin) 将搜索框作为网络搜索框进行搜索，支持百度，Google以及自定义搜索url

---

## 如何手动更新

- 进入File-Engine目录，用新版本覆盖File-Engine.exe，然后打开File-Engine.exe即可。

- 4.0以前版本还需删除data目录下的File-Engine.jar

---

## 使用

- ### [查看帮助](https://github.com/XUANXUQAQ/File-Engine/wiki/Usage)

- 或者在搜索框中输入 :help    默认Ctrl + Alt + K呼出搜索框

---

## 特性

* 快速搜索，***实测100W文件索引仅需要10秒。搜索速度取决于你的硬盘的速度***

* 绿色便携软件，所有配置文件均在自身目录。

* 自定义命令，快速进行所选操作。

* 实时更新文件，自动检测已删除和新增文件。

* 支持指定优先搜索文件夹，常用文件无需等待，立即打开。

* 简洁UI，设置方便。

* 自适应分辨率显示。    

* 多语言支持。   
  
  ### 该软件仅支持64位系统

---

## 💖感谢以下项目：

1. gson (https://github.com/google/gson)
2. FlatLaf (https://github.com/JFormDesigner/FlatLaf)   
3. sqlite_jdbc (https://github.com/xerial/sqlite-jdbc)   
4. lombok (https://projectlombok.org/)   
5. TinyPinyin (https://github.com/promeG/TinyPinyin)
6. zip (https://github.com/kuba--/zip)
7. OpenCLWrapper (https://github.com/ProjectPhysX/OpenCL-Wrapper)
8. oshi (https://github.com/oshi/oshi)
9. jna (https://github.com/java-native-access/jna)