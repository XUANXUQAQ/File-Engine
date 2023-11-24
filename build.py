import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
import vswhere
import jproperties


rulesToDelete = [
    r'com\sun\jna\aix-ppc',
    r'com\sun\jna\darwin',
    r'com\sun\jna\freebsd',
    r'com\sun\jna\linux',
    r'com\sun\jna\openbsd',
    r'com\sun\jna\sunos',
    r'com\sun\jna\win32-aarch64',
    r'com\sun\jna\win32-x86',
    r'com\sun\jna\platform\linux',
    r'com\sun\jna\platform\mac',
    r'com\sun\jna\platform\unix',
    r'org\sqlite\native',
    r'oshi\driver\linux',
    r'oshi\driver\mac',
    r'oshi\driver\unix',
    r'oshi\hardware\platform\linux',
    r'oshi\hardware\platform\mac',
    r'oshi\hardware\platform\unix',
    r'oshi\jna\platform\linux',
    r'oshi\jna\platform\mac',
    r'oshi\jna\platform\unix',
    r'oshi\software\os\linux',
    r'oshi\software\os\mac',
    r'oshi\software\os\unix',
    r'oshi\util\platform\linux',
    r'oshi\util\platform\mac',
    r'oshi\util\platform\unix'
]

rulesToSave = [
    r'com\sun\jna\win32-x86-64',
]


def unzipFile(zip_src, dst_dir):  # 解压函数，将zip_src解压到dst_dir
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip......')


def delFileInZip():
    print("UnZip:" + 'File-Engine.jar')
    pathName = 'File-Engine'
    unzipFile('File-Engine.jar', pathName)
    for root, _, files in os.walk(pathName):  # 遍历pathName文件夹
        for f in files:
            nextFile = False
            fileName = os.path.join(root, f)
            for deleteRule in rulesToDelete:
                deletePrefix = os.path.join(pathName, deleteRule)
                if fileName.startswith(deletePrefix):
                    for saveRule in rulesToSave:
                        savePrefix = os.path.join(pathName, saveRule)
                        if fileName.startswith(savePrefix):
                            nextFile = True
                            break
                    if nextFile:
                        break
                    os.remove(fileName)
    os.remove('File-Engine.jar')
    delDir(pathName)
    shutil.make_archive(pathName, 'zip', pathName)  # 压缩
    shutil.rmtree(pathName)  # 删除临时文件
    os.rename('File-Engine.zip', 'File-Engine.jar')
    print('=======Finish!======')


def delDir(path):
    """
    清理空文件夹和空文件
    param path: 文件路径，检查此文件路径下的子文件
    """
    try:
        files = os.listdir(path)  # 获取路径下的子文件(夹)列表
        for file in files:
            if os.path.isdir(os.fspath(path+'/'+file)):  # 如果是文件夹
                if not os.listdir(os.fspath(path+'/'+file)):  # 如果子文件为空
                    os.rmdir(os.fspath(path+'/'+file))  # 删除这个空文件夹
                else:
                    delDir(os.fspath(path+'/'+file))  # 遍历子文件
                    if not os.listdir(os.fspath(path+'/'+file)):  # 如果子文件为空
                        os.rmdir(os.fspath(path+'/'+file))  # 删除这个空文件夹
            elif os.path.isfile(os.fspath(path+'/'+file)):  # 如果是文件
                if os.path.getsize(os.fspath(path+'/'+file)) == 0:  # 文件大小为0
                    os.remove(os.fspath(path+'/'+file))  # 删除这个文件
        return
    except FileNotFoundError:
        print("文件夹路径错误")


def getFileMd5(file_name):
    """
    计算文件的md5
    :param file_name:
    :return:
    """
    m = hashlib.md5()  # 创建md5对象
    with open(file_name, 'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)  # 更新md5对象

    return m.hexdigest()  # 返回md5对象


buildDir = 'build'
if not os.path.exists(buildDir):
    os.mkdir(buildDir)

jdkPath = ''
if len(sys.argv) == 2:
    jdkPath = sys.argv[1]
    print("JAVA_HOME set to " + jdkPath)

# 编译jar
if os.system('set JAVA_HOME=' + jdkPath + '&& mvn clean compile package') != 0:
    print('maven compile failed')
    exit()
# 获取版本
configs = jproperties.Properties()
with open(r'.\target\maven-archiver\pom.properties', 'rb') as f:
    configs.load(f)
fileEngineVersion = configs.get('version')

# 切换到build
os.chdir(buildDir)

if os.system(r'xcopy ..\target\File-Engine.jar . /Y') != 0:
    print('xcopy File-Engine.jar failed.')
    exit()
os.system(r'del /Q /F File-Engine.zip')

# 生成jre
binPath = os.path.join(jdkPath, 'bin')
jdepExe = os.path.join(binPath, 'jdeps.exe')
deps = subprocess.check_output([jdepExe, '--ignore-missing-deps', '--print-module-deps', r'..\target\File-Engine-' + fileEngineVersion.data + '.jar'])
depsStr = deps.decode().strip()
modulesFromJar = depsStr.split(',')
javaExe = os.path.join(binPath, 'java.exe')
modules = subprocess.check_output([javaExe, '--list-modules'])
modulesStr = modules.decode().strip()
tmpModuleList = modulesStr.splitlines()
moduleList = []
for each in tmpModuleList:
    if not each.startswith('jdk.') and each.startswith('java.'):
        moduleName = each.split('@')
        moduleList.append(moduleName[0])
for eachModule in modulesFromJar:
    moduleList.append(eachModule)
moduleList = set(moduleList)
print("deps: " + str(moduleList))
depsStr = ','.join(moduleList)
shutil.rmtree('jre')
jlinkExe = os.path.join(binPath, 'jlink.exe')
jlinkExe = jlinkExe[0:1] + '\"' + jlinkExe[1:] + '\"'
if os.system(jlinkExe + r' --no-header-files --no-man-pages --module-path jmods --add-modules ' + depsStr + ' --output jre') != 0:
    print('Generate jre failed.')
    exit()
os.system('pause')
# 精简jar
delFileInZip()
md5Str = getFileMd5('File-Engine.jar')
print("File-Engine.jar md5: " + md5Str)

current_dir = os.getcwd()
launchWrapCppFile = os.path.join(
    current_dir, r'..\C++\launcherWrap\launcherWrap\launcherWrap.cpp')

# 计算File-Engine.jar md5
strs: list
with open(launchWrapCppFile, 'r', encoding='utf-8') as f:
    strs = f.readlines()
    for i in range(len(strs)):
        if strs[i].startswith('#define FILE_ENGINE_JAR_MD5'):
            strs[i] = '#define FILE_ENGINE_JAR_MD5 ' + \
                "\"" + md5Str + "\"" + '\n'
            break

with open(launchWrapCppFile, 'w', encoding='utf-8') as f:
    f.writelines(strs)

print('Generate File-Engine.zip')
shutil.make_archive('File-Engine', 'zip', root_dir='.', base_dir='jre')

with zipfile.ZipFile('File-Engine.zip', mode="a") as f:
    f.write('File-Engine.jar')

if os.system(r'xcopy File-Engine.zip "..\C++\launcherWrap\launcherWrap\" /Y') != 0:
    print('xcopy File-Engine.zip to launcherWrap directory failed.')
    exit()

# 编译启动器
vsPathList = vswhere.find(
    latest=True, requires='Microsoft.Component.MSBuild', find='MSBuild\**\Bin\MSBuild.exe')

if not vsPathList:
    raise RuntimeError("Cannot find visual studio installation or MSBuild.exe")
vsPath = vsPathList[0]
vsPath = vsPath[0:1] + '\"' + vsPath[1:] + "\""
os.system(vsPath + r' ..\C++\launcherWrap\launcherWrap.sln /p:Configuration=Release')

os.system(r'xcopy ..\C++\launcherWrap\x64\Release\launcherWrap.exe .\ /F /Y')
os.system(r'del /Q /F File-Engine.exe')
os.system(r'ren launcherWrap.exe File-Engine.exe')
