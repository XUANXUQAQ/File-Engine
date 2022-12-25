import hashlib
import os
import shutil
import zipfile


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
    raise RuntimeError('build dir not exist.')

jreDir = os.path.join(buildDir, 'jre')
if not os.path.exists(jreDir):
    raise RuntimeError('jre runtime dir not exist.')

os.chdir(buildDir)

os.system(r'xcopy ..\target\File-Engine.jar . /Y')
os.system(r'del /Q /F File-Engine.zip')

delFileInZip()
md5Str = getFileMd5('File-Engine.jar')
print("File-Engine.jar md5: " + md5Str)

current_dir = os.getcwd()
launchWrapCppFile = os.path.join(
    current_dir, r'..\C++\launcherWrap\launcherWrap\launcherWrap.cpp')

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

os.system(r'xcopy File-Engine.zip "..\C++\launcherWrap\launcherWrap\" /Y')

os.system(r'msbuild ..\C++\launcherWrap\launcherWrap.sln /p:Configuration=Release')

os.system(r'xcopy ..\C++\launcherWrap\x64\Release\launcherWrap.exe .\ /F /Y')
os.system(r'del /Q /F File-Engine.exe')
os.system(r'ren launcherWrap.exe File-Engine.exe')
