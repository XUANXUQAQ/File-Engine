// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <Windows.h>
#include <tchar.h>
#include <iomanip>
#include <string>
#include <fstream>
#include <ctype.h>
#include <thread>
#include <mutex>
//#define TEST


extern "C" __declspec(dllexport) void monitor(const char* path);
extern "C" __declspec(dllexport) void stop_monitor();
extern "C" __declspec(dllexport) void set_output(const char* path);

void monitor_path(const char* path);
std::string to_utf8(const std::wstring& str);
std::string to_utf8(const wchar_t* buffer, int len);
std::wstring StringToWString(const std::string& str);
void write_to_file(std::string record, const char* file_path);


using namespace std;

wchar_t fileName[1000];
wchar_t fileRename[1000];
static volatile bool isRunning = true;
char* output = new char[1000];
mutex m;


BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

void write_to_file(string record, const char* file_path) 
{
    m.lock();
    ofstream file(file_path, ios::app | ios::binary);
    file << record << endl;
    file.close();
    m.unlock();
}

std::string to_utf8(const wchar_t* buffer, int len)
{
    int nChars = ::WideCharToMultiByte(
        CP_UTF8,
        0,
        buffer,
        len,
        NULL,
        0,
        NULL,
        NULL);
    if (nChars == 0)
    {
        return "";
    }
    string newbuffer;
    newbuffer.resize(nChars);
    ::WideCharToMultiByte(
        CP_UTF8,
        0,
        buffer,
        len,
        const_cast<char*>(newbuffer.c_str()),
        nChars,
        NULL,
        NULL);

    return newbuffer;
}

std::string to_utf8(const std::wstring & str)
{
    return to_utf8(str.c_str(), (int)str.size());
}

std::wstring StringToWString(const std::string & str)
{
    setlocale(LC_ALL, "chs");
    const char* point_to_source = str.c_str();
    size_t new_size = str.size() + 1;
    wchar_t* point_to_destination = new wchar_t[new_size];
    wmemset(point_to_destination, 0, new_size);
    mbstowcs(point_to_destination, point_to_source, new_size);
    std::wstring result = point_to_destination;
    setlocale(LC_ALL, "C");
    return result;
}


__declspec(dllexport) void set_output(const char* path)
{
    memset(output, 0, 1000);
    strcpy_s(output, 1000,  path);
}

__declspec(dllexport) void monitor(const char* path)
{
    isRunning = true;
    cout << "Monitoring " << path << endl;
    thread t(monitor_path, path);
    t.detach();
    Sleep(1000); //防止路径在被保存前就被覆盖
}

__declspec(dllexport) void stop_monitor()
{
    isRunning = false;
    delete[] output;
}

void monitor_path(const char* path)
{
    DWORD cbBytes;
    char file_name[1000];   //设置文件名
    char file_rename[1000]; //设置文件重命名后的名字;
    char OUTPUT[1000];
    char fileRemoved[1000];
    char fileAdded[1000];
    char _path[1000];
    char notify[1024];

    memset(_path, 0, 1000);
    strcpy_s(_path, 1000, path);

    WCHAR _dir[1000];
    memset(_dir, 0, sizeof(_dir));
    MultiByteToWideChar(CP_ACP, 0, _path, 1000, _dir,
        sizeof(_dir) / sizeof(_dir[0]));

    HANDLE dirHandle = CreateFile(_dir,
        GENERIC_READ | GENERIC_WRITE | FILE_LIST_DIRECTORY,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        NULL);

    if (dirHandle == INVALID_HANDLE_VALUE) //若网络重定向或目标文件系统不支持该操作，函数失败，同时调用GetLastError()返回ERROR_INVALID_FUNCTION
    {
        cout << "error " << GetLastError() << endl;
        exit(0);
    }

    FILE_NOTIFY_INFORMATION* pnotify = (FILE_NOTIFY_INFORMATION*)notify;

    strcpy_s(OUTPUT, 1000, output);

    memset(fileRemoved, 0, 1000);
    memset(fileAdded, 0, 1000);

    strcpy_s(fileRemoved, 1000, OUTPUT);
    strcat_s(fileRemoved, "\\fileRemoved.txt");
    strcpy_s(fileAdded, 1000, OUTPUT);
    strcat_s(fileAdded, "\\fileAdded.txt");
    while (isRunning)
    {
        if (ReadDirectoryChangesW(dirHandle, &notify, 1024, true,
            FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME | FILE_NOTIFY_CHANGE_SIZE,
            &cbBytes, NULL, NULL))
        {
            //转换文件名为多字节字符串;
            if (pnotify->FileName)
            {
                memset(file_name, 0, sizeof(file_name));
                memset(fileName, 0, sizeof(fileName));
                wcscpy_s(fileName, pnotify->FileName);
                WideCharToMultiByte(CP_ACP, 0, pnotify->FileName, pnotify->FileNameLength / 2, file_name, 250, NULL, NULL);
            }

            //获取重命名的文件名;
            if (pnotify->NextEntryOffset != 0 && (pnotify->FileNameLength > 0 && pnotify->FileNameLength < 1000))
            {
                PFILE_NOTIFY_INFORMATION p = (PFILE_NOTIFY_INFORMATION)((char*)pnotify + pnotify->NextEntryOffset);
                memset(file_rename, 0, sizeof(file_rename));
                memset(fileRename, 0, sizeof(fileRename));
                wcscpy_s(fileRename, pnotify->FileName);
                WideCharToMultiByte(CP_ACP, 0, p->FileName, p->FileNameLength / 2, file_rename, 250, NULL, NULL);
            }

            if (file_name[strlen(file_name) - 1] == '~')
            {
                file_name[strlen(file_name) - 1] = '\0';
            }

            //设置类型过滤器,监听文件创建、更改、删除、重命名等;
            switch (pnotify->Action)
            {
            case FILE_ACTION_ADDED:
                if (strstr(file_name, "$RECYCLE.BIN") == NULL)
                {
#ifdef TEST
                    cout << "file add : ";
#endif
                    string data;
                    data.append(_path);
                    data.append(file_name);
#ifdef TEST
                    cout << data << endl;
#endif
                    write_to_file(to_utf8(StringToWString(data)), fileAdded);
                }
                break;

            case FILE_ACTION_MODIFIED:
                if (strstr(file_name, "$RECYCLE.BIN") == NULL && strstr(file_name, "fileAdded.txt") == NULL && strstr(file_name, "fileRemoved.txt") == NULL)
                {
#ifdef TEST
                    cout << "file add : ";
#endif
                    string data;
                    data.append(_path);
                    data.append(file_name);
#ifdef TEST
                    cout << data << endl;
#endif
                    write_to_file(to_utf8(StringToWString(data)), fileAdded);
                }
                break;

            case FILE_ACTION_REMOVED:
                if (strstr(file_name, "$RECYCLE.BIN") == NULL)
                {
#ifdef TEST
                    cout << "file removed : ";
#endif
                    string data;
                    data.append(_path);
                    data.append(file_name);
#ifdef TEST
                    cout << data << endl;
#endif
                    write_to_file(to_utf8(StringToWString(data)), fileRemoved);
                }
                break;

            case FILE_ACTION_RENAMED_OLD_NAME:
                if (strstr(file_name, "$RECYCLE.BIN") == NULL)
                {
#ifdef TEST
                    cout << "file renamed : ";
#endif
                    string data;
                    data.append(_path);
                    data.append(file_name);
#ifdef TEST
                    cout << data << "->";
#endif
                    write_to_file(to_utf8(StringToWString(data)), fileRemoved);

                    data.clear();
                    data.append(_path);
                    data.append(file_rename);
#ifdef TEST
                    cout << data << endl;
#endif
                    write_to_file(to_utf8(StringToWString(data)), fileAdded);
                }
                break;

            default:
                cout << "Unknown command!" << endl;
            }
        }
    }
    CloseHandle(dirHandle);
    cout << "stop monitoring " << _path << endl;
    return;
}

