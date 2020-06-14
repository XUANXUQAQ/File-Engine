#include <iostream>
#include <windows.h>
#include <tchar.h>
#include <string>
#include <tlhelp32.h>
#include <fstream>
#pragma comment(lib, "shell32.lib")

using namespace std;
//#define TEST

const int maxn = (1 << 10);

BOOL IsExistProcess(const char* pName)
{
    HANDLE hProcessSnap;
    PROCESSENTRY32 pe32;
    DWORD dwPriorityClass;

    bool bFind = false;
    // Take a snapshot of all processes in the system.
    hProcessSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hProcessSnap == INVALID_HANDLE_VALUE)
    {
        return false;
    }

    // Set the size of the structure before using it.
    pe32.dwSize = sizeof(PROCESSENTRY32);

    // Retrieve information about the first process,
    // and exit if unsuccessful
    if (!Process32First(hProcessSnap, &pe32))
    {
        CloseHandle(hProcessSnap);          // clean the snapshot object
        return false;
    }

    // Now walk the snapshot of processes, and
    // display information about each process in turn
    do
    {
        // Retrieve the priority class.
        dwPriorityClass = 0;
        if (::strstr((const char*)pe32.szExeFile, pName) != NULL)
        {
            bFind = true;
            break;
        }
    } while (Process32Next(hProcessSnap, &pe32));

    CloseHandle(hProcessSnap);
    return bFind;
}

string GetExePath(void)
{
    char szFilePath[1000] = { 0 };
    GetModuleFileNameA(NULL, szFilePath, 1000);
    (strrchr(szFilePath, '\\'))[0] = 0; // 删除文件名，只获得路径字串
    string path = szFilePath;

    return path;
}

boolean removeFile(string path)
{
    char _path[1000];
    memset(&_path, 0, 1000);
    strcpy_s(_path, path.c_str());
    if (remove(_path) == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void copyFile(string file, string dest)
{
    ifstream in;
    ofstream out;
    char _file[1000];
    char _dest[1000];
    strcpy_s(_file, file.c_str());
    strcpy_s(_dest, dest.c_str());
    in.open(_file, ios::binary); //读取文件
    out.open(_dest, ios::binary);
    while (!in.eof())
    {
        char buf[maxn] = "";
        int len = sizeof(buf);
        in.read(buf, sizeof(buf));
        if (in.peek() == -1)
        {
            char* p = &buf[len - 1];
            while ((*p) == 0)
            {
                len--;
                p--;
            }
        }
        out.write(buf, len);
    }
    in.close();
    out.flush();
    out.close();
}

#ifndef TEST
int main(int argc, char* argv[])
{
    if (argc == 2)
    {
        char fileName[1000];
        memset(fileName, 0, 1000);
        strcpy_s(fileName, argv[1]);
        string currentPath = GetExePath();
        cout << "currentPath:" << currentPath.c_str() << endl;
        string fileSearcherPath = currentPath + "\\user\\fileSearcher.exe";
        string fileMonitorPath = currentPath + "\\user\\fileMonitor.dll";
        string hotkeyListenerPath = currentPath + "\\user\\hotkeyListener.dll";
        string getAscIIPath = currentPath + "\\user\\getAscII.dll";
        string shortcutGenPath = currentPath + "\\user\\shortcutGenerator.vbs";
        string isLocalDiskPath = currentPath + "\\user\\isLocalDisk.dll";
        string origin;
        origin.append(currentPath);
        origin.append("\\");
        origin.append(fileName); //拼接原文件地址
        cout << "origin:" << origin.c_str() << endl;
        string newFile;
        newFile.append(currentPath);
        newFile.append("\\tmp\\");
        newFile.append(fileName);
        cout << "new:" << newFile.c_str() << endl;
        int count = 0;
        while (true)
        {
            Sleep(50);
            count++;
            if (count > 200)
            {
                cout << "time out" << endl;
                return 0;
            }
            if (!IsExistProcess(fileName))
            {
                break;
            }
            else
            {
                cout << "still exist" << endl;
            }
        }
        if (!removeFile(fileSearcherPath))
        {
            cout << "fileSearcher remove failed" << endl;
        }
        else
        {
            cout << "fileSearcher remove successfully" << endl;
        }
        if (!removeFile(fileMonitorPath))
        {
            cout << "fileMonitor remove failed" << endl;
        }
        else
        {
            cout << "fileMonitor remove successfully" << endl;
        }
        if (!removeFile(hotkeyListenerPath))
        {
            cout << "hotkeyListener remove failed" << endl;
        }
        else
        {
            cout << "hotkeyListener remove successfully" << endl;
        }
        if (!removeFile(getAscIIPath))
        {
            cout << "getAscII remove failed" << endl;
        }
        else
        {
            cout << "getAscII remove successfully" << endl;
        }
        if (!removeFile(origin))
        {
            cout << "origin file remove failed" << endl;
        }
        else
        {
            cout << "origin file remove successfully" << endl;
        }
        if (!removeFile(shortcutGenPath))
        {
            cout << "shortcutGenerator remove failed" << endl;
        }
        else
        {
            cout << "shortcutGenerator remove successfully" << endl;
        }
        if (!removeFile(isLocalDiskPath))
        {
            cout << "isLocaldisk dll remove failed" << endl;
        }
        else {
            cout << "isLocaldisk dll remove successfully" << endl;
        }
        cout << "copy new File" << endl;
        copyFile(newFile, origin);
        ShellExecute(NULL, _T("open"), (LPCWSTR)origin.c_str(), NULL, NULL, SW_SHOW);
        return 0;
    }
}
#else
int main() {
    cout << IsExistProcess("File-Engine-x64.exe") << endl;
    getchar();
}
#endif