#include <iostream>
#include <Windows.h>
#include <direct.h>
#include "psapi.h"    
#include <tlhelp32.h> 
#pragma comment(lib, "shell32.lib")
//#define TEST

using namespace std;

char closeSignalFile[1000];
char fileEngineExeDir[1000];
char fileEngineDir[1000];
#ifdef TEST
int checkTimeCount = 10;
#else
int checkTimeCount = 100;
#endif

bool isCloseExist();
BOOL FindProcess(const WCHAR* procName);
void restartFileEngine();

int main(int argc, char* argv[])
{
    if (argc == 2)
    {
        strcpy_s(fileEngineDir, argv[1]);
        string fileEngineExeDirString(fileEngineDir);
        fileEngineExeDirString += "\\";
        fileEngineExeDirString += "File-Engine-x64.exe";
        strcpy_s(fileEngineExeDir, fileEngineExeDirString.c_str());
        string fileEngineDirectory(fileEngineDir);
        fileEngineDirectory += "\\";
        fileEngineDirectory += "tmp";
        fileEngineDirectory += "\\";
        fileEngineDirectory += "closeDaemon";
        strcpy_s(closeSignalFile, fileEngineDirectory.c_str());

        cout << "file-engine-x64.exe path : " << fileEngineExeDir << endl;
        cout << "close signal file : " << closeSignalFile << endl;

        int count = 0;
        const WCHAR* procName = L"File-Engine-x64.exe";
        while (!isCloseExist())
        {
            count++;
            if (count > checkTimeCount)
            {
                count = 0;
                if (!FindProcess(procName))
                {
                    cout << "File-Engine process not exist" << endl;
                    restartFileEngine();
                }
            }
            Sleep(100);
        }
    }
#ifdef TEST
    else
    {
        cout << "error arg" << endl;
    }
#endif
}

void restartFileEngine()
{
    ShellExecuteA(NULL, "open", fileEngineExeDir, NULL, fileEngineDir, SW_SHOWNORMAL);
}


bool isCloseExist()
{
    FILE* fp;
    fopen_s(&fp, closeSignalFile ,"rb");
    if (fp != NULL)
    {
        fclose(fp);
        return true;
    }
    else
    {
        return false;
    }
}


BOOL FindProcess(const WCHAR* procName)
{
    PROCESSENTRY32 pe;
    DWORD id = 0;
    HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    pe.dwSize = sizeof(PROCESSENTRY32);
    if (!Process32First(hSnapshot, &pe))
        return 0;
    while (1)
    {
        pe.dwSize = sizeof(PROCESSENTRY32);
        if (Process32Next(hSnapshot, &pe) == FALSE)
            break;
        if (wcscmp(pe.szExeFile, procName) == 0)
        {
            id = pe.th32ProcessID;
            break;
        }
    }
    CloseHandle(hSnapshot);
    return id;
}