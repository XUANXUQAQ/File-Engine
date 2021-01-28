#include <iostream>
#include <Windows.h>
#include <direct.h>
#include "Psapi.h"
#include <TlHelp32.h>
#include <objbase.h>
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "Ole32.lib")
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

bool is_close_exist();
BOOL find_process(const WCHAR* procName);
void restart_file_engine();

int main(const int argc, char* argv[])
{
    try
    {
        if (argc == 2)
        {
            strcpy_s(fileEngineDir, argv[1]);
            string file_engine_exe_dir_string(fileEngineDir);
            file_engine_exe_dir_string += "\\";
            file_engine_exe_dir_string += "File-Engine-x64.exe";
            strcpy_s(fileEngineExeDir, file_engine_exe_dir_string.c_str());
            string file_engine_directory(fileEngineDir);
            file_engine_directory += "\\";
            file_engine_directory += "tmp";
            file_engine_directory += "\\";
            file_engine_directory += "closeDaemon";
            strcpy_s(closeSignalFile, file_engine_directory.c_str());

            cout << "file-engine-x64.exe path : " << fileEngineExeDir << endl;
            cout << "close signal file : " << closeSignalFile << endl;

            auto count = 0;
            const auto* const proc_name = L"File-Engine-x64.exe";

            while (!is_close_exist())
            {
                count++;
                if (count > checkTimeCount)
                {
                    count = 0;
                    if (!find_process(proc_name))
                    {
                        cout << "File-Engine process not exist" << endl;
                        restart_file_engine();
                    }
                }
                Sleep(100);
            }
        }
    }
    catch (...)
    {}
#ifdef TEST
    else
    {
        cout << "error args" << endl;
    }
#endif
}

void restart_file_engine()
{
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    ShellExecuteA(nullptr, "open", fileEngineExeDir, nullptr, fileEngineDir, SW_SHOWNORMAL);
    CoUninitialize();
}


bool is_close_exist()
{
    FILE* fp;
    fopen_s(&fp, closeSignalFile ,"rb");
    if (fp != nullptr)
    {
        fclose(fp);
        return true;
    }
    else
    {
        return false;
    }
}


BOOL find_process(const WCHAR* procName)
{
    PROCESSENTRY32 pe;
    DWORD id = 0;
    auto* const hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    pe.dwSize = sizeof(PROCESSENTRY32);
    if (!Process32First(hSnapshot, &pe))
        return 0;
    while (true)
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