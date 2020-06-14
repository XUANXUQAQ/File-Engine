// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <Windows.h>
//#define TEST

using namespace std;

extern "C" __declspec(dllexport) bool isLocalDisk(const char* path);

#ifndef TEST
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
#endif

__declspec(dllexport) bool isLocalDisk(const char* path)
{
    WCHAR disk[260];
    memset(disk,0, lstrlenW(disk));
    MultiByteToWideChar(CP_ACP, 0, path, strlen(path) + 1, disk, sizeof(disk) / sizeof(disk[0]));
    UINT type = GetDriveType(disk);
    if (type == DRIVE_FIXED)
    {
        return true;
    }
    return false;
}

#ifdef TEST
int CALLBACK WinMain(
    _In_  HINSTANCE hInstance,
    _In_  HINSTANCE hPrevInstance,
    _In_  LPSTR lpCmdLine,
    _In_  int nCmdShow
)
{
    const char* path = "D:\\";
    bool test = isLocalDisk(path);
    MessageBox(NULL, L"test", L"测试", MB_OKCANCEL);
}
#endif

