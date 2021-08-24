// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <Windows.h>
//#define TEST

using namespace std;

extern "C" __declspec(dllexport) BOOL isLocalDisk(const char* path);
extern "C" __declspec(dllexport) BOOL isDiskNTFS(const char* disk);


__declspec(dllexport) BOOL isDiskNTFS(const char* disk)
{
	char lpRootPathName[20];
	strcpy_s(lpRootPathName, disk);
	char lpVolumeNameBuffer[MAX_PATH];
	DWORD lpVolumeSerialNumber;
	DWORD lpMaximumComponentLength;
	DWORD lpFileSystemFlags;
	char lpFileSystemNameBuffer[MAX_PATH];

	if (GetVolumeInformationA(
		lpRootPathName,
		lpVolumeNameBuffer,
		MAX_PATH,
		&lpVolumeSerialNumber,
		&lpMaximumComponentLength,
		&lpFileSystemFlags,
		lpFileSystemNameBuffer,
		MAX_PATH
	)) {
		if (!strcmp(lpFileSystemNameBuffer, "NTFS")) {
			return TRUE;
		}
	}
	return FALSE;
}

__declspec(dllexport) BOOL isLocalDisk(const char* path)
{
    const UINT type = GetDriveTypeA(path);
    if (type == DRIVE_FIXED)
    {
        return TRUE;
    }
    return FALSE;
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

