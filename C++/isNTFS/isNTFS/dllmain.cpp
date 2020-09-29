// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
//#define TEST

using namespace std;

extern "C" __declspec(dllexport) bool isDiskNTFS(const char* disk);

__declspec(dllexport) bool isDiskNTFS(const char* disk) {
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
			return true;
		}
	}
	return false;
}

#ifdef TEST
int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
	_In_opt_ HINSTANCE hPrevInstance,
	_In_ LPWSTR    lpCmdLine,
	_In_ int       nCmdShow)
{
	isNTFS("C:\\");
}
#endif
