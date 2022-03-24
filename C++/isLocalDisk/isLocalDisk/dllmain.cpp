// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <Windows.h>
#include "file_engine_dllInterface_IsLocalDisk.h"
//#define TEST

using namespace std;

extern "C" __declspec(dllexport) BOOL isLocalDisk(const char* path);
extern "C" __declspec(dllexport) BOOL isDiskNTFS(const char* disk);

JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_IsLocalDisk_isLocalDisk
(JNIEnv* env, jobject, jstring path)
{
	const BOOL isLocalDiskVal = isLocalDisk(env->GetStringUTFChars(path, nullptr));
	return static_cast<jboolean>(isLocalDiskVal);
}

JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_IsLocalDisk_isDiskNTFS
(JNIEnv* env, jobject, jstring disk)
{
	const BOOL tmp = isDiskNTFS(env->GetStringUTFChars(disk, nullptr));
	return static_cast<jboolean>(tmp);
}

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

