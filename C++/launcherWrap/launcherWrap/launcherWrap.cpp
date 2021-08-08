#define _CRT_SECURE_NO_WARNINGS

#include <corecrt_io.h>
#include <direct.h>
#include <Windows.h>
#include <ShlObj.h>
#include "resource.h"
#include <TlHelp32.h>
#include <objbase.h>
#include <iostream>
#include <string>
#include "zip/zip.h"
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment( linker, "/subsystem:windows /entry:mainCRTStartup" )

// #define TEST

char g_close_signal_file[1000];
char g_file_engine_exe_path[1000];
char g_file_engine_working_dir[1000];
char g_update_signal_file[1000];
char g_new_file_engine_exe_path[1000];
#ifdef TEST
int checkTimeCount = 10;
#else
int checkTimeCount = 100;
#endif

static int restart_count = 0;
static std::time_t restart_time = std::time(nullptr);
static bool is_restart_on_release_file = false;

bool is_close_exist();
BOOL find_process(const WCHAR* procName);
void restart_file_engine(bool);
bool release_resources();
void extract_zip();
bool is_file_exist(const char* file_path);
void release_all();
bool is_dir_not_exist(const char* path);
void update();

int main()
{
	try
	{
		char current_dir[MAX_PATH];
		_getcwd(current_dir, sizeof current_dir);
		std::string file_engine_exe_dir_string(current_dir);
		file_engine_exe_dir_string += "\\data\\";
		strcpy_s(g_file_engine_working_dir, file_engine_exe_dir_string.c_str());

		file_engine_exe_dir_string += "File-Engine-x64.exe";
		strcpy_s(g_file_engine_exe_path, file_engine_exe_dir_string.c_str());

		std::string file_engine_directory(g_file_engine_working_dir);
		file_engine_directory += "tmp\\closeDaemon";
		strcpy_s(g_close_signal_file, file_engine_directory.c_str());

		std::string new_file_engine_path(g_file_engine_working_dir);
		new_file_engine_path += "tmp\\File-Engine-x64.exe";
		strcpy_s(g_new_file_engine_exe_path, new_file_engine_path.c_str());

		std::string update_signal_file(g_file_engine_working_dir);
		update_signal_file += "user\\update";
		strcpy_s(g_update_signal_file, update_signal_file.c_str());

		std::cout << "file-engine-x64.exe path :  " << g_file_engine_exe_path << std::endl;
		std::cout << "file-engine working dir: " << g_file_engine_working_dir << std::endl;
		std::cout << "new file-engine-x64.exe path: " << g_new_file_engine_exe_path << std::endl;
		std::cout << "update signal file: " << g_update_signal_file << std::endl;
		std::cout << "close signal file : " << g_close_signal_file << std::endl;

		auto count = 0;
		if (is_dir_not_exist(g_file_engine_working_dir))
		{
			_mkdir(g_file_engine_working_dir);
		}
		restart_file_engine(true);
		while (!is_close_exist())
		{
			count++;
			if (count > checkTimeCount)
			{
				constexpr auto* const proc_name = L"File-Engine-x64.exe";
				if (!find_process(proc_name))
				{
					std::cout << "File-Engine process not exist" << std::endl;
					restart_file_engine(false);
				}
				count = 0;
			}
			Sleep(100);
		}
	}
	catch (...)
	{
	}
}

void release_all()
{
	if (release_resources())
	{
		extract_zip();
		remove("File-Engine.zip");
	}
}

/**
 * 释放File-Engine.zip文件
 */
bool release_resources()
{
	const HRSRC hRsrc = FindResourceA(nullptr, MAKEINTRESOURCEA(IDR_ZIP2), "ZIP");
	if (nullptr == hRsrc)
	{
		return false;
	}
	const DWORD size = SizeofResource(nullptr, hRsrc);
	if (0 == size)
	{
		return false;
	}
	const HGLOBAL hGlobal = LoadResource(nullptr, hRsrc);
	if (nullptr == hGlobal)
	{
		return false;
	}
	const LPVOID pBuffer = LockResource(hGlobal);
	if (nullptr == pBuffer)
	{
		return false;
	}
	FILE* fp;
	if (fopen_s(&fp, "File-Engine.zip", "wb"))
	{
		return false;
	}
	if (nullptr != fp)
	{
		if (size != fwrite(pBuffer, 1, size, fp))
		{
			fclose(fp);
			return false;
		}
		fclose(fp);
		return true;
	}
	return false;
}

/**
 * 解压File-Engine.zip
 */
void extract_zip()
{
	zip_extract("File-Engine.zip", g_file_engine_working_dir, nullptr, nullptr);
}

/**
 * 重启程序
 */
void restart_file_engine(bool isIgnoreCloseFile)
{
	if (isIgnoreCloseFile)
	{
		remove(g_close_signal_file);
	}
	else
	{
		if (is_close_exist())
		{
			return;
		}
	}
	if (is_file_exist(g_update_signal_file))
	{
		update();
	}
	std::time_t tmp_restart_time = std::time(nullptr);
	// 这次重启距离上次时间超过了10分钟，视为正常重启
	const std::time_t tmp = tmp_restart_time - restart_time;
	if (tmp > 600)
	{
		restart_count = 0;
		is_restart_on_release_file = false;
	}
	if (is_restart_on_release_file && restart_count >= 1)
	{
		MessageBoxA(nullptr, "Launch failed after 3 retries", "Error", MB_OK);
		exit(-1);
	}
	if (restart_count > 3 || !is_file_exist(g_file_engine_exe_path))
	{
		release_all();
		is_restart_on_release_file = true;
		restart_count = 0;
	}
	restart_count++;
	CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
	ShellExecuteA(nullptr, "open", g_file_engine_exe_path, nullptr, g_file_engine_working_dir, SW_SHOWNORMAL);
	CoUninitialize();
}

void update()
{
	CopyFileA(g_file_engine_exe_path, g_new_file_engine_exe_path, false);
}

/**
 * 检查关闭标志是否存在
 */
bool is_close_exist()
{
	return is_file_exist(g_close_signal_file);
}

bool is_dir_not_exist(const char* path)
{
	return ENOENT == _access(path, 0);
}

bool is_file_exist(const char* file_path)
{
	FILE* fp = nullptr;
	fopen_s(&fp, file_path, "rb");
	if (fp != nullptr){
		fclose(fp);
		return true;
	}
	return false;
}

/**
 * 查找File-Engine进程是否存在
 */
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
