#define _CRT_SECURE_NO_WARNINGS

#include <corecrt_io.h>
#include <direct.h>
#include <Windows.h>
#include <ShlObj.h>
#include "resource.h"
#include <TlHelp32.h>
#include <iostream>
#include <string>
#include <Psapi.h>
#include <tchar.h>
#include <codecvt>
#include "zip/zip.h"
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment(lib, "User32.lib")

// #define TEST

#ifndef TEST
#pragma comment( linker, "/subsystem:windows /entry:mainCRTStartup" )
#endif

constexpr auto* g_file_engine_zip_name = "File-Engine.zip";
constexpr auto* g_jvm_parameters =
	"-Xms8M -Xmx128M -XX:+UseParallelGC -XX:MaxHeapFreeRatio=20 -XX:MinHeapFreeRatio=10 -XX:NewRatio=3 -XX:+CompactStrings -XX:MaxTenuringThreshold=16";

char g_close_signal_file[1000];
char g_file_engine_jar_path[1000];
char g_file_engine_working_dir[1000];
char g_jre_path[1000];
char g_update_signal_file[1000];
char g_new_file_engine_jar_path[1000];
#ifdef TEST
int g_check_time_threshold = 1;
#else
int g_check_time_threshold = 5;
#endif

int g_restart_count = 0;
std::time_t g_restart_time = std::time(nullptr);
bool g_is_restart_on_release_file = false;

bool is_close_exist();
BOOL find_process();
void restart_file_engine(bool);
bool release_resources();
void extract_zip();
bool is_file_exist(const char* file_path);
void release_all();
bool is_dir_not_exist(const char* path);
void update();
void init_path();
bool is_launched();
std::wstring get_self_name();
void deleteJreDir();
bool removeDir(const char* szFileDir);

int main()
{
	if (is_launched())
	{
		return 0;
	}
	init_path();
#ifdef TEST
	std::cout << "file-engine.jar path :  " << g_file_engine_jar_path << std::endl;
	std::cout << "jre path: " << g_jre_path << std::endl;
	std::cout << "file-engine working dir: " << g_file_engine_working_dir << std::endl;
	std::cout << "new file-engine.jar path: " << g_new_file_engine_jar_path << std::endl;
	std::cout << "update signal file: " << g_update_signal_file << std::endl;
	std::cout << "close signal file : " << g_close_signal_file << std::endl;
#endif
	if (is_dir_not_exist(g_file_engine_working_dir))
	{
		if (_mkdir(g_file_engine_working_dir))
		{
			std::string msg;
			msg.append("Create dir ").append(g_file_engine_working_dir).append(" failed");
			MessageBoxA(nullptr, msg.c_str(), "Error", MB_OK);
			return 0;
		}
	}
	if (!find_process())
	{
		restart_file_engine(true);
	}
	std::time_t start_time = std::time(nullptr);
	while (!is_close_exist())
	{
		const std::time_t tmp = std::time(nullptr) - start_time;
		if (tmp > g_check_time_threshold)
		{
			start_time = std::time(nullptr);
			if (!find_process())
			{
				std::cout << "File-Engine process not exist" << std::endl;
				restart_file_engine(false);
			}
		}
		Sleep(50);
	}
	return 0;
}

inline void init_path()
{
	char current_dir[1000];
	GetModuleFileNameA(nullptr, current_dir, sizeof current_dir);
	const std::string tmp_current_dir(current_dir);
	strcpy_s(current_dir, tmp_current_dir.substr(0, tmp_current_dir.find_last_of('\\')).c_str());

	std::string file_engine_jar_dir_string(current_dir);
	file_engine_jar_dir_string += "\\data\\";
	strcpy_s(g_file_engine_working_dir, file_engine_jar_dir_string.c_str());

	std::string jre_path(file_engine_jar_dir_string);
	jre_path += "jre\\";
	strcpy_s(g_jre_path, jre_path.c_str());

	file_engine_jar_dir_string += "File-Engine.jar";
	strcpy_s(g_file_engine_jar_path, file_engine_jar_dir_string.c_str());

	std::string file_engine_directory(g_file_engine_working_dir);
	file_engine_directory += "tmp\\closeDaemon";
	strcpy_s(g_close_signal_file, file_engine_directory.c_str());

	std::string new_file_engine_path(g_file_engine_working_dir);
	new_file_engine_path += "tmp\\File-Engine.jar";
	strcpy_s(g_new_file_engine_jar_path, new_file_engine_path.c_str());

	std::string update_signal_file(g_file_engine_working_dir);
	update_signal_file += "user\\update";
	strcpy_s(g_update_signal_file, update_signal_file.c_str());
}

inline void deleteJreDir()
{
	removeDir(g_jre_path);
}

/**
 * 释放所有文件
 */
void release_all()
{
	// 删除jre文件夹
	deleteJreDir();
	if (release_resources())
	{
		extract_zip();
#ifndef TEST
		remove(g_file_engine_zip_name);
#endif
	}
}

/**
 * 释放File-Engine.zip文件到当前文件夹
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
	if (fopen_s(&fp, g_file_engine_zip_name, "wb"))
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
inline void extract_zip()
{
	zip_extract(g_file_engine_zip_name, g_file_engine_working_dir, nullptr, nullptr);
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
	const std::time_t tmp_restart_time = std::time(nullptr);
	// 这次重启距离上次时间超过了10分钟，视为正常重启
	const std::time_t tmp = tmp_restart_time - g_restart_time;
	if (tmp > 600)
	{
		g_restart_count = 0;
		g_is_restart_on_release_file = false;
	}
	if (g_is_restart_on_release_file && g_restart_count >= 1)
	{
		MessageBoxA(nullptr, "Launch failed", "Error", MB_OK);
		exit(-1);
	}
	if (g_restart_count > 3 || !is_file_exist(g_file_engine_jar_path))
	{
		release_all();
		g_is_restart_on_release_file = true;
		g_restart_count = 0;
	}
	g_restart_count++;
	std::string command("/c ");
	const std::string jre(g_jre_path);
	command.append(jre.substr(0, 2));
	command.append("\"");
	command.append(jre.substr(2));
	command.append("bin\\java.exe\" ").append(g_jvm_parameters).append(" -jar File-Engine.jar").append(" 1> normal.log")
	       .append(" 2> error.log");
#ifdef TEST
	std::cout << "running command: " << command << std::endl;
#endif
	ShellExecuteA(nullptr, "open", "cmd", command.c_str(), g_file_engine_working_dir, SW_HIDE);
}

/**
 * 更新File-Engine
 */
void update()
{
	CopyFileA(g_new_file_engine_jar_path, g_file_engine_jar_path, false);
	remove(g_update_signal_file);
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
	if (fp != nullptr)
	{
		fclose(fp);
		return true;
	}
	return false;
}

BOOL dos_path_to_nt_path(LPTSTR pszDosPath, LPTSTR pszNtPath)
{
	TCHAR szDriveStr[500];

	//检查参数
	if (!pszDosPath || !pszNtPath)
		return FALSE;

	//获取本地磁盘字符串
	if (GetLogicalDriveStrings(500, szDriveStr))
	{
		TCHAR szDrive[3];
		for (int i = 0; szDriveStr[i]; i += 4)
		{
			TCHAR szDevName[100];
			if (!lstrcmpi(&(szDriveStr[i]), TEXT("A:\\")) || !lstrcmpi(&(szDriveStr[i]), TEXT("B:\\")))
				continue;

			szDrive[0] = szDriveStr[i];
			szDrive[1] = szDriveStr[i + 1];
			szDrive[2] = '\0';
			if (!QueryDosDevice(szDrive, szDevName, 100)) //查询 Dos 设备名
				return FALSE;

			const int cchDevName = lstrlen(szDevName);
			if (_tcsnicmp(pszDosPath, szDevName, cchDevName) == 0) //命中
			{
				lstrcpy(pszNtPath, szDrive); //复制驱动器
				lstrcat(pszNtPath, pszDosPath + cchDevName); //复制路径

				return TRUE;
			}
		}
	}

	lstrcpy(pszNtPath, pszDosPath);

	return FALSE;
}

//获取进程完整路径
BOOL get_process_full_path(DWORD dwPID, TCHAR* pszFullPath)
{
	TCHAR szImagePath[MAX_PATH];
	if (!pszFullPath)
		return FALSE;

	pszFullPath[0] = '\0';
	HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION, 0, dwPID);
	if (!hProcess)
		return FALSE;

	if (!GetProcessImageFileName(hProcess, szImagePath, MAX_PATH))
	{
		CloseHandle(hProcess);
		return FALSE;
	}

	if (!dos_path_to_nt_path(szImagePath, pszFullPath))
	{
		CloseHandle(hProcess);
		return FALSE;
	}

	CloseHandle(hProcess);

	return TRUE;
}

std::wstring get_self_name()
{
	WCHAR _proc_name[MAX_PATH];
	GetModuleFileNameW(nullptr, _proc_name, sizeof _proc_name / 2);
	const std::wstring proc_name(_proc_name);
	return proc_name.substr(proc_name.find_last_of(L'\\') + 1);
}

bool is_launched()
{
	PROCESSENTRY32 pe;
	DWORD id = 0;
	auto* const hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	pe.dwSize = sizeof(PROCESSENTRY32);
	if (!Process32First(hSnapshot, &pe))
	{
		CloseHandle(hSnapshot);
		return false;
	}
	int proc_num = 0;
	while (true)
	{
		pe.dwSize = sizeof(PROCESSENTRY32);
		if (Process32Next(hSnapshot, &pe) == FALSE)
			break;
		if (wcscmp(pe.szExeFile, get_self_name().c_str()) == 0)
		{
			proc_num++;
			if (proc_num > 1)
			{
				CloseHandle(hSnapshot);
				return true;
			}
		}
	}
	CloseHandle(hSnapshot);
	return false;
}

/**
 * 查找File-Engine进程是否存在
 */
BOOL find_process()
{
	PROCESSENTRY32 pe;
	BOOL ret = 0;
	auto* const hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	pe.dwSize = sizeof(PROCESSENTRY32);
	const std::string _workingDir(g_file_engine_working_dir);
	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	const std::wstring workingDir = converter.from_bytes(_workingDir);
	if (!Process32First(hSnapshot, &pe))
	{
		CloseHandle(hSnapshot);
		return FALSE;
	}
	while (true)
	{
		pe.dwSize = sizeof(PROCESSENTRY32);
		if (Process32Next(hSnapshot, &pe) == FALSE)
			break;
		if (wcscmp(pe.szExeFile, L"java.exe") == 0)
		{
			DWORD id = pe.th32ProcessID;
			TCHAR szProcessName[1000] = {0};
			get_process_full_path(id, szProcessName);
			std::wstring processName(szProcessName);
			if (processName.find(workingDir) != std::wstring::npos)
			{
				ret = TRUE;
				break;
			}
		}
	}
	CloseHandle(hSnapshot);
	return ret;
}

bool removeDir(const char* szFileDir)
{
	std::string strDir = szFileDir;
	if (strDir.at(strDir.length() - 1) != '\\')
		strDir += '\\';
	WIN32_FIND_DATAA wfd;
	HANDLE hFind = FindFirstFileA((strDir + "*.*").c_str(), &wfd);
	if (hFind == INVALID_HANDLE_VALUE)
		return false;
	do
	{
		if (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			if (_stricmp(wfd.cFileName, ".") != 0 &&
				_stricmp(wfd.cFileName, "..") != 0)
				removeDir((strDir + wfd.cFileName).c_str());
		}
		else
		{
			DeleteFileA((strDir + wfd.cFileName).c_str());
		}
	} while (FindNextFileA(hFind, &wfd));
	FindClose(hFind);
	RemoveDirectoryA(szFileDir);
	return true;
}