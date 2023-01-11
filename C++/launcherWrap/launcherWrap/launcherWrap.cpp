// #define TEST
#include <corecrt_io.h>
#include <direct.h>
#include <Windows.h>
#include <ShlObj.h>
#include "resource.h"
#include <TlHelp32.h>
#include <fstream>
#include <string>
#include <Psapi.h>
#include <tchar.h>
#include <codecvt>
#include "zip/zip.h"
#include "md5.h"
#ifdef TEST
#include <iostream>
#endif

#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment(lib, "User32.lib")
#define MAX_LOG_PRESERVE_DAYS 5
#define CHECK_TIME_THRESHOLD 1

// TODO 该变量为File-Engine.zip中的File-Engine.jar的md5值
#define FILE_ENGINE_JAR_MD5 "5d1f875480cafada521fbe0cf6d99db9"

#ifndef TEST
#pragma comment( linker, "/subsystem:windows /entry:mainCRTStartup" )
#endif

constexpr auto* g_file_engine_zip_name = "File-Engine.zip";
std::string g_jvm_parameters =
"-Xms8M -Xmx256M -XX:+UseG1GC -XX:+UseStringDeduplication -XX:MaxHeapFreeRatio=20 -XX:MinHeapFreeRatio=10 -XX:+CompactStrings";

char g_close_signal_file[1000];
char g_open_from_jar_signal_file[1000];
char g_file_engine_jar_path[1000];
char g_file_engine_working_dir[1000];
char g_jre_path[1000];
char g_update_signal_file[1000];
char g_new_file_engine_jar_path[1000];
char g_log_file_path[1000];
char g_jvm_parameter_file_path[1000];

short g_restart_count = 0;
std::time_t g_restart_time = std::time(nullptr);

bool is_close_exist();
DWORD find_process();
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
void delete_jre_dir();
bool remove_dir(const char* szFileDir);
std::string get_date();
time_t convert(int year, int month, int day);
int get_days(const char* from, const char* to);
void check_logs();
void init_jvm_parameters();
tm get_tm();
std::wstring string2wstring(const std::string& str);

int main()
{
	if (is_launched())
	{
		return 0;
	}
	init_path();
	check_logs();
#ifdef TEST
	std::cout << "file-engine.jar path :  " << g_file_engine_jar_path << std::endl;
	std::cout << "jre path: " << g_jre_path << std::endl;
	std::cout << "file-engine working dir: " << g_file_engine_working_dir << std::endl;
	std::cout << "new file-engine.jar path: " << g_new_file_engine_jar_path << std::endl;
	std::cout << "update signal file: " << g_update_signal_file << std::endl;
	std::cout << "close signal file : " << g_close_signal_file << std::endl;
	std::cout << "log file path: " << g_log_file_path << std::endl;
#endif
	if (is_dir_not_exist(g_log_file_path))
	{
		if (_mkdir(g_log_file_path))
		{
			std::string msg;
			msg.append("Create dir ").append(g_log_file_path).append(" failed");
			MessageBoxA(nullptr, msg.c_str(), "Error", MB_OK);
			return 0;
		}
	}
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
	if (!is_file_exist(g_open_from_jar_signal_file))
	{
		const DWORD pid = find_process();
		if (pid)
		{
			system(("TASKKILL /PID " + std::to_string(pid) + " /F").c_str());
		}
#ifdef TEST
		std::cout << "starting file-engine" << std::endl;
#endif
		restart_file_engine(true);
	}
	else
	{
		remove(g_open_from_jar_signal_file);
	}
#ifdef TEST
	std::cout << "start loop" << std::endl;
#endif
	std::time_t start_time = std::time(nullptr);
	while (!is_close_exist())
	{
		const std::time_t tmp = std::time(nullptr) - start_time;
		if (tmp > CHECK_TIME_THRESHOLD)
		{
			start_time = std::time(nullptr);
			if (!find_process())
			{
#ifdef TEST
				std::cout << "File-Engine process not exist" << std::endl;
#endif
				restart_file_engine(false);
			}
		}
		Sleep(50);
	}
	return 0;
}

void init_jvm_parameters()
{
	if (is_file_exist(g_jvm_parameter_file_path))
	{
		std::ifstream input_stream(g_jvm_parameter_file_path, std::ios::binary);
		std::string line;
		std::string jvm_parameter;
		while (std::getline(input_stream, line))
		{
			jvm_parameter += line;
			jvm_parameter += " ";
		}
		if (!jvm_parameter.empty())
		{
			g_jvm_parameters = jvm_parameter;
		}
		input_stream.close();
	}
	else
	{
		std::ofstream output_stream(g_jvm_parameter_file_path, std::ios::binary);
		for (char each_char : g_jvm_parameters)
		{
			output_stream.put(each_char == ' ' ? '\n' : each_char);
		}
		output_stream.close();
	}
}

inline void init_path()
{
	char current_dir[1000];
	GetModuleFileNameA(nullptr, current_dir, sizeof current_dir);
	const std::string tmp_current_dir(current_dir);
	strcpy_s(current_dir, tmp_current_dir.substr(0, tmp_current_dir.find_last_of('\\')).c_str());

	std::string _file_engine_log_path(current_dir);
	_file_engine_log_path += "\\logs\\";
	strcpy_s(g_log_file_path, _file_engine_log_path.c_str());

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

	std::string open_from_jar_signal_file(g_file_engine_working_dir);
	open_from_jar_signal_file += "tmp\\openFromJar";
	strcpy_s(g_open_from_jar_signal_file, open_from_jar_signal_file.c_str());

	std::string new_file_engine_path(g_file_engine_working_dir);
	new_file_engine_path += "tmp\\File-Engine.jar";
	strcpy_s(g_new_file_engine_jar_path, new_file_engine_path.c_str());

	std::string update_signal_file(g_file_engine_working_dir);
	update_signal_file += "user\\update";
	strcpy_s(g_update_signal_file, update_signal_file.c_str());

	std::string jvm_parameter_file(g_file_engine_working_dir);
	jvm_parameter_file += "jvm.vmoptions";
	strcpy_s(g_jvm_parameter_file_path, jvm_parameter_file.c_str());
}

void check_logs()
{
	WIN32_FIND_DATAA FindFileData;
	char tmp[1000];
	strcpy_s(tmp, g_log_file_path);
	std::string _log_dir(tmp);
	_log_dir += "*";
	strcpy_s(tmp, _log_dir.c_str());

	HANDLE hFind = FindFirstFileA(tmp, &FindFileData);

	if (hFind == INVALID_HANDLE_VALUE) //如果hFind句柄创建失败，输出错误信息
	{
		FindClose(hFind);
		return;
	}
	while (FindNextFileA(hFind, &FindFileData) != 0) //当文件或者文件夹存在时
	{
		if ((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0 && strcmp(FindFileData.cFileName, ".") == 0
			|| strcmp(FindFileData.cFileName, "..") == 0) //判断是文件夹&&表示为"."||表示为"."
		{
			continue;
		}
		if ((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) //如果不是文件夹
		{
			const int days = get_days(FindFileData.cFileName, get_date().c_str());
			if (days > MAX_LOG_PRESERVE_DAYS)
			{
				std::string full_path(g_log_file_path);
				full_path += FindFileData.cFileName;
				DeleteFileA(full_path.c_str());
			}
		}
	}
	FindClose(hFind);
}

inline void delete_jre_dir()
{
	remove_dir(g_jre_path);
}

inline time_t convert(const int year, const int month, const int day)
{
	tm info;
	info.tm_year = year - 1900;
	info.tm_mon = month - 1;
	info.tm_mday = day;
	return mktime(&info);
}

inline int get_days(const char* from, const char* to)
{
	int year, month, day;
	sscanf_s(from, "%d-%d-%d.log", &year, &month, &day);
	const int fromSecond = static_cast<int>(convert(year, month, day));
	sscanf_s(to, "%d-%d-%d", &year, &month, &day);
	const int toSecond = static_cast<int>(convert(year, month, day));
	return (toSecond - fromSecond) / 24 / 3600;
}

/**
 * 释放所有文件
 */
void release_all()
{
	// 删除jre文件夹
	delete_jre_dir();
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
	HRSRC hRsrc = FindResourceW(nullptr, MAKEINTRESOURCEW(IDR_ZIP2), L"ZIP");
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
void restart_file_engine(const bool is_ignore_close_file)
{
	init_jvm_parameters();
	if (is_ignore_close_file)
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
	// 更新到新版本
	if (is_file_exist(g_update_signal_file))
	{
		update();
	}
	else if (is_file_exist(g_file_engine_jar_path))
	{
		// 检查File-Engine.jar是否与启动器中的版本一致
		auto&& jar_md5 = GetFileHash(string2wstring(g_file_engine_jar_path).c_str());
		if (jar_md5 != FILE_ENGINE_JAR_MD5)
		{
			release_all();
		}
	}
	if (g_restart_count >= 4)
	{
		MessageBoxA(nullptr, "Launch failed", "Error", MB_OK);
		std::quick_exit(-1);
	}
	if (g_restart_count >= 3 || !is_file_exist(g_file_engine_jar_path))
	{
#ifdef TEST
		std::cout << "release File-Engine.zip" << std::endl;
#endif
		release_all();
	}
	g_restart_count++;

	std::ofstream log_file(g_log_file_path + std::string(get_date()) + ".log", std::ios::app);
	const tm t = get_tm();
	log_file << "------------------------------------------------------------------------------------------------------"
		<< std::endl;
	log_file << t.tm_hour << ":" << t.tm_min << ":" << t.tm_sec << std::endl;
	log_file.close();

	std::string command("/c ");
	const std::string jre(g_jre_path);
	command.append(jre.substr(0, 2));
	command.append("\"");
	command.append(jre.substr(2));
	command.append("bin\\java.exe\" ").append(g_jvm_parameters).append(" -jar File-Engine.jar").append(" 1> normal.log")
		.append(" 2>> ").append("\"").append(g_log_file_path).append(get_date()).append(".log").append("\"");
#ifdef TEST
	std::cout << "running command: " << command << std::endl;
#endif
	ShellExecuteA(nullptr, "open", "cmd", command.c_str(), g_file_engine_working_dir, SW_HIDE);
}

/**
 * 获取具体时间
*/
tm get_tm()
{
	time_t timep;
	time(&timep);
	tm tmpTime;
	localtime_s(&tmpTime, &timep);
	return tmpTime;
}

/**
 * 获取当前日期
 */
std::string get_date()
{
	const tm tmpTime = get_tm();
	char tmp[64];
	strftime(tmp, sizeof(tmp), "%Y-%m-%d", &tmpTime);
	return tmp;
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
	const int tmp = _access(path, 0);
	return -1 == tmp;
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
DWORD find_process()
{
	try
	{
		PROCESSENTRY32 pe;
		DWORD ret = 0;
		auto* const hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
		pe.dwSize = sizeof(PROCESSENTRY32);
		const std::string _workingDir(g_file_engine_working_dir);
		const std::wstring workingDir = string2wstring(_workingDir);
		if (!Process32First(hSnapshot, &pe))
		{
			CloseHandle(hSnapshot);
			return ret;
		}
		while (true)
		{
			pe.dwSize = sizeof(PROCESSENTRY32);
			if (Process32Next(hSnapshot, &pe) == FALSE)
				break;
			if (wcscmp(pe.szExeFile, L"java.exe") == 0)
			{
				const DWORD id = pe.th32ProcessID;
				TCHAR szProcessName[1000] = { 0 };
				get_process_full_path(id, szProcessName);
				std::wstring processName(szProcessName);
				if (processName.find(workingDir) != std::wstring::npos)
				{
					ret = id;
					break;
				}
			}
		}
		CloseHandle(hSnapshot);
		return ret;
	}
	catch (std::exception&)
	{
		std::quick_exit(-1);
	}
}


std::wstring string2wstring(const std::string& str)
{
	std::wstring result;
	//获取缓冲区大小，并申请空间，缓冲区大小按字符计算  
	const int len = MultiByteToWideChar(CP_ACP, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
	const auto buffer = new TCHAR[len + 1];
	//多字节编码转换成宽字节编码  
	MultiByteToWideChar(CP_ACP, 0, str.c_str(), static_cast<int>(str.size()), buffer, len);
	buffer[len] = '\0';
	//删除缓冲区并返回值  
	result.append(buffer);
	delete[] buffer;
	return result;
}

bool remove_dir(const char* szFileDir)
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
				remove_dir((strDir + wfd.cFileName).c_str());
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
