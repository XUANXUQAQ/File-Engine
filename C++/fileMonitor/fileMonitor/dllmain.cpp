// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <Windows.h>
#include <tchar.h>
#include <iomanip>
#include <string>
#include <thread>
#include <concurrent_queue.h>
#include <io.h>
#include <mutex>
#include "string_wstring_converter.h"
#include "dir_changes_reader.h"

#include "file_engine_dllInterface_FileMonitor.h"
//#define TEST

using namespace concurrency;

typedef concurrent_queue<std::wstring> file_record_queue;

void monitor(const char* path);
void stop_monitor();
void monitor_path(const std::string& path);
static void add_record(const std::wstring& record);
static void delete_record(const std::wstring& record);
inline bool is_dir(const wchar_t* path);
static bool pop_del_file(std::wstring& record);
static bool pop_add_file(std::wstring& record);

static volatile bool is_running = true;
file_record_queue file_added_queue;
file_record_queue file_del_queue;

JNIEXPORT void JNICALL Java_file_engine_dllInterface_FileMonitor_monitor
(JNIEnv* env, jobject, jstring path)
{
	const char* str = env->GetStringUTFChars(path, nullptr);
	monitor(str);
	env->ReleaseStringUTFChars(path, str);
}

JNIEXPORT void JNICALL Java_file_engine_dllInterface_FileMonitor_stop_1monitor
(JNIEnv*, jobject)
{
	stop_monitor();
}

JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_FileMonitor_pop_1add_1file
(JNIEnv* env, jobject)
{
	std::wstring record;
	if (pop_add_file(record))
	{
		const auto str = wstring2string(record);
		return env->NewStringUTF(str.c_str());
	}
	return nullptr;
}

JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_FileMonitor_pop_1del_1file
(JNIEnv* env, jobject)
{
	std::wstring record;
	if (pop_del_file(record))
	{
		const auto str = wstring2string(record);
		return env->NewStringUTF(str.c_str());
	}
	return nullptr;
}

/**
 * 检查路径是否是文件夹
 */
inline bool is_dir(const wchar_t* path)
{
	struct _stat64i32 s{};
	if (_wstat(path, &s) == 0)
	{
		if (s.st_mode & S_IFDIR)
		{
			return true;
		}
	}
	return false;
}

/**
 * 搜索文件夹
 */
void search_dir(const std::wstring& path)
{
	//文件句柄
	intptr_t hFile;
	//文件信息
	_wfinddata_t fileinfo{};
	std::wstring pathName;
	const std::wstring exdName = L"\\*";

	if ((hFile = _wfindfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是文件夹中仍有文件夹,加入列表后迭代
			//如果不是,加入列表
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (wcscmp(fileinfo.name, L".") != 0 && wcscmp(fileinfo.name, L"..") != 0)
				{
					std::wstring name(fileinfo.name);
					std::wstring _path = pathName.assign(path).append(L"\\").append(fileinfo.name);
					add_record(_path);
					search_dir(_path);
				}
			}
			else
			{
				if (wcscmp(fileinfo.name, L".") != 0 && wcscmp(fileinfo.name, L"..") != 0)
				{
					std::wstring name(fileinfo.name);
					std::wstring _path = pathName.assign(path).append(L"\\").append(fileinfo.name);
					add_record(_path);
				}
			}
		}
		while (_wfindnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

/**
 * 添加文件到file_added_queue
 */
static void add_record(const std::wstring& record)
{
	file_added_queue.push(record);
}

/**
 * 添加文件到file_del_queue
 */
static void delete_record(const std::wstring& record)
{
	file_del_queue.push(record);
}

/**
 * 从file_del_queue中取出一个结果
 */
static bool pop_del_file(std::wstring& record)
{
	return file_del_queue.try_pop(record);
}

/**
 * 从file_add_queue中取出一个结果
 */
static bool pop_add_file(std::wstring& record)
{
	return file_added_queue.try_pop(record);
}

void monitor(const char* path)
{
	is_running = true;
	std::string _path(path);
	std::cout << "Monitoring " << _path << std::endl;
	std::thread t(monitor_path, _path);
	t.join();
}

void stop_monitor()
{
	is_running = false;
}

/**
 * 开始监控文件夹
 */
void monitor_path(const std::string& path)
{
	const auto wPath = string2wstring(path);
	DirectoryChangesReader dcr(wPath);
	std::wstring dir_to_search;
	while (is_running)
	{
		dcr.EnqueueReadDirectoryChanges();
		const DWORD rv = dcr.WaitForHandles();
		if (rv == WAIT_OBJECT_0)
		{
			auto res = dcr.GetDirectoryChangesResultW();
			for (const auto& pair : res)
			{
				const DWORD action = pair.first;
				std::wstring data = pair.second;
				switch (action)
				{
				case FILE_ACTION_ADDED:
				case FILE_ACTION_RENAMED_NEW_NAME:
					if (wcsstr(data.c_str(), L"$RECYCLE.BIN") == nullptr)
					{
						std::wstring data_with_disk;
						data_with_disk.append(wPath).append(data);
						add_record(data_with_disk);
						if (!dir_to_search.empty() && dir_to_search != data_with_disk)
						{
							search_dir(dir_to_search);
#ifdef TEST
							std::cout << "start search dir: " << wstring2string(dir_to_search) << std::endl;
#endif
							dir_to_search.clear();
						}
#ifdef TEST
						else
						{
							std::cout << "delay search dir: " << wstring2string(dir_to_search) << std::endl;
						}
#endif
						if (is_dir(data_with_disk.c_str()))
						{
							dir_to_search = data_with_disk;
						}
#ifdef TEST
						std::cout << "file added: " << wstring2string(data_with_disk) << std::endl;
#endif
					}
					break;
				case FILE_ACTION_REMOVED:
				case FILE_ACTION_RENAMED_OLD_NAME:
					if (wcsstr(data.c_str(), L"$RECYCLE.BIN") == nullptr)
					{
						std::wstring data_with_disk;
						data_with_disk.append(wPath).append(data);
						delete_record(data_with_disk);
#ifdef TEST
						std::cout << "file removed: " << wstring2string(data_with_disk) << std::endl;
#endif
					}
					break;
				case FILE_ACTION_MODIFIED:
					break;
				default:
					std::cout << "Unknown command!  " << action << std::endl;
				}
			}
		}
		Sleep(100);
	}
	std::cout << "stop monitoring " << path << std::endl;
}
