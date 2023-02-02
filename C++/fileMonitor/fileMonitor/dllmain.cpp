// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#ifdef TEST
#include <iostream>
#endif
#include <Windows.h>
#include <tchar.h>
#include <iomanip>
#include <string>
#include <thread>
#include <concurrent_queue.h>
#include <concurrent_unordered_map.h>
#include <io.h>
#include "string_wstring_converter.h"
#include "dir_changes_reader.h"

#include "file_engine_dllInterface_FileMonitor.h"
//#define TEST

using namespace concurrency;

using file_record_queue = concurrent_queue<std::wstring>;

void monitor(const char* path);
void stop_monitor(const std::string path);
void monitor_path(const std::string& path);
void add_record(const std::wstring& record);
void delete_record(const std::wstring& record);
bool pop_del_file(std::wstring& record);
bool pop_add_file(std::wstring& record);

file_record_queue file_added_queue;
file_record_queue file_del_queue;
concurrent_unordered_map<std::string, std::atomic_bool> stop_flag;

JNIEXPORT void JNICALL Java_file_engine_dllInterface_FileMonitor_monitor
(JNIEnv* env, jobject, jstring path)
{
	const char* str = env->GetStringUTFChars(path, nullptr);
	monitor(str);
	env->ReleaseStringUTFChars(path, str);
}

JNIEXPORT void JNICALL Java_file_engine_dllInterface_FileMonitor_stop_1monitor
(JNIEnv* env, jobject, jstring path)
{
	const char* str = env->GetStringUTFChars(path, nullptr);
	stop_monitor(str);
	env->ReleaseStringUTFChars(path, str);
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
 * 添加文件到file_added_queue
 */
void add_record(const std::wstring& record)
{
	file_added_queue.push(record);
}

/**
 * 添加文件到file_del_queue
 */
void delete_record(const std::wstring& record)
{
	file_del_queue.push(record);
}

/**
 * 从file_del_queue中取出一个结果
 */
bool pop_del_file(std::wstring& record)
{
	return file_del_queue.try_pop(record);
}

/**
 * 从file_add_queue中取出一个结果
 */
bool pop_add_file(std::wstring& record)
{
	return file_added_queue.try_pop(record);
}

void monitor(const char* path)
{
	std::string _path(path);
	auto&& flag = stop_flag.find(_path);
	if (flag == stop_flag.end())
	{
		stop_flag.insert(std::make_pair(_path, true));
	}
	else
	{
		flag->second.store(true);
	}
	std::thread t(monitor_path, _path);
	t.join();
}

void stop_monitor(const std::string path)
{
	auto&& flag = stop_flag.find(path);
	if (flag == stop_flag.end()) 
	{
		return;
	}
	flag->second.store(false);
}

/**
 * 开始监控文件夹
 */
void monitor_path(const std::string& path)
{
	const auto wPath = string2wstring(path);
	DirectoryChangesReader dcr(wPath);
	auto&& flag = stop_flag.at(path);
	while (flag.load())
	{
		dcr.EnqueueReadDirectoryChanges();
		if (const DWORD rv = dcr.WaitForHandles(); rv == WAIT_OBJECT_0)
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
					if (data.find(L"$RECYCLE.BIN") == std::wstring::npos)
					{
						std::wstring data_with_disk;
						data_with_disk.append(wPath).append(data);
						add_record(data_with_disk);
					}
					break;
				case FILE_ACTION_REMOVED:
				case FILE_ACTION_RENAMED_OLD_NAME:
					if (data.find(L"$RECYCLE.BIN") == std::wstring::npos)
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
#ifdef TEST
					std::cout << "Unknown command!  " << action << std::endl;
#endif
					break;
				}
			}
		}
		Sleep(100);
	}
#ifdef TEST
	std::cout << "stop monitoring " << path << std::endl;
#endif
}
