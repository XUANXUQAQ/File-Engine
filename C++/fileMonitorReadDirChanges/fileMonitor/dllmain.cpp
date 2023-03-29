// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#ifdef TEST
#include <iostream>
#endif
#include <Windows.h>
#include <iomanip>
#include <string>
#include <thread>
#include <concurrent_queue.h>
#include <concurrent_unordered_map.h>
#include "string_wstring_converter.h"
#include "dir_changes_reader.h"

#include "file_engine_dllInterface_FileMonitor.h"
//#define TEST

using namespace concurrency;

using file_record_queue = concurrent_queue<std::wstring>;

void monitor(const char* path);
void stop_monitor(const std::string& path);
void monitor_path(const std::string& path);
bool pop_del_file(std::wstring& record);
bool pop_add_file(std::wstring& record);

file_record_queue file_added_queue;
file_record_queue file_del_queue;
concurrent_unordered_map<std::string, bool*> stop_flag;

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
	if (std::wstring record; pop_add_file(record))
	{
		const auto str = wstring2string(record);
		return env->NewStringUTF(str.c_str());
	}
	return nullptr;
}

JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_FileMonitor_pop_1del_1file
(JNIEnv* env, jobject)
{
	if (std::wstring record; pop_del_file(record))
	{
		const auto str = wstring2string(record);
		return env->NewStringUTF(str.c_str());
	}
	return nullptr;
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
	std::string path_str(path);
	if (auto&& flag = stop_flag.find(path_str); flag == stop_flag.end())
	{
		auto flag_ptr = new bool(true);
		stop_flag.insert(std::make_pair(path_str, flag_ptr));
	}
	else
	{
		*flag->second = true;
	}
	monitor_path(path_str);
}

void stop_monitor(const std::string& path)
{
	auto&& flag = stop_flag.find(path);
	if (flag == stop_flag.end())
	{
		return;
	}
	*flag->second = false;
}

/**
 * 开始监控文件夹
 */
void monitor_path(const std::string& path)
{
	const auto wPath = string2wstring(path);
	DirectoryChangesReader<1, 65536> dcr(wPath);
	const bool* flag = stop_flag.at(path);
	while (*flag)
	{
		dcr.EnqueueReadDirectoryChanges();
		if (const DWORD rv = dcr.WaitForHandles(); rv == WAIT_OBJECT_0)
		{
			for (auto&& res = dcr.GetDirectoryChangesResultW();
				const auto & [action, data] : res)
			{
				switch (action)
				{
				case FILE_ACTION_ADDED:
				case FILE_ACTION_RENAMED_NEW_NAME:
					if (data.find(L"$RECYCLE.BIN") == std::wstring::npos)
					{
						std::wstring data_with_disk;
						data_with_disk.append(wPath).append(data);
						file_added_queue.push(data_with_disk);
					}
					break;
				case FILE_ACTION_RENAMED_OLD_NAME:
				case FILE_ACTION_REMOVED:
					if (data.find(L"$RECYCLE.BIN") == std::wstring::npos)
					{
						std::wstring data_with_disk;
						data_with_disk.append(wPath).append(data);
						file_del_queue.push(data_with_disk);
					}
					break;
				case FILE_ACTION_MODIFIED:
				default:
					break;
				}
			}
		}
		Sleep(500);
	}
	delete flag;
#ifdef TEST
	std::cout << "stop monitoring " << path << std::endl;
#endif
}
