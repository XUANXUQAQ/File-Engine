// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include <Windows.h>
#include <iomanip>
#include <string>
#include <concurrent_queue.h>
#include <concurrent_unordered_map.h>
#include <mutex>
#include "string_wstring_converter.h"
#include "file_engine_dllInterface_FileMonitor.h"
#include "NTFSChangesWatcher.h"

// #define TEST

#ifdef TEST
#include <iostream>
#endif

using namespace concurrency;

using file_record_queue = concurrent_queue<std::string>;

void monitor(const char* path);
void stop_monitor(const std::string& path);
void monitor_path(const std::string& path);
bool pop_del_file(std::string& record);
bool pop_add_file(std::string& record);
void push_add_file(const std::u16string& record);
void push_del_file(const std::u16string& record);

file_record_queue file_added_queue;
file_record_queue file_del_queue;
concurrent_unordered_map<std::string, NTFSChangesWatcher*> ntfs_changes_watcher_map;

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
	if (std::string record; pop_add_file(record))
	{
		return env->NewStringUTF(record.c_str());
	}
	return nullptr;
}

JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_FileMonitor_pop_1del_1file
(JNIEnv* env, jobject)
{
	if (std::string record; pop_del_file(record))
	{
		return env->NewStringUTF(record.c_str());
	}
	return nullptr;
}

JNIEXPORT void JNICALL Java_file_engine_dllInterface_FileMonitor_delete_1usn
(JNIEnv* env, jobject, jstring path)
{
	const char* str = env->GetStringUTFChars(path, nullptr);
	if (auto&& watcher = ntfs_changes_watcher_map.find(str);
		watcher != ntfs_changes_watcher_map.end())
	{
		if (!watcher->second->DeleteJournal())
		{
			fprintf(stderr, "Failed to delete journal %s\n", str);
		}
	}
	env->ReleaseStringUTFChars(path, str);
}

/**
 * 从file_del_queue中取出一个结果
 */
bool pop_del_file(std::string& record)
{
	return file_del_queue.try_pop(record);
}

/**
 * 从file_add_queue中取出一个结果
 */
bool pop_add_file(std::string& record)
{
	return file_added_queue.try_pop(record);
}

void push_add_file(const std::u16string& record)
{
	const std::wstring wstr(reinterpret_cast<LPCWSTR>(record.c_str()));
	auto&& str = wstring2string(wstr);
#ifdef TEST
	std::cout << "file added: " << str << std::endl;
#endif
	file_added_queue.push(str);
}


void push_del_file(const std::u16string& record)
{
	const std::wstring wstr(reinterpret_cast<LPCWSTR>(record.c_str()));
	auto&& str = wstring2string(wstr);
#ifdef TEST
	std::cout << "file removed: " << str << std::endl;
#endif
	file_del_queue.push(str);
}

void monitor(const char* path)
{
	const std::string path_str(path);
	auto&& watcher_iter = ntfs_changes_watcher_map.find(path_str);
	if (watcher_iter == ntfs_changes_watcher_map.end())
	{
		monitor_path(path_str);
		return;
	}
	stop_monitor(path_str);
	const auto watcher_ptr = watcher_iter->second;
	delete watcher_ptr;
	monitor_path(path_str);
}

void stop_monitor(const std::string& path)
{
	auto&& watcher_iter = ntfs_changes_watcher_map.find(path);
	if (watcher_iter == ntfs_changes_watcher_map.end())
	{
		return;
	}
	watcher_iter->second->stopWatch();
	Sleep(100);
	unsigned count = 0;
	while (!watcher_iter->second->isStopWatch())
	{
		watcher_iter->second->stopWatch();
		if (count > 100)
		{
			printf("%s\n", "Error wait for ntfs watcher timeout");
			return;
		}
		++count;
		Sleep(100);
	}
}

/**
 * 开始监控文件夹
 */
void monitor_path(const std::string& path)
{
#ifdef TEST
	std::cout << "monitoring " << path << std::endl;
#endif
	auto watcher = new NTFSChangesWatcher(path[0]);
	ntfs_changes_watcher_map.insert(std::make_pair(path, watcher));
	watcher->WatchChanges(push_add_file, push_del_file);
}
