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
void monitor_path(const std::string& path, const bool* flag);
bool pop_del_file(std::string& record);
bool pop_add_file(std::string& record);
void push_add_file(const std::string& record);
void push_del_file(const std::string& record);

file_record_queue file_added_queue;
file_record_queue file_del_queue;
concurrent_unordered_map<std::string, bool*> monitor_thread_status_map;

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
	std::string path_str(path);
	if (auto&& packaged_task_iter = monitor_thread_status_map.find(path_str); 
		packaged_task_iter == monitor_thread_status_map.end())
	{
		auto* flag = new bool(true);
		monitor_thread_status_map.insert(std::make_pair(path_str, flag));
		monitor_path(path_str, flag);
	}
}

void stop_monitor(const std::string& path)
{
	auto&& packaged_task_iter = monitor_thread_status_map.find(path);
	if (packaged_task_iter == monitor_thread_status_map.end())
	{
		return;
	}
	static std::mutex lock;
	std::lock_guard lock_guard(lock);
	monitor_thread_status_map.unsafe_erase(path);

	*packaged_task_iter->second = false;

	auto&& exit_file = path + "$$__File-Engine-Exit-Monitor__$$";
	FILE* fp = nullptr;
	if (fopen_s(&fp, exit_file.c_str(), "w") != 0)
	{
		fprintf(stderr, "Create exit monitor file failed.");
	}
	if (fp != nullptr)
	{
		if (fclose(fp) != 0)
		{
			fprintf(stderr, "Close exit monitor file failed.");
		}
	}
#ifndef TEST
	if (remove(exit_file.c_str()) != 0)
	{
		fprintf(stderr, "Delete exit monitor file failed.");
	}
#endif
}

/**
 * 开始监控文件夹
 */
void monitor_path(const std::string& path, const bool* flag)
{
	NTFSChangesWatcher watcher(path[0]);
	watcher.WatchChanges(flag, push_add_file, push_del_file);
}
