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
#include "file_engine_dllInterface_FileMonitor.h"
//#define TEST

using namespace std;
using namespace concurrency;

typedef concurrent_queue<string> file_record_queue;

void monitor(const char* path);
void stop_monitor();
void monitor_path(const char* path);
inline std::string to_utf8(const std::wstring& str);
inline std::string to_utf8(const wchar_t* buffer, int len);
inline std::wstring string_to_w_string(const std::string& str);
inline void add_record(const std::string& record);
inline void delete_record(const std::string& record);
void searchDir(const std::string&, const std::string&);
inline bool is_dir(const char* path);
bool pop_del_file(string& record);
bool pop_add_file(string& record);

static volatile bool is_running = true;
file_record_queue file_added_queue;
file_record_queue file_del_queue;

JNIEXPORT void JNICALL Java_file_engine_dllInterface_FileMonitor_monitor
(JNIEnv* env, jobject, jstring path)
{
	monitor(env->GetStringUTFChars(path, nullptr));
}

JNIEXPORT void JNICALL Java_file_engine_dllInterface_FileMonitor_stop_1monitor
(JNIEnv*, jobject)
{
	stop_monitor();
}

JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_FileMonitor_pop_1add_1file
(JNIEnv* env, jobject)
{
	string record;
	if (pop_add_file(record))
	{
		return env->NewStringUTF(record.c_str());
	}
	return nullptr;
}

JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_FileMonitor_pop_1del_1file
(JNIEnv* env, jobject)
{
	string record;
	if (pop_del_file(record))
	{
		return env->NewStringUTF(record.c_str());
	}
	return nullptr;
}

/**
 * 检查路径是否是文件夹
 */
inline bool is_dir(const char* path)
{
	struct stat s{};
	if (stat(path, &s) == 0)
	{
		if (s.st_mode & S_IFDIR)
		{
			return true;
		}
	}
	return false;
}

/**
 * 搜索文件夹，并将搜索结果输出到output_path
 */
void search_dir(const string& path)
{
	//文件句柄
	intptr_t hFile;
	//文件信息
	_finddata_t fileinfo{};
	string pathName;
	const string exdName = "\\*";

	if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是文件夹中仍有文件夹,加入列表后迭代
			//如果不是,加入列表
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					string name(fileinfo.name);
					string _path = pathName.assign(path).append("\\").append(fileinfo.name);
					add_record(to_utf8(string_to_w_string(_path)));
					search_dir(_path);
				}
			}
			else
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					string name(fileinfo.name);
					string _path = pathName.assign(path).append("\\").append(fileinfo.name);
					add_record(to_utf8(string_to_w_string(_path)));
				}
			}
		}
		while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

/**
 * 添加文件到add_set
 */
inline void add_record(const string& record)
{
	file_added_queue.push(record);
}

/**
 * 添加文件到delete_set
 */
inline void delete_record(const string& record)
{
	file_del_queue.push(record);
}

bool pop_del_file(string& record)
{
	return file_del_queue.try_pop(record);
}

bool pop_add_file(string& record)
{
	return file_added_queue.try_pop(record);
}

/**
 * 转换字符串到utf-8
 */
inline std::string to_utf8(const wchar_t* buffer, int len)
{
	const auto nChars = ::WideCharToMultiByte(
		CP_UTF8,
		0,
		buffer,
		len,
		nullptr,
		0,
		nullptr,
		nullptr);
	if (nChars == 0)
	{
		return "";
	}
	string newbuffer;
	newbuffer.resize(nChars);
	WideCharToMultiByte(
		CP_UTF8,
		0,
		buffer,
		len,
		const_cast<char*>(newbuffer.c_str()),
		nChars,
		nullptr,
		nullptr);

	return newbuffer;
}

inline std::string to_utf8(const std::wstring& str)
{
	return to_utf8(str.c_str(), static_cast<int>(str.size()));
}

inline std::wstring string_to_w_string(const std::string& str)
{
	setlocale(LC_ALL, "chs");
	const auto* const point_to_source = str.c_str();
	const auto new_size = str.size() + 1;
	auto* const point_to_destination = new wchar_t[new_size];
	wmemset(point_to_destination, 0, new_size);
	mbstowcs(point_to_destination, point_to_source, new_size);
	std::wstring result = point_to_destination;
	setlocale(LC_ALL, "C");
	return result;
}

void monitor(const char* path)
{
	is_running = true;
	cout << "Monitoring " << path << endl;
	thread t(monitor_path, path);
	t.detach();
}

void stop_monitor()
{
	is_running = false;
}

/**
 * 开始监控文件夹
 */
void monitor_path(const char* path)
{
	DWORD cb_bytes;
	char file_name[1000] = {"\0"}; //设置文件名
	char file_rename[1000] = {"\0"}; //设置文件重命名后的名字;
	char _path[1000] = {"\0"};
	char notify[1024] = {"\0"};
	wchar_t w_file_name[1000] = {L"\0"};
	wchar_t w_file_rename[1000] = {L"\0"};

	memset(_path, 0, 1000);
	strcpy_s(_path, 1000, path);

	WCHAR _dir[1000] = {};
	MultiByteToWideChar(CP_ACP, 0, _path, 1000, _dir,
	                    std::size(_dir));

	auto* const dirHandle = CreateFile(_dir,
	                                   GENERIC_READ | GENERIC_WRITE | FILE_LIST_DIRECTORY,
	                                   FILE_SHARE_READ | FILE_SHARE_WRITE,
	                                   nullptr,
	                                   OPEN_EXISTING,
	                                   FILE_FLAG_BACKUP_SEMANTICS,
	                                   nullptr);

	if (dirHandle == INVALID_HANDLE_VALUE) //若网络重定向或目标文件系统不支持该操作，函数失败，同时调用GetLastError()返回ERROR_INVALID_FUNCTION
	{
		cout << "error " << GetLastError() << endl;
		return;
	}

	auto* pnotify = reinterpret_cast<FILE_NOTIFY_INFORMATION*>(notify);

	while (is_running)
	{
		if (ReadDirectoryChangesW(dirHandle, &notify, 1024, true,
		                          FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME | FILE_NOTIFY_CHANGE_SIZE,
		                          &cb_bytes, nullptr, nullptr))
		{
			//转换文件名为多字节字符串;
			if (pnotify->FileName)
			{
				memset(file_name, 0, sizeof(file_name));
				memset(w_file_name, 0, sizeof(w_file_name));
				wcscpy_s(w_file_name, pnotify->FileName);
				WideCharToMultiByte(CP_ACP, 0, pnotify->FileName, pnotify->FileNameLength / 2, file_name, 250, nullptr,
				                    nullptr);
			}

			//获取重命名的文件名;
			if (pnotify->NextEntryOffset != 0 && (pnotify->FileNameLength > 0 && pnotify->FileNameLength < 1000))
			{
				const auto p = reinterpret_cast<PFILE_NOTIFY_INFORMATION>(reinterpret_cast<char*>(pnotify) + pnotify->
					NextEntryOffset);
				memset(file_rename, 0, 1000);
				memset(w_file_rename, 0, 1000);
				wcscpy_s(w_file_rename, pnotify->FileName);
				WideCharToMultiByte(CP_ACP, 0, p->FileName, p->FileNameLength / 2, file_rename, 250, nullptr, nullptr);
			}

			if (file_name[strlen(file_name) - 1] == '~')
			{
				file_name[strlen(file_name) - 1] = '\0';
			}
			if (file_rename[strlen(file_rename) - 1] == '~')
			{
				file_rename[strlen(file_rename) - 1] = '\0';
			}

			//设置类型过滤器,监听文件创建、更改、删除、重命名等;
			switch (pnotify->Action)
			{
			case FILE_ACTION_ADDED:
				if (strstr(file_name, "$RECYCLE.BIN") == nullptr)
				{
					string data;
					data.append(_path);
					data.append(file_name);
#ifdef TEST
                    cout << "file add : " << data << endl;
#endif
					add_record(to_utf8(string_to_w_string(data)));
					if (is_dir(data.c_str()))
					{
						search_dir(data);
					}
				}
				break;
			case FILE_ACTION_MODIFIED:
				if (strstr(file_name, "$RECYCLE.BIN") == nullptr && strstr(file_name, "fileAdded.txt") == nullptr &&
					strstr(file_name, "fileRemoved.txt") == nullptr)
				{
					string data;
					data.append(_path);
					data.append(file_name);
#ifdef TEST
                    cout << "file add : " << data << endl;
#endif
					add_record(to_utf8(string_to_w_string(data)));
					if (is_dir(data.c_str()))
					{
						search_dir(data);
					}
				}
				break;
			case FILE_ACTION_REMOVED:
				if (strstr(file_name, "$RECYCLE.BIN") == nullptr)
				{
					string data;
					data.append(_path);
					data.append(file_name);
#ifdef TEST
                    cout << "file removed : " << data << endl;
#endif
					delete_record(to_utf8(string_to_w_string(data)));
				}
				break;
			case FILE_ACTION_RENAMED_OLD_NAME:
				if (strstr(file_name, "$RECYCLE.BIN") == nullptr)
				{
					string data;
					data.append(_path);
					data.append(file_name);

					delete_record(to_utf8(string_to_w_string(data)));

					data.clear();
					data.append(_path);
					data.append(file_rename);
#ifdef TEST
                    cout << "file renamed : " << data << "->" << data << endl;
#endif
					add_record(to_utf8(string_to_w_string(data)));
					if (is_dir(data.c_str()))
					{
						search_dir(data);
					}
				}
				break;
			default:
				cout << "Unknown command!" << endl;
			}
		}
	}
	CloseHandle(dirHandle);
	cout << "stop monitoring " << _path << endl;
}
