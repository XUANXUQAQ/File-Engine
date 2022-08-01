// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <iostream>
#include <Windows.h>
#include <tchar.h>
#include <iomanip>
#include <string>
#include <thread>
#include <concurrent_queue.h>
#include <unordered_set>
#include <io.h>
#include <mutex>
#include <ranges>

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
std::string wstring2string(const std::wstring& wstr);
std::wstring string2wstring(const std::string& str);

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

std::wstring string2wstring(const std::string& str)
{
	const int buf_size = MultiByteToWideChar(CP_ACP,
	                                         0, str.c_str(), -1, nullptr, 0);
	const std::unique_ptr<wchar_t> wsp(new wchar_t[buf_size]);
	MultiByteToWideChar(CP_ACP,
	                    0, str.c_str(), -1, wsp.get(), buf_size);
	std::wstring wstr(wsp.get());
	return wstr;
}

std::string wstring2string(const std::wstring& wstr)
{
	const int buf_size = WideCharToMultiByte(CP_UTF8,
	                                         0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
	const std::unique_ptr<char> sp(new char[buf_size]);
	WideCharToMultiByte(CP_UTF8,
	                    0, wstr.c_str(), -1, sp.get(), buf_size, nullptr, nullptr);
	std::string str(sp.get());
	return str;
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
 * 添加文件到add_set
 */
static void add_record(const std::wstring& record)
{
	file_added_queue.push(record);
}

/**
 * 添加文件到delete_set
 */
static void delete_record(const std::wstring& record)
{
	file_del_queue.push(record);
}

static bool pop_del_file(std::wstring& record)
{
	return file_del_queue.try_pop(record);
}

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

// A base class for handles with different invalid values.
template <std::uintptr_t hInvalid>
class Handle
{
public:
	Handle(const Handle&) = delete;

	Handle(Handle&& rhs) noexcept :
		hHandle(std::exchange(rhs.hHandle, hInvalid))
	{
	}

	Handle& operator=(const Handle&) = delete;

	Handle& operator=(Handle&& rhs) noexcept
	{
		std::swap(hHandle, rhs.hHandle);
		return *this;
	}

	// converting to a normal HANDLE
	operator HANDLE() const { return hHandle; }

protected:
	Handle(HANDLE v) : hHandle(v)
	{
		// throw if we got an invalid handle
		if (hHandle == reinterpret_cast<HANDLE>(hInvalid) || hHandle == INVALID_HANDLE_VALUE)
			throw std::runtime_error("invalid handle");
	}

	~Handle()
	{
		if (hHandle != reinterpret_cast<HANDLE>(hInvalid)) CloseHandle(hHandle);
	}

private:
	HANDLE hHandle;
};

using InvalidNullptrHandle = Handle<(std::uintptr_t)nullptr>;

// A class for directory handles
class DirectoryHandleW : public InvalidNullptrHandle
{
public:
	DirectoryHandleW(const std::wstring& dir) :
		Handle(
			CreateFileW(
				dir.c_str(), FILE_LIST_DIRECTORY,
				FILE_SHARE_READ | FILE_SHARE_DELETE | FILE_SHARE_WRITE,
				nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS |
				FILE_FLAG_OVERLAPPED, nullptr)
		)
	{
	}
};

// A class for event handles
class EventHandle : public InvalidNullptrHandle
{
public:
	EventHandle() : Handle(CreateEvent(nullptr, true, false, nullptr))
	{
	}
};

// A stepping function for FILE_NOTIFY_INFORMATION*
bool StepToNextNotifyInformation(FILE_NOTIFY_INFORMATION*& cur)
{
	if (cur->NextEntryOffset == 0) return false;
	cur = reinterpret_cast<FILE_NOTIFY_INFORMATION*>(
		reinterpret_cast<char*>(cur) + cur->NextEntryOffset
	);
	return true;
}

// A ReadDirectoryChanges support class
template <size_t Handles = 1, size_t BufByteSize = 4096>
class DirectoryChangesReader
{
public:
	static_assert(Handles > 0, "There must be room for at least 1 HANDLE");
	static_assert(BufByteSize >= sizeof(FILE_NOTIFY_INFORMATION) + MAX_PATH, "BufByteSize too small");
	static_assert(BufByteSize % sizeof(DWORD) == 0, "BufByteSize must be a multiple of sizeof(DWORD)");

	DirectoryChangesReader(const std::wstring& dirname) :
		hDir(dirname),
		ovl{},
		hEv{},
		handles{hEv},
		buffer{std::make_unique<DWORD[]>(BufByteSize / sizeof(DWORD))}
	{
	}

	// A function to fill in data to use with ReadDirectoryChangesW
	void EnqueueReadDirectoryChanges()
	{
		ovl = OVERLAPPED{};
		ovl.hEvent = hEv;
		const BOOL rdc = ReadDirectoryChangesW(
			hDir,
			buffer.get(),
			BufByteSize,
			TRUE,
			FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME |
			FILE_NOTIFY_CHANGE_ATTRIBUTES | FILE_NOTIFY_CHANGE_SIZE |
			FILE_NOTIFY_CHANGE_LAST_WRITE | FILE_NOTIFY_CHANGE_LAST_ACCESS |
			FILE_NOTIFY_CHANGE_CREATION | FILE_NOTIFY_CHANGE_SECURITY,
			nullptr,
			&ovl,
			nullptr
		);
		if (rdc == 0) throw std::runtime_error("EnqueueReadDirectoryChanges failed");
	}

	// A function to get a vector of <Action>, <Filename> pairs
	std::vector<std::pair<DWORD, std::wstring>>
	GetDirectoryChangesResultW()
	{
		std::vector<std::pair<DWORD, std::wstring>> retval;

		auto* fni = reinterpret_cast<FILE_NOTIFY_INFORMATION*>(buffer.get());

		DWORD ovlBytesReturned;
		if (GetOverlappedResult(hDir, &ovl, &ovlBytesReturned, TRUE))
		{
			do
			{
				retval.emplace_back(
					fni->Action,
					std::wstring{
						fni->FileName,
						fni->FileName + fni->FileNameLength / sizeof(wchar_t)
					}
				);
			}
			while (StepToNextNotifyInformation(fni));
		}
		return retval;
	}

	// wait for the handles in the handles array
	DWORD WaitForHandles()
	{
		return ::WaitForMultipleObjects(Handles, handles, false, INFINITE);
	}

	// access to the handles array
	HANDLE& operator[](size_t idx) { return handles[idx]; }
	constexpr size_t handles_count() const { return Handles; }
private:
	DirectoryHandleW hDir;
	OVERLAPPED ovl;
	EventHandle hEv;
	HANDLE handles[Handles];
	std::unique_ptr<DWORD[]> buffer; // DWORD-aligned
};

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
