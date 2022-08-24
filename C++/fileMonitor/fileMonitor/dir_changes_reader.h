#pragma once
#include <Windows.h>
#include <string>
#include <thread>
#include <unordered_set>
#include <mutex>

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
		handles{ hEv },
		buffer{ std::make_unique<DWORD[]>(BufByteSize / sizeof(DWORD)) }
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
			} while (StepToNextNotifyInformation(fni));
		}
		return retval;
	}

	// wait for the handles in the handles array
	DWORD WaitForHandles()
	{
		constexpr DWORD wait_threshold = 5000;
		return ::WaitForMultipleObjects(Handles, handles, false, wait_threshold);
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