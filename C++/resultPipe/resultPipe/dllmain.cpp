#include "pch.h"
#include <string>
#include <Windows.h>
#include <concurrent_unordered_map.h>
#include "file_engine_dllInterface_ResultPipe.h"
using namespace std;

concurrency::concurrent_unordered_map<string, pair<HANDLE, LPVOID>> connectionPool;

extern "C" __declspec(dllexport) char* getResult(char disk, const char* listName, int priority, int offset);
extern "C" __declspec(dllexport) void closeAllSharedMemory();
extern "C" __declspec(dllexport) BOOL isComplete();


JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_ResultPipe_getResult
(JNIEnv* env, jobject, jchar disk, jstring listName, jint priority, jint offset)
{
	const char* tmp = getResult(static_cast<char>(disk), env->GetStringUTFChars(listName, nullptr), priority, offset);
	return env->NewStringUTF(tmp);
}

JNIEXPORT void JNICALL Java_file_engine_dllInterface_ResultPipe_closeAllSharedMemory
(JNIEnv*, jobject)
{
	closeAllSharedMemory();
}

JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_ResultPipe_isComplete
(JNIEnv*, jobject)
{
	return static_cast<jboolean>(isComplete());
}

inline void createFileMapping(HANDLE& hMapFile, LPVOID& pBuf, size_t memorySize, const char* sharedMemoryName);

char* getResult(char disk, const char* listName, const int priority, const int offset)
{
	constexpr int maxPath = 500;
	string memoryName("sharedMemory:");
	memoryName += disk;
	memoryName += ":";
	memoryName += listName;
	memoryName += ":";
	memoryName += to_string(priority);
	const string resultMemoryName(memoryName);
	memoryName += "size";
	const string resultSizeMemoryName(memoryName);
	HANDLE hMapFile;
	void* resultsSize = nullptr;
	try
	{
		resultsSize = static_cast<int*>(connectionPool.at(resultSizeMemoryName).second);
	}
	catch(exception&)
	{
		createFileMapping(hMapFile, resultsSize, sizeof size_t, resultSizeMemoryName.c_str());
	}
	if (resultsSize == nullptr)
	{
		return nullptr;
	}
	const int resultCount = *static_cast<int*>(resultsSize) / maxPath;
	if (resultCount < offset)
	{
		return nullptr;
	}
	void* resultsPtr = nullptr;
	try
	{
		resultsPtr = connectionPool.at(resultMemoryName).second;
	}
	catch(exception&)
	{
		createFileMapping(hMapFile, resultsPtr, *static_cast<int*>(resultsSize), resultMemoryName.c_str());
	}
	return reinterpret_cast<char*>(reinterpret_cast<long long>(resultsPtr) + static_cast<long long>(offset) * maxPath);
}

inline void createFileMapping(HANDLE& hMapFile, LPVOID& pBuf, size_t memorySize, const char* sharedMemoryName)
{
	// 创建共享文件句柄
	hMapFile = CreateFileMappingA(
		INVALID_HANDLE_VALUE, // 物理文件句柄
		nullptr, // 默认安全级别
		PAGE_READWRITE, // 可读可写
		0, // 高位文件大小
		static_cast<DWORD>(memorySize), // 低位文件大小
		sharedMemoryName
	);

	pBuf = MapViewOfFile(
		hMapFile, // 共享内存的句柄
		FILE_MAP_ALL_ACCESS, // 可读写许可
		0,
		0,
		memorySize
	);
	connectionPool.insert(pair<string, pair<HANDLE, LPVOID>>(sharedMemoryName, pair<HANDLE, void*>(hMapFile, pBuf)));
}

BOOL isComplete()
{
	void* pBuf;
	static constexpr auto completeSignal = "sharedMemory:complete:status";
	if (connectionPool.find(completeSignal) == connectionPool.end())
	{
		HANDLE hMapFile;
		createFileMapping(hMapFile, pBuf, sizeof(BOOL), completeSignal);
	}
	else
	{
		pBuf = connectionPool.at(completeSignal).second;
	}
	if (pBuf == nullptr)
	{
		return FALSE;
	}
	return *static_cast<BOOL*>(pBuf);
}

void closeAllSharedMemory()
{
	for (const auto& each : connectionPool)
	{
		UnmapViewOfFile(each.second.second);
		CloseHandle(each.second.first);
	}
	connectionPool.clear();
}