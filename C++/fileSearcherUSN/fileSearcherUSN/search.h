#pragma once

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <winioctl.h>
#include <string>
#include <vector>
#include <algorithm>
#include "sqlite3.h"
#include <concurrent_queue.h>
#include <concurrent_unordered_map.h>
#include <atomic>

#define CONCURRENT_MAP concurrency::concurrent_unordered_map
#define CONCURRENT_QUEUE concurrency::concurrent_queue

typedef struct _pfrn_name
{
	DWORDLONG pfrn = 0;
	CString filename;
} pfrn_name;

typedef std::unordered_map<std::string, int> PriorityMap;
typedef std::unordered_map<DWORDLONG, pfrn_name> Frn_Pfrn_Name_Map;

class volume
{
public:
	volume(char vol, sqlite3* database, std::vector<std::string>* ignorePaths, PriorityMap* priorityMap);

	~volume() = default;

	char getDiskPath() const
	{
		return vol;
	}

	void collectResult(int ascii, const std::string& fullPath);

	/**
	 * 将内存中的数据保存到共享内存中
	 */
	void copyResultsToSharedMemory();

	void initVolume();

private:
	char vol;
	HANDLE hVol;
	pfrn_name pfrnName;
	Frn_Pfrn_Name_Map frnPfrnNameMap;
	sqlite3* db;
	CString path;
	sqlite3_stmt* stmt0 = nullptr;
	sqlite3_stmt* stmt1 = nullptr;
	sqlite3_stmt* stmt2 = nullptr;
	sqlite3_stmt* stmt3 = nullptr;
	sqlite3_stmt* stmt4 = nullptr;
	sqlite3_stmt* stmt5 = nullptr;
	sqlite3_stmt* stmt6 = nullptr;
	sqlite3_stmt* stmt7 = nullptr;
	sqlite3_stmt* stmt8 = nullptr;
	sqlite3_stmt* stmt9 = nullptr;
	sqlite3_stmt* stmt10 = nullptr;
	sqlite3_stmt* stmt11 = nullptr;
	sqlite3_stmt* stmt12 = nullptr;
	sqlite3_stmt* stmt13 = nullptr;
	sqlite3_stmt* stmt14 = nullptr;
	sqlite3_stmt* stmt15 = nullptr;
	sqlite3_stmt* stmt16 = nullptr;
	sqlite3_stmt* stmt17 = nullptr;
	sqlite3_stmt* stmt18 = nullptr;
	sqlite3_stmt* stmt19 = nullptr;
	sqlite3_stmt* stmt20 = nullptr;
	sqlite3_stmt* stmt21 = nullptr;
	sqlite3_stmt* stmt22 = nullptr;
	sqlite3_stmt* stmt23 = nullptr;
	sqlite3_stmt* stmt24 = nullptr;
	sqlite3_stmt* stmt25 = nullptr;
	sqlite3_stmt* stmt26 = nullptr;
	sqlite3_stmt* stmt27 = nullptr;
	sqlite3_stmt* stmt28 = nullptr;
	sqlite3_stmt* stmt29 = nullptr;
	sqlite3_stmt* stmt30 = nullptr;
	sqlite3_stmt* stmt31 = nullptr;
	sqlite3_stmt* stmt32 = nullptr;
	sqlite3_stmt* stmt33 = nullptr;
	sqlite3_stmt* stmt34 = nullptr;
	sqlite3_stmt* stmt35 = nullptr;
	sqlite3_stmt* stmt36 = nullptr;
	sqlite3_stmt* stmt37 = nullptr;
	sqlite3_stmt* stmt38 = nullptr;
	sqlite3_stmt* stmt39 = nullptr;
	sqlite3_stmt* stmt40 = nullptr;

	USN_JOURNAL_DATA ujd{};
	CREATE_USN_JOURNAL_DATA cujd{};

	std::vector<std::string>* ignorePathVector = nullptr;
	PriorityMap* priorityMap = nullptr;
	CONCURRENT_MAP<std::string, CONCURRENT_MAP<int, CONCURRENT_QUEUE<std::string>&>*> allResultsMap;

	bool getHandle();
	bool createUSN();
	bool getUSNInfo();
	bool getUSNJournal();
	bool deleteUSN() const;
	void saveResult(const std::string& _path, int ascII) const;
	void getPath(DWORDLONG frn, CString& _path);
	static int getAscIISum(const std::string& name);
	bool isIgnore(const std::string& path) const;
	void finalizeAllStatement() const;
	void saveSingleRecordToDB(sqlite3_stmt* stmt, const std::string& record, int ascii) const;
	int getPriorityBySuffix(const std::string& suffix) const;
	int getPriorityByPath(const std::string& _path) const;
	void initAllPrepareStatement();
	void initSinglePrepareStatement(sqlite3_stmt** statement, const char* init) const;
	void saveAllResultsToDb();
	void createSharedMemoryAndCopy(const std::string& listName, int priority, size_t* size,
	                               const std::string& sharedMemoryName);
	static void setCompleteSignal();
};


std::string to_utf8(const wchar_t* buffer, int len);

std::string to_utf8(const std::wstring& str);

std::string getFileName(const std::string& path);

bool initCompleteSignalMemory();

void closeSharedMemory();

void createFileMapping(HANDLE& hMapFile, LPVOID& pBuf, size_t memorySize, const char* sharedMemoryName);
