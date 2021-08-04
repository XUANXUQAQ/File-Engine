#include <iostream>
#include "stdafx.h"
#include "Volume.h"
#include <fstream>
#include <thread>
// #define TEST


typedef struct {
	char disk;
	vector<string> ignorePath;
	sqlite3* db;
} parameter;

static PriorityMap suffixPriorityMap;

void initUSN(parameter p);
void splitString(char* str, vector<string>& vec);


void initUSN(const parameter p) {
	sqlite3_exec(p.db, "BEGIN;", nullptr, nullptr, nullptr);
	volume volumeInstance(p.disk, p.db, p.ignorePath, suffixPriorityMap);
	volumeInstance.initVolume();
	sqlite3_exec(p.db, "COMMIT;", nullptr, nullptr, nullptr);
	sqlite3_close(p.db);
#ifdef TEST
	cout << "path : " << p.disk << endl;
	cout << "Initialize done " << p.disk << endl;
#endif
}

inline bool initPriorityMap(PriorityMap& priority_map, const char* priorityDbPath)
{
	char* error;
	char** pResult;
	int row, column;
	sqlite3* cacheDb;
	const string sql = "select * from priority;";
	sqlite3_open(priorityDbPath, &cacheDb);
	const size_t ret = sqlite3_get_table(cacheDb, sql.c_str(), &pResult, &row, &column, &error);
	if (ret != SQLITE_OK)
	{
		cerr << "error init priority map" << error << endl;
		sqlite3_free(error);
		return false;
	}
	//由File-Engine保证result不为空
	auto i = 2;
	for (const auto total = column * row + 2; i < total; i += 2)
	{
		const string suffix(pResult[i]);
		const string priorityVal(pResult[i + 1]);
		auto pairPriority = pair<string, int>{ suffix, stoi(priorityVal) };
		priority_map.insert(pairPriority);
	}
	sqlite3_free_table(pResult);
	sqlite3_close(cacheDb);
	return true;
}

void splitString(char* str, vector<string>& vec) {
	char* remainDisk = nullptr;
	char diskPath[5000];
	strcpy_s(diskPath, str);
	char* _diskPath = diskPath;
	char* p = strtok_s(_diskPath, ",", &remainDisk);
	if (p != nullptr) {
		vec.emplace_back(p);
	}
	while (p != nullptr) {
		p = strtok_s(nullptr, ",", &remainDisk);
		if (p != nullptr) {
			vec.emplace_back(p);
		}
	}
}

int main() {
	char diskPath[1000];
	char output[1000];
	char ignorePath[1000];

	vector<string> diskVec;
	vector<string> ignorePathsVec;

	ifstream input("MFTSearchInfo.dat", ios::in);
	input.getline(diskPath, 1000);
	input.getline(output, 1000);
	input.getline(ignorePath, 1000);
	input.close();

	diskVec.reserve(26);

	sqlite3_config(SQLITE_CONFIG_MULTITHREAD);
	sqlite3_config(SQLITE_CONFIG_MEMSTATUS, 0);

	splitString(diskPath, diskVec);
	splitString(ignorePath, ignorePathsVec);

	bool isPriorityMapInitialized = false;
	vector<thread> threads;
	// 创建线程
	for (auto& iter : diskVec)
	{
		const auto disk = iter[0];
		if ('A' <= disk && disk <= 'Z') {
			parameter p;
			p.disk = disk;
			p.ignorePath = ignorePathsVec;
			char tmpDbPath[1000];
			strcpy_s(tmpDbPath, output);
			strcat_s(tmpDbPath, "\\");
			const size_t length = strlen(tmpDbPath);
			tmpDbPath[length] = disk;
			tmpDbPath[length + 1] = '\0';
			strcat_s(tmpDbPath, ".db");
			const size_t ret = sqlite3_open(tmpDbPath, &p.db);
#ifdef TEST
			cout << "database path: " << tmpDbPath << endl;
#endif
			if (SQLITE_OK != ret) {
				cout << "error opening database" << endl;
				return 0;
			}
			tmpDbPath[strlen(tmpDbPath) - 4] = '\0';
			strcat_s(tmpDbPath, "cache.db");

			if (!isPriorityMapInitialized)
			{
				isPriorityMapInitialized = true;
				initPriorityMap(suffixPriorityMap, tmpDbPath);
			}
			sqlite3_exec(p.db, "PRAGMA TEMP_STORE=MEMORY;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA journal_mode=OFF;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA cache_size=262144;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA page_size=65535;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA auto_vacuum=0;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA mmap_size=4096;", nullptr, nullptr, nullptr);
			// initUSN(p);
			threads.emplace_back(thread(initUSN, p));
		}
	}

	for (auto& each_thread : threads)
	{
		each_thread.join();
	}
	return 0;
}

