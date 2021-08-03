#include <iostream>
#include "stdafx.h"
#include "Volume.h"
#include <fstream>
#include <thread>
//#define TEST


typedef struct {
	char disk;
	vector<string> ignorePath;
	sqlite3* db;
	char priorityMapDbPath[1000];
} parameter;

void initUSN(parameter p);
void splitString(char* str, vector<string>& vec);


void initUSN(const parameter p) {
	sqlite3_exec(p.db, "BEGIN;", nullptr, nullptr, nullptr);
	volume volumeInstance(p.disk, p.db, p.ignorePath, p.priorityMapDbPath);
	volumeInstance.initVolume();
	tasksFinished++;
	sqlite3_exec(p.db, "COMMIT;", nullptr, nullptr, nullptr);
	sqlite3_close(p.db);
#ifdef TEST
	cout << "path : " << p.disk << endl;
	cout << "Initialize done " << p.disk << endl;
#endif
}

void splitString(char* str, vector<string>& vec) {
	char* _diskPath;
	char* remainDisk = nullptr;
	char* p;
	char diskPath[5000];
	strcpy_s(diskPath, str);
	_diskPath = diskPath;
	p = strtok_s(_diskPath, ",", &remainDisk);
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

	for (auto& iter : diskVec)
	{
		const auto disk = iter[0];
		if (65 <= disk && disk <= 90 || 97 <= disk && disk <= 122) {
			parameter p;
			p.disk = disk;
			p.ignorePath = ignorePathsVec;
			char tmpDbPath[1000];
			strcpy_s(tmpDbPath, output);
			strcat_s(tmpDbPath, "\\");
			int length = strlen(tmpDbPath);
			tmpDbPath[length] = disk;
			tmpDbPath[length + 1] = '\0';
			strcat_s(tmpDbPath, ".db");
			size_t ret = sqlite3_open(tmpDbPath, &p.db);
			if (SQLITE_OK != ret) {
				cout << "error opening database" << endl;
				return 0;
			}
			tmpDbPath[strlen(tmpDbPath) - 4] = '\0';
			strcat_s(tmpDbPath, "cache.db");
			strcpy_s(p.priorityMapDbPath, tmpDbPath);
			sqlite3_exec(p.db, "PRAGMA TEMP_STORE=MEMORY;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA journal_mode=OFF;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA cache_size=262144;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA page_size=65535;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA auto_vacuum=0;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA mmap_size=4096;", nullptr, nullptr, nullptr);
			totalTasks++;
			thread t(initUSN, p);
			t.detach();
		}
	}
	//等待搜索任务执行
	while (tasksFinished < totalTasks) {
		Sleep(10);
	}
	return 0;
}

