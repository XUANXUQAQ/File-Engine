#include <iostream>
#include "stdafx.h"
#include "Volume.h"
#include <fstream>
#include <thread>
//#define TEST

typedef struct{
	char disk;
	vector<string> ignorePath;
} parameter;

sqlite3* db;
static volatile UINT tasksFinished = 0;
static volatile UINT totalTasks = 0;

void initUSN(parameter p);
void splitString(char* str, vector<string>& vec);


void initUSN(parameter p) {
	const auto ret = (65 <= p.disk && p.disk <= 90) || (97 <= p.disk && p.disk <= 122);
	if (ret) {
		Volume volume(p.disk, db, p.ignorePath);
		volume.initVolume();
		tasksFinished++;
#ifdef TEST
		cout << "path : " << p.disk << endl;
		cout << "Initialize done " << p.disk << endl;
#endif
	}
}

void splitString(char* str, vector<string>& vec) {
	char* _diskPath;
	char* remainDisk;
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

	size_t ret = sqlite3_open(output, &db);
	if (SQLITE_OK != ret) {
		cout << "error opening database" << endl;
		return 0;
	}

	sqlite3_exec(db, "PRAGMA TEMP_STORE=MEMORY;", nullptr, nullptr, nullptr);
	sqlite3_exec(db, "PRAGMA journal_mode=OFF;", nullptr, nullptr, nullptr);
	sqlite3_exec(db, "PRAGMA cache_size=262144;", nullptr, nullptr, nullptr);
	sqlite3_exec(db, "PRAGMA page_size=65535;", nullptr, nullptr, nullptr);
	sqlite3_exec(db, "PRAGMA auto_vacuum=0;", nullptr, nullptr, nullptr);
	sqlite3_exec(db, "PRAGMA mmap_size=4096;", nullptr, nullptr, nullptr);

	sqlite3_exec(db, "BEGIN;", nullptr, nullptr, nullptr);

	splitString(diskPath, diskVec);
	splitString(ignorePath, ignorePathsVec);

	for (vector<string>::iterator iter = diskVec.begin(); iter != diskVec.end(); iter++) {
		const auto disk = (*iter)[0];
		if (65 <= disk && disk <= 90 || 97 <= disk && disk <= 122) {
			parameter p;
			p.disk = disk;
			p.ignorePath = ignorePathsVec;
			thread t(initUSN, p);
			totalTasks++;
			t.detach();
		}
	}

	while (tasksFinished < totalTasks) {
		Sleep(10);
	}
	sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);
	sqlite3_close(db);
	return 0;
}

