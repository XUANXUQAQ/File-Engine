
#include <iostream>
#include "stdafx.h"
#include "Volume.h"
#include <fstream>
#include <thread>
#include <future>
//#define TEST

sqlite3* db;
vector<Volume> volumeList;
static volatile UINT taskFinishedNum = 0;

#ifdef TEST
int main()
{
	Volume volume('C', "C:\\Users\\13927\\Desktop\\test.db");
	volume.initVolume();
	volume.saveToDatabase();
}
#else

void initUSN(char disk) {
	bool ret = (65 <= disk && disk <= 90) || (97 <= disk && disk <= 122);
	if (ret) {
		Volume volume(disk, db);
		volume.initVolume();
		volumeList.push_back(volume);
		taskFinishedNum++;
	}
}

int main() {
	char disk;
	char* p = NULL;
	char diskPath[1000];
	char output[1000];
	char* _diskPath;
	char* remainDisk;
	UINT diskCount = 0;

	ifstream input("MFTSearchInfo.dat", ios::in | ios::binary);
	input.getline(diskPath, 1000);
	input.getline(output, 1000);
	input.close();

	volumeList.reserve(26);
	
	int ret = sqlite3_open(output, &db);
	if (ret) {
		cout << "open database failed" << endl;
		exit(0);
	}
	else {
		cout << "open database successfully" << endl;
	}

	sqlite3_exec(db, "PRAGMA TEMP_STORE=MEMORY;", 0, 0, 0);
	sqlite3_exec(db, "PRAGMA journal_mode=OFF;", 0, 0, 0);
	sqlite3_exec(db, "PRAGMA page_size=4096;", 0, 0, 0);
	sqlite3_exec(db, "PRAGMA cache_size=8000;", 0, 0, 0);
	sqlite3_exec(db, "PRAGMA auto_vacuum=0;", 0, 0, 0);
	sqlite3_exec(db, "PRAGMA mmap_size=4096;", 0, 0, 0);

	_diskPath = diskPath;
	p = strtok_s(_diskPath, ",", &remainDisk);
	if (p != NULL) {
		disk = p[0];
		if (((65 <= disk) && (disk <= 90)) || ((97 <= disk) && (disk <= 122))) {
			thread t(initUSN, disk);
			t.detach();
			diskCount++;
		}
	}
	while (p != NULL) {
		p = strtok_s(NULL, ",", &remainDisk);
		if (p != NULL) {
			disk = p[0];
			if (((65 <= disk) && (disk <= 90)) || ((97 <= disk) && (disk <= 122))) {
				thread t(initUSN, disk);
				t.detach();
				diskCount++;
			}
		}
	}

	//等待线程
	while (taskFinishedNum != diskCount) {
		Sleep(1);
	}
	vector<Volume>::iterator iter = volumeList.begin();
	vector<Volume>::iterator end = volumeList.end();
	sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);
	for (; iter != end; ++iter) {
		Volume volume = *iter;
		volume.saveToDatabase();
		cout << "The search for drive " << volume.getPath() << " has completed" << endl;
	}
	sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);

	//创建索引
	sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);
	char num[5];
	string str;
	for (int i = 0; i <= 40; i++) {
		_itoa_s(i, num, 10);
		str.append("CREATE UNIQUE INDEX list").append(num).append("_index ON list").append(num).append("(PATH)").append(";");
		sqlite3_exec(db, str.c_str(), NULL, NULL, NULL);
		str.clear();
	}
	sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
	sqlite3_close(db);
	return 0;
}
#endif