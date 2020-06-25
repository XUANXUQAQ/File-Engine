
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
	if ((65 <= disk <= 90) || (97 <= disk <= 122)) {
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

	
	int ret = sqlite3_open(output, &db);
	if (ret) {
		cout << "open database failed" << endl;
		exit(0);
	}
	else {
		cout << "open database successfully" << endl;
	}

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

	//wait for threads
	while (taskFinishedNum != diskCount) {
		Sleep(5);
	}
	sqlite3_exec(db, "PRAGMA synchronous = OFF;", NULL, NULL, NULL);
	sqlite3_exec(db, "PRAGMA SQLITE_TEMP_STORE=2;", NULL, NULL, NULL);
	vector<Volume>::iterator iter = volumeList.begin();
	vector<Volume>::iterator end = volumeList.end();
	for (; iter != end; ++iter) {
		Volume volume = *iter;
		volume.saveToDatabase();
		cout << "The search for drive " << volume.getPath() << " has completed" << endl;
	}
	sqlite3_close(db);
	return 0;
}
#endif