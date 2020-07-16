#pragma once

#include <iostream>
#include <unordered_map>
#include <Winioctl.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>
#include "sqlite3.h"

using namespace std;

constexpr auto MAXVOL = 3;
static mutex mutex_lock;

std::string to_utf8(const std::wstring& str);
std::string to_utf8(const wchar_t* buffer, int len);

typedef struct _pfrn_name {
	DWORDLONG pfrn = 0;
	CString filename;
}Pfrn_Name;

typedef unordered_map<DWORDLONG, Pfrn_Name> Frn_Pfrn_Name_Map;


std::string to_utf8(const wchar_t* buffer, int len)
{
	int nChars = ::WideCharToMultiByte(
		CP_UTF8,
		0,
		buffer,
		len,
		NULL,
		0,
		NULL,
		NULL);
	if (nChars == 0)
	{
		return "";
	}
	string newbuffer;
	newbuffer.resize(nChars);
	::WideCharToMultiByte(
		CP_UTF8,
		0,
		buffer,
		len,
		const_cast<char*>(newbuffer.c_str()),
		nChars,
		NULL,
		NULL);

	return newbuffer;
}

std::string to_utf8(const std::wstring& str)
{
	return to_utf8(str.c_str(), (int)str.size());
}


class Volume {
public:
	Volume(char vol, sqlite3* database, vector<string> ignorePaths) {
		this->vol = vol;
		hVol = NULL;
		path = "";
		db = database;
		addIgnorePath(ignorePaths);
	}
	~Volume() {
	}

	char getPath() {
		return vol;
	}

	bool initVolume() {
		if (
			// 2.获取驱动盘句柄
			getHandle() &&
			// 3.创建USN日志
			createUSN() &&
			// 4.获取USN日志信息
			getUSNInfo() &&
			// 5.获取 USN Journal 文件的基本信息
			getUSNJournal() &&
			// 06. 删除 USN 日志文件 ( 也可以不删除 ) 
			deleteUSN()) {
			mutex_lock.lock();
			initAllPrepareStatement();
			int ascii = 0;
			wstring name;
			CString path;
			wstring record;
			Frn_Pfrn_Name_Map::iterator endIter = frnPfrnNameMap.end();
			for (Frn_Pfrn_Name_Map::iterator iter = frnPfrnNameMap.begin(); iter != endIter; ++iter) {
				name = iter->second.filename;
				ascii = getAscIISum(to_utf8(name));
				path = L"\0";
				getPath(iter->first, path);
				record = vol + path;
				string fullPath = to_utf8(record);
				if (!(isIgnore(fullPath))) {
					saveResult(fullPath, ascii);
				}
			}
			finalizeAllStatement();
			mutex_lock.unlock();
			return true;
		}
		else {
			return false;
		}
	}

private:
	char vol;
	HANDLE hVol;
	Pfrn_Name pfrnName;
	Frn_Pfrn_Name_Map frnPfrnNameMap;
	sqlite3* db;
	CString path;
	sqlite3_stmt* stmt0 = NULL;
	sqlite3_stmt* stmt1 = NULL;
	sqlite3_stmt* stmt2 = NULL;
	sqlite3_stmt* stmt3 = NULL;
	sqlite3_stmt* stmt4 = NULL;
	sqlite3_stmt* stmt5 = NULL;
	sqlite3_stmt* stmt6 = NULL;
	sqlite3_stmt* stmt7 = NULL;
	sqlite3_stmt* stmt8 = NULL;
	sqlite3_stmt* stmt9 = NULL;
	sqlite3_stmt* stmt10 = NULL;
	sqlite3_stmt* stmt11 = NULL;
	sqlite3_stmt* stmt12 = NULL;
	sqlite3_stmt* stmt13 = NULL;
	sqlite3_stmt* stmt14 = NULL;
	sqlite3_stmt* stmt15 = NULL;
	sqlite3_stmt* stmt16 = NULL;
	sqlite3_stmt* stmt17 = NULL;
	sqlite3_stmt* stmt18 = NULL;
	sqlite3_stmt* stmt19 = NULL;
	sqlite3_stmt* stmt20 = NULL;
	sqlite3_stmt* stmt21 = NULL;
	sqlite3_stmt* stmt22 = NULL;
	sqlite3_stmt* stmt23 = NULL;
	sqlite3_stmt* stmt24 = NULL;
	sqlite3_stmt* stmt25 = NULL;
	sqlite3_stmt* stmt26 = NULL;
	sqlite3_stmt* stmt27 = NULL;
	sqlite3_stmt* stmt28 = NULL;
	sqlite3_stmt* stmt29 = NULL;
	sqlite3_stmt* stmt30 = NULL;
	sqlite3_stmt* stmt31 = NULL;
	sqlite3_stmt* stmt32 = NULL;
	sqlite3_stmt* stmt33 = NULL;
	sqlite3_stmt* stmt34 = NULL;
	sqlite3_stmt* stmt35 = NULL;
	sqlite3_stmt* stmt36 = NULL;
	sqlite3_stmt* stmt37 = NULL;
	sqlite3_stmt* stmt38 = NULL;
	sqlite3_stmt* stmt39 = NULL;
	sqlite3_stmt* stmt40 = NULL;

	USN_JOURNAL_DATA ujd;
	CREATE_USN_JOURNAL_DATA cujd;

	vector<string> ignorePathVector;

	bool getHandle();
	bool createUSN();
	bool getUSNInfo();
	bool getUSNJournal();
	bool deleteUSN();
	void saveResult(string path, int ascII);
	void getPath(DWORDLONG frn, CString& path);
	int getAscIISum(string name);
	bool isIgnore(string path);
	void finalizeAllStatement();
	void saveSingleRecordToDB(sqlite3_stmt* stmt, string record);
	void addIgnorePath(vector<string> vec) {
		ignorePathVector = vec;
	}
	void initAllPrepareStatement();
	void initSinglePrepareStatement(sqlite3_stmt** statement, const char* init);
};

void Volume::initSinglePrepareStatement(sqlite3_stmt** statement, const char* init) {
	size_t ret = sqlite3_prepare_v2(db, init, strlen(init), statement, 0);
	if (SQLITE_OK != ret) {
		cout << "error preparing stmt \"" << init << "\"" << endl;
	}
}

void Volume::finalizeAllStatement() {
	sqlite3_finalize(stmt0);
	sqlite3_finalize(stmt1);
	sqlite3_finalize(stmt2);
	sqlite3_finalize(stmt3);
	sqlite3_finalize(stmt4);
	sqlite3_finalize(stmt5);
	sqlite3_finalize(stmt6);
	sqlite3_finalize(stmt7);
	sqlite3_finalize(stmt8);
	sqlite3_finalize(stmt9);
	sqlite3_finalize(stmt10);
	sqlite3_finalize(stmt11);
	sqlite3_finalize(stmt12);
	sqlite3_finalize(stmt13);
	sqlite3_finalize(stmt14);
	sqlite3_finalize(stmt15);
	sqlite3_finalize(stmt16);
	sqlite3_finalize(stmt17);
	sqlite3_finalize(stmt18);
	sqlite3_finalize(stmt19);
	sqlite3_finalize(stmt20);
	sqlite3_finalize(stmt21);
	sqlite3_finalize(stmt22);
	sqlite3_finalize(stmt23);
	sqlite3_finalize(stmt24);
	sqlite3_finalize(stmt25);
	sqlite3_finalize(stmt26);
	sqlite3_finalize(stmt27);
	sqlite3_finalize(stmt28);
	sqlite3_finalize(stmt29);
	sqlite3_finalize(stmt30);
	sqlite3_finalize(stmt31);
	sqlite3_finalize(stmt32);
	sqlite3_finalize(stmt33);
	sqlite3_finalize(stmt34);
	sqlite3_finalize(stmt35);
	sqlite3_finalize(stmt36);
	sqlite3_finalize(stmt37);
	sqlite3_finalize(stmt38);
	sqlite3_finalize(stmt39);
	sqlite3_finalize(stmt40);
}

void Volume::saveSingleRecordToDB(sqlite3_stmt* stmt, string record) {
	sqlite3_reset(stmt);
	sqlite3_bind_text(stmt, 1, record.c_str(), -1, SQLITE_STATIC);
	sqlite3_step(stmt);
}

void Volume::initAllPrepareStatement() {
	initSinglePrepareStatement(&stmt0, "INSERT INTO list0(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt1, "INSERT INTO list1(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt2, "INSERT INTO list2(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt3, "INSERT INTO list3(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt4, "INSERT INTO list4(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt5, "INSERT INTO list5(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt6, "INSERT INTO list6(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt7, "INSERT INTO list7(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt8, "INSERT INTO list8(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt9, "INSERT INTO list9(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt10, "INSERT INTO list10(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt11, "INSERT INTO list11(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt12, "INSERT INTO list12(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt13, "INSERT INTO list13(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt14, "INSERT INTO list14(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt15, "INSERT INTO list15(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt16, "INSERT INTO list16(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt17, "INSERT INTO list17(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt18, "INSERT INTO list18(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt19, "INSERT INTO list19(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt20, "INSERT INTO list20(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt21, "INSERT INTO list21(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt22, "INSERT INTO list22(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt23, "INSERT INTO list23(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt24, "INSERT INTO list24(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt25, "INSERT INTO list25(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt26, "INSERT INTO list26(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt27, "INSERT INTO list27(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt28, "INSERT INTO list28(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt29, "INSERT INTO list29(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt30, "INSERT INTO list30(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt31, "INSERT INTO list31(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt32, "INSERT INTO list32(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt33, "INSERT INTO list33(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt34, "INSERT INTO list34(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt35, "INSERT INTO list35(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt36, "INSERT INTO list36(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt37, "INSERT INTO list37(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt38, "INSERT INTO list38(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt39, "INSERT INTO list39(PATH) VALUES(?);");
	initSinglePrepareStatement(&stmt40, "INSERT INTO list40(PATH) VALUES(?);");
}

bool Volume::isIgnore(string path) {
	if (path.find("$") != string::npos)
	{
		return true;
	}
	transform(path.begin(), path.end(), path.begin(), ::tolower);
	size_t size = ignorePathVector.size();
	for (int i = 0; i < size; i++)
	{
		if (path.find(ignorePathVector[i]) != string::npos)
		{
			return true;
		}
	}
	return false;
}

void Volume::saveResult(string path, int ascII) {
	int asciiGroup = ascII / 100;
	switch (asciiGroup)
	{
	case 0:
		saveSingleRecordToDB(stmt0, path);
		break;
	case 1:
		saveSingleRecordToDB(stmt1, path);
		break;
	case 2:
		saveSingleRecordToDB(stmt2, path);
		break;
	case 3:
		saveSingleRecordToDB(stmt3, path);
		break;
	case 4:
		saveSingleRecordToDB(stmt4, path);
		break;
	case 5:
		saveSingleRecordToDB(stmt5, path);
		break;
	case 6:
		saveSingleRecordToDB(stmt6, path);
		break;
	case 7:
		saveSingleRecordToDB(stmt7, path);
		break;
	case 8:
		saveSingleRecordToDB(stmt8, path);
		break;
	case 9:
		saveSingleRecordToDB(stmt9, path);
		break;
	case 10:
		saveSingleRecordToDB(stmt10, path);
		break;
	case 11:
		saveSingleRecordToDB(stmt11, path);
		break;
	case 12:
		saveSingleRecordToDB(stmt12, path);
		break;
	case 13:
		saveSingleRecordToDB(stmt13, path);
		break;
	case 14:
		saveSingleRecordToDB(stmt14, path);
		break;
	case 15:
		saveSingleRecordToDB(stmt15, path);
		break;
	case 16:
		saveSingleRecordToDB(stmt16, path);
		break;
	case 17:
		saveSingleRecordToDB(stmt17, path);
		break;
	case 18:
		saveSingleRecordToDB(stmt18, path);
		break;
	case 19:
		saveSingleRecordToDB(stmt19, path);
		break;
	case 20:
		saveSingleRecordToDB(stmt20, path);
		break;
	case 21:
		saveSingleRecordToDB(stmt21, path);
		break;
	case 22:
		saveSingleRecordToDB(stmt22, path);
		break;
	case 23:
		saveSingleRecordToDB(stmt23, path);
		break;
	case 24:
		saveSingleRecordToDB(stmt24, path);
		break;
	case 25:
		saveSingleRecordToDB(stmt25, path);
		break;
	case 26:
		saveSingleRecordToDB(stmt26, path);
		break;
	case 27:
		saveSingleRecordToDB(stmt27, path);
		break;
	case 28:
		saveSingleRecordToDB(stmt28, path);
		break;
	case 29:
		saveSingleRecordToDB(stmt29, path);
		break;
	case 30:
		saveSingleRecordToDB(stmt30, path);
		break;
	case 31:
		saveSingleRecordToDB(stmt31, path);
		break;
	case 32:
		saveSingleRecordToDB(stmt32, path);
		break;
	case 33:
		saveSingleRecordToDB(stmt33, path);
		break;
	case 34:
		saveSingleRecordToDB(stmt34, path);
		break;
	case 35:
		saveSingleRecordToDB(stmt35, path);
		break;
	case 36:
		saveSingleRecordToDB(stmt36, path);
		break;
	case 37:
		saveSingleRecordToDB(stmt37, path);
		break;
	case 38:
		saveSingleRecordToDB(stmt38, path);
		break;
	case 39:
		saveSingleRecordToDB(stmt39, path);
		break;
	case 40:
		saveSingleRecordToDB(stmt40, path);
		break;
	default:
		break;
	}
}

int Volume::getAscIISum(string name) {
	int sum = 0;
	size_t length = name.length();
	for (size_t i = 0; i < length; i++)
	{
		if (name[i] > 0)
		{
			sum += name[i];
		}
	}
	return sum;
}

void Volume::getPath(DWORDLONG frn, CString& path) {
	Frn_Pfrn_Name_Map::iterator it;
	Frn_Pfrn_Name_Map::iterator end = frnPfrnNameMap.end();
	while (true) {
		it = frnPfrnNameMap.find(frn);
		if (it == end) {
			//path = path.Right(path.GetLength() - 1);
			path = L":" + path;
			return;
		}
		path = _T("\\") + it->second.filename + path;
		frn = it->second.pfrn;
	}
}

bool Volume::getHandle() {
	// 为\\.\C:的形式
	CString lpFileName(_T("\\\\.\\c:"));
	lpFileName.SetAt(4, vol);


	hVol = CreateFile(lpFileName,
		GENERIC_READ | GENERIC_WRITE, // 可以为0
		FILE_SHARE_READ | FILE_SHARE_WRITE, // 必须包含有FILE_SHARE_WRITE
		NULL,
		OPEN_EXISTING, // 必须包含OPEN_EXISTING, CREATE_ALWAYS可能会导致错误
		FILE_ATTRIBUTE_READONLY, // FILE_ATTRIBUTE_NORMAL可能会导致错误
		NULL);


	if (INVALID_HANDLE_VALUE != hVol) {
		return true;
	}
	else {
		return false;
		//		exit(1);
		//MessageBox(NULL, _T("USN错误"), _T("错误"), MB_OK);
	}
}

bool Volume::createUSN() {
	cujd.MaximumSize = 0; // 0表示使用默认值  
	cujd.AllocationDelta = 0; // 0表示使用默认值

	DWORD br;
	if (
		DeviceIoControl(hVol,// handle to volume
			FSCTL_CREATE_USN_JOURNAL,      // dwIoControlCode
			&cujd,           // input buffer
			sizeof(cujd),         // size of input buffer
			NULL,                          // lpOutBuffer
			0,                             // nOutBufferSize
			&br,     // number of bytes returned
			NULL) // OVERLAPPED structure	
		) {
		return true;
	}
	else {
		return false;
	}
}


bool Volume::getUSNInfo() {
	DWORD br;
	if (
		DeviceIoControl(hVol, // handle to volume
			FSCTL_QUERY_USN_JOURNAL,// dwIoControlCode
			NULL,            // lpInBuffer
			0,               // nInBufferSize
			&ujd,     // output buffer
			sizeof(ujd),  // size of output buffer
			&br, // number of bytes returned
			NULL) // OVERLAPPED structure
		) {
		return true;
	}
	else {
		return false;
	}
}

bool Volume::getUSNJournal() {
	MFT_ENUM_DATA med;
	med.StartFileReferenceNumber = 0;
	med.LowUsn = ujd.FirstUsn;
	med.HighUsn = ujd.NextUsn;

	// 根目录
	CString tmp(_T("C:"));
	tmp.SetAt(0, vol);
	frnPfrnNameMap[0x20000000000005].filename = tmp;
	frnPfrnNameMap[0x20000000000005].pfrn = 0;

	constexpr auto BUF_LEN = 0x3900;	// 尽可能地大，提高效率;

	CHAR Buffer[BUF_LEN];
	DWORD usnDataSize;
	PUSN_RECORD UsnRecord;
	int USN_counter = 0;
	wstring fileName;

	while (0 != DeviceIoControl(hVol,
		FSCTL_ENUM_USN_DATA,
		&med,
		sizeof(med),
		Buffer,
		BUF_LEN,
		&usnDataSize,
		NULL))
	{

		DWORD dwRetBytes = usnDataSize - sizeof(USN);
		// 找到第一个 USN 记录  
		UsnRecord = (PUSN_RECORD)(((PCHAR)Buffer) + sizeof(USN));

		while (dwRetBytes > 0) {
			// 获取到的信息  	
			CString CfileName(UsnRecord->FileName, UsnRecord->FileNameLength / 2);
			fileName = CfileName;

			pfrnName.filename = CfileName;
			pfrnName.pfrn = UsnRecord->ParentFileReferenceNumber; 

			frnPfrnNameMap[UsnRecord->FileReferenceNumber] = pfrnName;
			// 获取下一个记录  
			DWORD recordLen = UsnRecord->RecordLength;
			dwRetBytes -= recordLen;
			UsnRecord = (PUSN_RECORD)(((PCHAR)UsnRecord) + recordLen);
		}
		// 获取下一页数据 
		med.StartFileReferenceNumber = *(USN*)&Buffer;
	}
	return true;
}

bool Volume::deleteUSN() {
	DELETE_USN_JOURNAL_DATA dujd;
	dujd.UsnJournalID = ujd.UsnJournalID;
	dujd.DeleteFlags = USN_DELETE_FLAG_DELETE;
	DWORD br;

	if (DeviceIoControl(hVol,
		FSCTL_DELETE_USN_JOURNAL,
		&dujd,
		sizeof(dujd),
		NULL,
		0,
		&br,
		NULL)
		) {
		CloseHandle(hVol);
		return true;
	}
	else {
		CloseHandle(hVol);
		return false;
	}
}