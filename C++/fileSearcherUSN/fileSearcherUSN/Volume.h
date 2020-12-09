#pragma once

#include <iostream>
#include <unordered_map>
#include <winioctl.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>
#include "sqlite3.h"
//#define TEST

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


inline std::string to_utf8(const wchar_t* buffer, int len)
{
	int nChars = ::WideCharToMultiByte(
		CP_UTF8,
		0,
		buffer,
		len,
		nullptr,
		0,
		nullptr,
		nullptr);
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
		nullptr,
		nullptr);

	return newbuffer;
}

inline std::string to_utf8(const std::wstring& str)
{
	return to_utf8(str.c_str(), (int)str.size());
}


class Volume {
public:
	Volume(char vol, sqlite3* database, vector<string> ignorePaths) {
		this->vol = vol;
		hVol = nullptr;
		path = "";
		db = database;
		addIgnorePath(ignorePaths);
	}
	~Volume() = default;

	char getPath() const
	{
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
			int ascii;
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
		return false;
	}

private:
	char vol;
	HANDLE hVol;
	Pfrn_Name pfrnName;
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

	USN_JOURNAL_DATA ujd;
	CREATE_USN_JOURNAL_DATA cujd;

	vector<string> ignorePathVector;

	bool getHandle();
	bool createUSN();
	bool getUSNInfo();
	bool getUSNJournal();
	bool deleteUSN() const;
	void saveResult(string path, int ascII) const;
	void getPath(DWORDLONG frn, CString& path);
	static int getAscIISum(string name);
	bool isIgnore(string path);
	void finalizeAllStatement() const;
	static void saveSingleRecordToDB(sqlite3_stmt* stmt, string record, int ascii);
	void addIgnorePath(vector<string> vec) {
		ignorePathVector = vec;
	}
	void initAllPrepareStatement();
	void initSinglePrepareStatement(sqlite3_stmt** statement, const char* init) const;
};

inline void Volume::initSinglePrepareStatement(sqlite3_stmt** statement, const char* init) const
{
	size_t ret = sqlite3_prepare_v2(db, init, strlen(init), statement, 0);
	if (SQLITE_OK != ret) {
		cout << "error preparing stmt \"" << init << "\"" << endl;
	}
}

inline void Volume::finalizeAllStatement() const
{
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

inline void Volume::saveSingleRecordToDB(sqlite3_stmt* stmt, const string record, const int ascii) {
	sqlite3_reset(stmt);
	sqlite3_bind_int(stmt, 1, ascii);
	sqlite3_bind_text(stmt, 2, record.c_str(), -1, SQLITE_STATIC);
	sqlite3_step(stmt);
}

inline void Volume::initAllPrepareStatement() {
	initSinglePrepareStatement(&stmt0, "INSERT INTO list0 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt1, "INSERT INTO list1 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt2, "INSERT INTO list2 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt3, "INSERT INTO list3 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt4, "INSERT INTO list4 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt5, "INSERT INTO list5 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt6, "INSERT INTO list6 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt7, "INSERT INTO list7 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt8, "INSERT INTO list8 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt9, "INSERT INTO list9 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt10, "INSERT INTO list10 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt11, "INSERT INTO list11 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt12, "INSERT INTO list12 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt13, "INSERT INTO list13 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt14, "INSERT INTO list14 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt15, "INSERT INTO list15 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt16, "INSERT INTO list16 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt17, "INSERT INTO list17 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt18, "INSERT INTO list18 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt19, "INSERT INTO list19 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt20, "INSERT INTO list20 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt21, "INSERT INTO list21 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt22, "INSERT INTO list22 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt23, "INSERT INTO list23 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt24, "INSERT INTO list24 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt25, "INSERT INTO list25 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt26, "INSERT INTO list26 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt27, "INSERT INTO list27 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt28, "INSERT INTO list28 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt29, "INSERT INTO list29 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt30, "INSERT INTO list30 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt31, "INSERT INTO list31 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt32, "INSERT INTO list32 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt33, "INSERT INTO list33 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt34, "INSERT INTO list34 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt35, "INSERT INTO list35 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt36, "INSERT INTO list36 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt37, "INSERT INTO list37 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt38, "INSERT INTO list38 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt39, "INSERT INTO list39 VALUES(?, ?);");
	initSinglePrepareStatement(&stmt40, "INSERT INTO list40 VALUES(?, ?);");
}

inline bool Volume::isIgnore(string path) {
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

inline void Volume::saveResult(string path, int ascII) const
{
#ifdef TEST
	cout << "path = " << path << endl;
#endif
	const int asciiGroup = ascII / 100;
	switch (asciiGroup)
	{
	case 0:
		saveSingleRecordToDB(stmt0, path, ascII);
		break;
	case 1:
		saveSingleRecordToDB(stmt1, path, ascII);
		break;
	case 2:
		saveSingleRecordToDB(stmt2, path, ascII);
		break;
	case 3:
		saveSingleRecordToDB(stmt3, path, ascII);
		break;
	case 4:
		saveSingleRecordToDB(stmt4, path, ascII);
		break;
	case 5:
		saveSingleRecordToDB(stmt5, path, ascII);
		break;
	case 6:
		saveSingleRecordToDB(stmt6, path, ascII);
		break;
	case 7:
		saveSingleRecordToDB(stmt7, path, ascII);
		break;
	case 8:
		saveSingleRecordToDB(stmt8, path, ascII);
		break;
	case 9:
		saveSingleRecordToDB(stmt9, path, ascII);
		break;
	case 10:
		saveSingleRecordToDB(stmt10, path, ascII);
		break;
	case 11:
		saveSingleRecordToDB(stmt11, path, ascII);
		break;
	case 12:
		saveSingleRecordToDB(stmt12, path, ascII);
		break;
	case 13:
		saveSingleRecordToDB(stmt13, path, ascII);
		break;
	case 14:
		saveSingleRecordToDB(stmt14, path, ascII);
		break;
	case 15:
		saveSingleRecordToDB(stmt15, path, ascII);
		break;
	case 16:
		saveSingleRecordToDB(stmt16, path, ascII);
		break;
	case 17:
		saveSingleRecordToDB(stmt17, path, ascII);
		break;
	case 18:
		saveSingleRecordToDB(stmt18, path, ascII);
		break;
	case 19:
		saveSingleRecordToDB(stmt19, path, ascII);
		break;
	case 20:
		saveSingleRecordToDB(stmt20, path, ascII);
		break;
	case 21:
		saveSingleRecordToDB(stmt21, path, ascII);
		break;
	case 22:
		saveSingleRecordToDB(stmt22, path, ascII);
		break;
	case 23:
		saveSingleRecordToDB(stmt23, path, ascII);
		break;
	case 24:
		saveSingleRecordToDB(stmt24, path, ascII);
		break;
	case 25:
		saveSingleRecordToDB(stmt25, path, ascII);
		break;
	case 26:
		saveSingleRecordToDB(stmt26, path, ascII);
		break;
	case 27:
		saveSingleRecordToDB(stmt27, path, ascII);
		break;
	case 28:
		saveSingleRecordToDB(stmt28, path, ascII);
		break;
	case 29:
		saveSingleRecordToDB(stmt29, path, ascII);
		break;
	case 30:
		saveSingleRecordToDB(stmt30, path, ascII);
		break;
	case 31:
		saveSingleRecordToDB(stmt31, path, ascII);
		break;
	case 32:
		saveSingleRecordToDB(stmt32, path, ascII);
		break;
	case 33:
		saveSingleRecordToDB(stmt33, path, ascII);
		break;
	case 34:
		saveSingleRecordToDB(stmt34, path, ascII);
		break;
	case 35:
		saveSingleRecordToDB(stmt35, path, ascII);
		break;
	case 36:
		saveSingleRecordToDB(stmt36, path, ascII);
		break;
	case 37:
		saveSingleRecordToDB(stmt37, path, ascII);
		break;
	case 38:
		saveSingleRecordToDB(stmt38, path, ascII);
		break;
	case 39:
		saveSingleRecordToDB(stmt39, path, ascII);
		break;
	case 40:
		saveSingleRecordToDB(stmt40, path, ascII);
		break;
	default:
		break;
	}
}

inline int Volume::getAscIISum(string name) {
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

inline void Volume::getPath(DWORDLONG frn, CString& path) {
	Frn_Pfrn_Name_Map::iterator it;
	const auto end = frnPfrnNameMap.end();
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

inline bool Volume::getHandle() {
	// 为\\.\C:的形式
	CString lpFileName(_T("\\\\.\\c:"));
	lpFileName.SetAt(4, vol);


	hVol = CreateFile(lpFileName,
		GENERIC_READ | GENERIC_WRITE, // 可以为0
		FILE_SHARE_READ | FILE_SHARE_WRITE, // 必须包含有FILE_SHARE_WRITE
		nullptr,
		OPEN_EXISTING, // 必须包含OPEN_EXISTING, CREATE_ALWAYS可能会导致错误
		FILE_ATTRIBUTE_READONLY, // FILE_ATTRIBUTE_NORMAL可能会导致错误
		nullptr);


	if (INVALID_HANDLE_VALUE != hVol) {
		return true;
	}
	
	return false;
}

inline bool Volume::createUSN() {
	cujd.MaximumSize = 0; // 0表示使用默认值  
	cujd.AllocationDelta = 0; // 0表示使用默认值

	DWORD br;
	if (
		DeviceIoControl(hVol,// handle to volume
			FSCTL_CREATE_USN_JOURNAL,      // dwIoControlCode
			&cujd,           // input buffer
			sizeof(cujd),         // size of input buffer
			nullptr,                          // lpOutBuffer
			0,                             // nOutBufferSize
			&br,     // number of bytes returned
			nullptr) // OVERLAPPED structure	
		) {
		return true;
	}
	return false;
}


inline bool Volume::getUSNInfo() {
	DWORD br;
	if (
		DeviceIoControl(hVol, // handle to volume
			FSCTL_QUERY_USN_JOURNAL,// dwIoControlCode
			nullptr,            // lpInBuffer
			0,               // nInBufferSize
			&ujd,     // output buffer
			sizeof(ujd),  // size of output buffer
			&br, // number of bytes returned
			nullptr) // OVERLAPPED structure
		) {
		return true;
	}
	else {
		return false;
	}
}

inline bool Volume::getUSNJournal() {
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
		PUSN_RECORD UsnRecord = reinterpret_cast<PUSN_RECORD>(static_cast<PCHAR>(Buffer) + sizeof(USN));

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

inline bool Volume::deleteUSN() const
{
	DELETE_USN_JOURNAL_DATA dujd;
	dujd.UsnJournalID = ujd.UsnJournalID;
	dujd.DeleteFlags = USN_DELETE_FLAG_DELETE;
	DWORD br;

	if (DeviceIoControl(hVol,
		FSCTL_DELETE_USN_JOURNAL,
		&dujd,
		sizeof(dujd),
		nullptr,
		0,
		&br,
		nullptr)
		) {
		CloseHandle(hVol);
		return true;
	}
	else {
		CloseHandle(hVol);
		return false;
	}
}
