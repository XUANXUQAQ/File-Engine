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

static volatile UINT tasksFinished = 0;
static volatile UINT totalTasks = 0;

typedef struct _pfrn_name {
	DWORDLONG pfrn = 0;
	CString filename;
}pfrn_name;

typedef unordered_map<string, int> PriorityMap;

typedef unordered_map<DWORDLONG, pfrn_name> Frn_Pfrn_Name_Map;

class volume {
public:
	volume(const char vol, sqlite3* database, vector<string> ignorePaths, const char* priorityDbPath) {
		this->vol = vol;
		hVol = nullptr;
		path = "";
		strcpy_s(this->priorityDbPath, priorityDbPath);
		db = database;
		addIgnorePath(ignorePaths);
	}
	~volume() = default;

	char getPath() const
	{
		return vol;
	}

	void initVolume()
	{
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
			deleteUSN()
			)
		{
			try {
				if (initPriorityMap(priorityMap))
				{
					initAllPrepareStatement();

					const auto endIter = frnPfrnNameMap.end();
					for (auto iter = frnPfrnNameMap.begin(); iter != endIter; ++iter) {
						auto name = iter->second.filename;
						const auto ascii = getAscIISum(to_utf8(wstring(name)));
						CString path = _T("\0");
						getPath(iter->first, path);
						CString record = vol + path;
						auto fullPath = to_utf8(wstring(record));
						if (!isIgnore(fullPath)) {
							saveResult(fullPath, ascii);
						}
					}
					finalizeAllStatement();
				}
			}
			catch (exception& e)
			{
				cerr << e.what() << endl;
			}
		}
	}


private:
	char vol;
	HANDLE hVol;
	pfrn_name pfrnName;
	Frn_Pfrn_Name_Map frnPfrnNameMap;
	sqlite3* db;
	CString path;
	char priorityDbPath[500];
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

	vector<string> ignorePathVector;
	PriorityMap priorityMap;

	static string to_utf8(const wchar_t* buffer, int len);

	string to_utf8(const std::wstring& str) const
	{
		return to_utf8(str.c_str(), static_cast<int>(str.size()));
	}

	bool getHandle();
	bool createUSN();
	bool getUSNInfo();
	bool getUSNJournal();
	bool deleteUSN() const;
	void saveResult(string path, int ascII);
	void getPath(DWORDLONG frn, CString& path);
	static int getAscIISum(string name);
	bool isIgnore(string path);
	void finalizeAllStatement() const;
	void saveSingleRecordToDB(sqlite3_stmt* stmt, string record, int ascii);
	void addIgnorePath(const vector<string>& vec) {
		ignorePathVector = vec;
	}
	int getPriorityBySuffix(const string& suffix);
	int getPriorityByPath(const string& path);
	bool initPriorityMap(PriorityMap& priority_map) const;
	void initAllPrepareStatement();
	void initSinglePrepareStatement(sqlite3_stmt** statement, const char* init) const;
};

inline std::string volume::to_utf8(const wchar_t* buffer, int len)
{
	const auto nChars = ::WideCharToMultiByte(
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
	string newBuffer;
	newBuffer.resize(nChars);
	::WideCharToMultiByte(
		CP_UTF8,
		0,
		buffer,
		len,
		const_cast<char*>(newBuffer.c_str()),
		nChars,
		nullptr,
		nullptr);

	return newBuffer;
}


inline int volume::getPriorityBySuffix(const string& suffix)
{
	auto iter = priorityMap.find(suffix);
	if (iter == priorityMap.end())
	{
		return getPriorityBySuffix("defaultPriority");
	}
	return iter->second;
}


inline int volume::getPriorityByPath(const string& path)
{
	auto suffix = path.substr(path.find_last_of('.') + 1);
	transform(suffix.begin(), suffix.end(), suffix.begin(), tolower);
	return getPriorityBySuffix(suffix);
}

inline bool volume::initPriorityMap(PriorityMap& priority_map) const
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
	const auto total = column * row + 2;
	for (; i < total; i += 2)
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


inline void volume::initSinglePrepareStatement(sqlite3_stmt** statement, const char* init) const
{
	const size_t ret = sqlite3_prepare_v2(db, init, static_cast<long>(strlen(init)), statement, nullptr);
	if (SQLITE_OK != ret)
	{
		cout << "error preparing stmt \"" << init << "\"" << endl;
	}
}

inline void volume::finalizeAllStatement() const
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

inline void volume::saveSingleRecordToDB(sqlite3_stmt* stmt, const string record, const int ascii) {
	sqlite3_reset(stmt);
	sqlite3_bind_int(stmt, 1, ascii);
	sqlite3_bind_text(stmt, 2, record.c_str(), -1, SQLITE_STATIC);
	sqlite3_bind_int(stmt, 3, getPriorityByPath(record));
	sqlite3_step(stmt);
}

inline void volume::initAllPrepareStatement() {
	initSinglePrepareStatement(&stmt0, "INSERT OR IGNORE INTO list0 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt1, "INSERT OR IGNORE INTO list1 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt2, "INSERT OR IGNORE INTO list2 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt3, "INSERT OR IGNORE INTO list3 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt4, "INSERT OR IGNORE INTO list4 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt5, "INSERT OR IGNORE INTO list5 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt6, "INSERT OR IGNORE INTO list6 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt7, "INSERT OR IGNORE INTO list7 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt8, "INSERT OR IGNORE INTO list8 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt9, "INSERT OR IGNORE INTO list9 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt10, "INSERT OR IGNORE INTO list10 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt11, "INSERT OR IGNORE INTO list11 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt12, "INSERT OR IGNORE INTO list12 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt13, "INSERT OR IGNORE INTO list13 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt14, "INSERT OR IGNORE INTO list14 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt15, "INSERT OR IGNORE INTO list15 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt16, "INSERT OR IGNORE INTO list16 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt17, "INSERT OR IGNORE INTO list17 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt18, "INSERT OR IGNORE INTO list18 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt19, "INSERT OR IGNORE INTO list19 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt20, "INSERT OR IGNORE INTO list20 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt21, "INSERT OR IGNORE INTO list21 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt22, "INSERT OR IGNORE INTO list22 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt23, "INSERT OR IGNORE INTO list23 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt24, "INSERT OR IGNORE INTO list24 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt25, "INSERT OR IGNORE INTO list25 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt26, "INSERT OR IGNORE INTO list26 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt27, "INSERT OR IGNORE INTO list27 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt28, "INSERT OR IGNORE INTO list28 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt29, "INSERT OR IGNORE INTO list29 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt30, "INSERT OR IGNORE INTO list30 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt31, "INSERT OR IGNORE INTO list31 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt32, "INSERT OR IGNORE INTO list32 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt33, "INSERT OR IGNORE INTO list33 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt34, "INSERT OR IGNORE INTO list34 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt35, "INSERT OR IGNORE INTO list35 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt36, "INSERT OR IGNORE INTO list36 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt37, "INSERT OR IGNORE INTO list37 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt38, "INSERT OR IGNORE INTO list38 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt39, "INSERT OR IGNORE INTO list39 VALUES(?, ?, ?);");
	initSinglePrepareStatement(&stmt40, "INSERT OR IGNORE INTO list40 VALUES(?, ?, ?);");
}

inline bool volume::isIgnore(string path) {
	if (path.find('$') != string::npos)
	{
		return true;
	}
	transform(path.begin(), path.end(), path.begin(), ::tolower);
	const auto size = ignorePathVector.size();
	for (auto i = 0; i < size; i++)
	{
		if (path.find(ignorePathVector[i]) != string::npos)
		{
			return true;
		}
	}
	return false;
}

inline void volume::saveResult(string path, const int ascII)
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

inline int volume::getAscIISum(string name) {
	auto sum = 0;
	const auto length = name.length();
	for (size_t i = 0; i < length; i++)
	{
		if (name[i] > 0)
		{
			sum += name[i];
		}
	}
	return sum;
}

inline void volume::getPath(DWORDLONG frn, CString& path) {
	Frn_Pfrn_Name_Map::iterator it;
	const auto end = frnPfrnNameMap.end();
	while (true) {
		it = frnPfrnNameMap.find(frn);
		if (it == end) {
			path = L":" + path;
			return;
		}
		path = _T("\\") + it->second.filename + path;
		frn = it->second.pfrn;
	}
}

inline bool volume::getHandle() {
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

inline bool volume::createUSN() {
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


inline bool volume::getUSNInfo() {
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

inline bool volume::getUSNJournal() {
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
		auto UsnRecord = reinterpret_cast<PUSN_RECORD>(static_cast<PCHAR>(Buffer) + sizeof(USN));

		while (dwRetBytes > 0) {
			// 获取到的信息  	
			const CString CfileName(UsnRecord->FileName, UsnRecord->FileNameLength / 2);

			pfrnName.filename = CfileName;
			pfrnName.pfrn = UsnRecord->ParentFileReferenceNumber;

			frnPfrnNameMap[UsnRecord->FileReferenceNumber] = pfrnName;
			// 获取下一个记录  
			auto recordLen = UsnRecord->RecordLength;
			dwRetBytes -= recordLen;
			UsnRecord = reinterpret_cast<PUSN_RECORD>(reinterpret_cast<PCHAR>(UsnRecord) + recordLen);
		}
		// 获取下一页数据 
		med.StartFileReferenceNumber = *(USN*)&Buffer;
	}
	return true;
}

inline bool volume::deleteUSN() const
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
