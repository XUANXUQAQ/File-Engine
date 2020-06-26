#pragma once

#include <iostream>
#include <unordered_map>
#include <Winioctl.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "sqlite3.h"

//#define TEST
constexpr auto MAXVOL = 3;

using namespace std;

std::string to_utf8(const std::wstring& str);
std::string to_utf8(const wchar_t* buffer, int len);

typedef struct _pfrn_name {
	DWORDLONG pfrn;
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
	Volume(char vol, sqlite3* database) {
		this->vol = vol;
		hVol = NULL;
		path = "";
		db = database;
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
			initAllVector();
			return true;
		}
		else {
			return false;
		}
	}

	void saveToDatabase();

private:
	char vol;
	HANDLE hVol;
	Pfrn_Name pfrnName;
	Frn_Pfrn_Name_Map frnPfrnNameMap;
	sqlite3* db;
	CString path;

	USN_JOURNAL_DATA ujd;
	CREATE_USN_JOURNAL_DATA cujd;
	vector<string> ignorePathVector;

	vector<string> command0;
	vector<string> command1;
	vector<string> command2;
	vector<string> command3;
	vector<string> command4;
	vector<string> command5;
	vector<string> command6;
	vector<string> command7;
	vector<string> command8;
	vector<string> command9;
	vector<string> command10;
	vector<string> command11;
	vector<string> command12;
	vector<string> command13;
	vector<string> command14;
	vector<string> command15;
	vector<string> command16;
	vector<string> command17;
	vector<string> command18;
	vector<string> command19;
	vector<string> command20;
	vector<string> command21;
	vector<string> command22;
	vector<string> command23;
	vector<string> command24;
	vector<string> command25;

	bool getHandle();
	bool createUSN();
	bool getUSNInfo();
	bool getUSNJournal();
	bool deleteUSN();
	void executeAll(vector<string>& vec, const char* init);
	void saveResult(string path, int ascII);
	void getPath(DWORDLONG frn, CString& path);
	int getAscIISum(string name);
	void initAllVector() {
		command0.reserve(2000);
		command1.reserve(3000);
		command2.reserve(3000);
		command3.reserve(3000);
		command4.reserve(3000);
		command5.reserve(3000);
		command6.reserve(5000);
		command7.reserve(5000);
		command8.reserve(5000);
		command9.reserve(5000);
		command10.reserve(5000);
		command11.reserve(5000);
		command12.reserve(5000);
		command13.reserve(5000);
		command14.reserve(4000);
		command15.reserve(4000);
		command16.reserve(4000);
		command17.reserve(4000);
		command18.reserve(4000);
		command19.reserve(4000);
		command20.reserve(3000);
		command21.reserve(3000);
		command22.reserve(3000);
		command23.reserve(3000);
		command24.reserve(3000);
		command25.reserve(10000);
	}
};

void Volume::executeAll(vector<string>& vec, const char* init) {
	sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);
	sqlite3_stmt* stmt = NULL;
	string str;
	int rc = sqlite3_prepare_v2(db, init, strlen(init), &stmt, NULL);
	if (rc != SQLITE_OK) {
		cout << "error preparing statement" << endl;
		exit(-1);
	}
	for (vector<string>::iterator iter = vec.begin(); iter != vec.end(); ++iter) {
		str = *iter;
		sqlite3_reset(stmt);
		sqlite3_bind_text(stmt, 1, str.c_str(), -1, SQLITE_STATIC);
		sqlite3_step(stmt);
	}
	sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
}

void Volume::saveResult(string path, int ascII) {
	if (2500 <= ascII && ascII <= 4000)
	{
		command25.push_back(path);
	}
	else if (100 < ascII && ascII <= 200)
	{
		command1.push_back(path);
	}
	else if (200 < ascII && ascII <= 300)
	{
		command2.push_back(path);
	}
	else if (300 < ascII && ascII <= 400)
	{
		command3.push_back(path);
	}
	else if (400 < ascII && ascII <= 500)
	{
		command4.push_back(path);
	}
	else if (500 < ascII && ascII <= 600)
	{
		command5.push_back(path);
	}
	else if (600 < ascII && ascII <= 700)
	{
		command6.push_back(path);
	}
	else if (700 < ascII && ascII <= 800)
	{
		command7.push_back(path);
	}
	else if (800 < ascII && ascII <= 900)
	{
		command8.push_back(path);
	}
	else if (900 < ascII && ascII <= 1000)
	{
		command9.push_back(path);
	}
	else if (1000 < ascII && ascII <= 1100)
	{
		command10.push_back(path);
	}
	else if (1100 < ascII && ascII <= 1200)
	{
		command11.push_back(path);
	}
	else if (1200 < ascII && ascII <= 1300)
	{
		command12.push_back(path);
	}
	else if (1300 < ascII && ascII <= 1400)
	{
		command13.push_back(path);
	}
	else if (1400 < ascII && ascII <= 1500)
	{
		command14.push_back(path);
	}
	else if (1500 < ascII && ascII <= 1600)
	{
		command15.push_back(path);
	}
	else if (1600 < ascII && ascII <= 1700)
	{
		command16.push_back(path);
	}
	else if (1700 < ascII && ascII <= 1800)
	{
		command17.push_back(path);
	}
	else if (1800 < ascII && ascII <= 1900)
	{
		command18.push_back(path);
	}
	else if (1900 < ascII && ascII <= 2000)
	{
		command19.push_back(path);
	}
	else if (2000 < ascII && ascII <= 2100)
	{
		command20.push_back(path);
	}
	else if (2100 < ascII && ascII <= 2200)
	{
		command21.push_back(path);
	}
	else if (2200 < ascII && ascII <= 2300)
	{
		command22.push_back(path);
	}
	else if (2300 < ascII && ascII <= 2400)
	{
		command23.push_back(path);
	}
	else if (2400 < ascII && ascII <= 2500)
	{
		command24.push_back(path);
	}
	else if (0 <= ascII && ascII <= 100)
	{
		command0.push_back(path);
	}
}

void Volume::saveToDatabase() {
	int ascii = 0;
	wstring name;
	CString path;
	string _utf8;
	wstring record;
	Frn_Pfrn_Name_Map::iterator endIter = frnPfrnNameMap.end();
	for (Frn_Pfrn_Name_Map::iterator iter = frnPfrnNameMap.begin(); iter != endIter; ++iter) {		
		name = iter->second.filename;
		if (name.find(_T("$")) == wstring::npos) {
			ascii = getAscIISum(to_utf8(name));
			path = L"\0";
			getPath(iter->first, path);
			record = path;
			_utf8 = to_utf8(record);
			saveResult(_utf8, ascii);
		}
	}
	executeAll(command0, "INSERT INTO list0(PATH) VALUES(?);");
	executeAll(command1, "INSERT INTO list1(PATH) VALUES(?);");
	executeAll(command2, "INSERT INTO list2(PATH) VALUES(?);");
	executeAll(command3, "INSERT INTO list3(PATH) VALUES(?);");
	executeAll(command4, "INSERT INTO list4(PATH) VALUES(?);");
	executeAll(command5, "INSERT INTO list5(PATH) VALUES(?);");
	executeAll(command6, "INSERT INTO list6(PATH) VALUES(?);");
	executeAll(command7, "INSERT INTO list7(PATH) VALUES(?);");
	executeAll(command8, "INSERT INTO list8(PATH) VALUES(?);");
	executeAll(command9, "INSERT INTO list9(PATH) VALUES(?);");
	executeAll(command10, "INSERT INTO list10(PATH) VALUES(?);");
	executeAll(command11, "INSERT INTO list11(PATH) VALUES(?);");
	executeAll(command12, "INSERT INTO list12(PATH) VALUES(?);");
	executeAll(command13, "INSERT INTO list13(PATH) VALUES(?);");
	executeAll(command14, "INSERT INTO list14(PATH) VALUES(?);");
	executeAll(command15, "INSERT INTO list15(PATH) VALUES(?);");
	executeAll(command16, "INSERT INTO list16(PATH) VALUES(?);");
	executeAll(command17, "INSERT INTO list17(PATH) VALUES(?);");
	executeAll(command18, "INSERT INTO list18(PATH) VALUES(?);");
	executeAll(command19, "INSERT INTO list19(PATH) VALUES(?);");
	executeAll(command20, "INSERT INTO list20(PATH) VALUES(?);");
	executeAll(command21, "INSERT INTO list21(PATH) VALUES(?);");
	executeAll(command22, "INSERT INTO list22(PATH) VALUES(?);");
	executeAll(command23, "INSERT INTO list23(PATH) VALUES(?);");
	executeAll(command24, "INSERT INTO list24(PATH) VALUES(?);");
	executeAll(command25, "INSERT INTO list25(PATH) VALUES(?);");
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
			path = path.Right(path.GetLength() - 1);
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
	frnPfrnNameMap[0x5000000000005].filename = tmp;
	frnPfrnNameMap[0x5000000000005].pfrn = 0;

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