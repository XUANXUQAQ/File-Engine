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
			initAllVector();
			int ascii = 0;
			wstring name;
			CString path;
			string* _utf8;
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
					_utf8 = new string(fullPath);
					saveResult(_utf8, ascii);
				}
			}
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

	vector<string*> command0;
	vector<string*> command1;
	vector<string*> command2;
	vector<string*> command3;
	vector<string*> command4;
	vector<string*> command5;
	vector<string*> command6;
	vector<string*> command7;
	vector<string*> command8;
	vector<string*> command9;
	vector<string*> command10;
	vector<string*> command11;
	vector<string*> command12;
	vector<string*> command13;
	vector<string*> command14;
	vector<string*> command15;
	vector<string*> command16;
	vector<string*> command17;
	vector<string*> command18;
	vector<string*> command19;
	vector<string*> command20;
	vector<string*> command21;
	vector<string*> command22;
	vector<string*> command23;
	vector<string*> command24;
	vector<string*> command25;
	vector<string*> command26;
	vector<string*> command27;
	vector<string*> command28;
	vector<string*> command29;
	vector<string*> command30;
	vector<string*> command31;
	vector<string*> command32;
	vector<string*> command33;
	vector<string*> command34;
	vector<string*> command35;
	vector<string*> command36;
	vector<string*> command37;
	vector<string*> command38;
	vector<string*> command39;
	vector<string*> command40;

	bool getHandle();
	bool createUSN();
	bool getUSNInfo();
	bool getUSNJournal();
	bool deleteUSN();
	void executeAll(vector<string*>& vec, const char* init);
	void saveResult(string* path, int ascII);
	void getPath(DWORDLONG frn, CString& path);
	int getAscIISum(string name);
	bool isIgnore(string path);
	void addIgnorePath(vector<string> vec) {
		ignorePathVector = vec;
	}
	void initAllVector() {
		command0.reserve(3000);
		command1.reserve(50000);
		command2.reserve(50000);
		command3.reserve(50000);
		command4.reserve(50000);
		command5.reserve(50000);
		command6.reserve(50000);
		command7.reserve(50000);
		command8.reserve(50000);
		command9.reserve(50000);
		command10.reserve(50000);
		command11.reserve(50000);
		command12.reserve(50000);
		command13.reserve(50000);
		command14.reserve(50000);
		command15.reserve(50000);
		command16.reserve(50000);
		command17.reserve(50000);
		command18.reserve(50000);
		command19.reserve(50000);
		command20.reserve(50000);
		command21.reserve(50000);
		command22.reserve(50000);
		command23.reserve(50000);
		command24.reserve(50000);
		command25.reserve(50000);
		command26.reserve(50000);
		command27.reserve(50000);
		command28.reserve(20000);
		command29.reserve(10000);
		command30.reserve(10000);
		command31.reserve(10000);
		command32.reserve(10000);
		command33.reserve(10000);
		command34.reserve(10000);
		command35.reserve(10000);
		command36.reserve(10000);
		command37.reserve(10000);
		command38.reserve(10000);
		command39.reserve(5000);
		command40.reserve(5000);
		ignorePathVector.reserve(50);
	}
};

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

void Volume::executeAll(vector<string*>& vec, const char* init) {
	sqlite3_stmt* stmt = NULL;
	string str;
	size_t rc = sqlite3_prepare_v2(db, init, strlen(init), &stmt, 0);
	if (rc != SQLITE_OK) {
		cout << "error preparing statement" << endl;
		exit(-1);
	}
	for (vector<string*>::iterator iter = vec.begin(); iter != vec.end(); ++iter) {
		str = **iter;
		sqlite3_reset(stmt);
		sqlite3_bind_text(stmt, 1, str.c_str(), -1, SQLITE_STATIC);
		sqlite3_step(stmt);
	}
	sqlite3_finalize(stmt);
}

void Volume::saveResult(string* path, int ascII) {
	int asciiGroup = ascII / 100;
	switch (asciiGroup)
	{
	case 0:
		command0.emplace_back(path);
		break;

	case 1:
		command1.emplace_back(path);
		break;

	case 2:
		command2.emplace_back(path);
		break;

	case 3:
		command3.emplace_back(path);
		break;

	case 4:
		command4.emplace_back(path);
		break;

	case 5:
		command5.emplace_back(path);
		break;
	case 6:
		command6.emplace_back(path);
		break;

	case 7:
		command7.emplace_back(path);
		break;

	case 8:
		command8.emplace_back(path);
		break;

	case 9:
		command9.emplace_back(path);
		break;

	case 10:
		command10.emplace_back(path);
		break;

	case 11:
		command11.emplace_back(path);
		break;

	case 12:
		command12.emplace_back(path);
		break;

	case 13:
		command13.emplace_back(path);
		break;

	case 14:
		command14.emplace_back(path);
		break;

	case 15:
		command15.emplace_back(path);
		break;

	case 16:
		command16.emplace_back(path);
		break;

	case 17:
		command17.emplace_back(path);
		break;

	case 18:
		command18.emplace_back(path);
		break;

	case 19:
		command19.emplace_back(path);
		break;

	case 20:
		command20.emplace_back(path);
		break;

	case 21:
		command21.emplace_back(path);
		break;

	case 22:
		command22.emplace_back(path);
		break;

	case 23:
		command23.emplace_back(path);
		break;

	case 24:
		command24.emplace_back(path);
		break;

	case 25:
		command25.emplace_back(path);
		break;

	case 26:
		command26.emplace_back(path);
		break;
	case 27:
		command27.emplace_back(path);
		break;
	case 28:
		command28.emplace_back(path);
		break;
	case 29:
		command29.emplace_back(path);
		break;
	case 30:
		command30.emplace_back(path);
		break;
	case 31:
		command31.emplace_back(path);
		break;
	case 32:
		command32.emplace_back(path);
		break;
	case 33:
		command33.emplace_back(path);
		break;
	case 34:
		command34.emplace_back(path);
		break;
	case 35:
		command35.emplace_back(path);
		break;
	case 36:
		command36.emplace_back(path);
		break;
	case 37:
		command37.emplace_back(path);
		break;
	case 38:
		command38.emplace_back(path);
		break;
	case 39:
		command39.emplace_back(path);
		break;
	case 40:
		command40.emplace_back(path);
		break;

	default:
		break;
	}
}

void Volume::saveToDatabase() {
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
	executeAll(command26, "INSERT INTO list26(PATH) VALUES(?);");
	executeAll(command27, "INSERT INTO list27(PATH) VALUES(?);");
	executeAll(command28, "INSERT INTO list28(PATH) VALUES(?);");
	executeAll(command29, "INSERT INTO list29(PATH) VALUES(?);");
	executeAll(command30, "INSERT INTO list30(PATH) VALUES(?);");
	executeAll(command31, "INSERT INTO list31(PATH) VALUES(?);");
	executeAll(command32, "INSERT INTO list32(PATH) VALUES(?);");
	executeAll(command33, "INSERT INTO list33(PATH) VALUES(?);");
	executeAll(command34, "INSERT INTO list34(PATH) VALUES(?);");
	executeAll(command35, "INSERT INTO list35(PATH) VALUES(?);");
	executeAll(command36, "INSERT INTO list36(PATH) VALUES(?);");
	executeAll(command37, "INSERT INTO list37(PATH) VALUES(?);");
	executeAll(command38, "INSERT INTO list38(PATH) VALUES(?);");
	executeAll(command39, "INSERT INTO list39(PATH) VALUES(?);");
	executeAll(command40, "INSERT INTO list40(PATH) VALUES(?);");
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