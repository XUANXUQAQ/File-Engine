#include "search.h"

CONCURRENT_MAP<HANDLE, LPVOID> sharedMemoryMap;
static std::atomic_int* completeTaskCount = new std::atomic_int(0);
static std::atomic_int* allTaskCount = new std::atomic_int(0);
static std::atomic<LPVOID> isCompletePtr(nullptr);
constexpr int RECORD_MAX_PATH = 500;
constexpr int MAX_RECORD_COUNT = 1000;

using namespace std;

volume::volume(const char vol, sqlite3* database, std::vector<std::string>* ignorePaths, PriorityMap* priorityMap)
{
	this->vol = vol;
	this->priorityMap = priorityMap;
	hVol = nullptr;
	path = "";
	db = database;
	ignorePathVector = ignorePaths;
	++*allTaskCount;
}

void volume::initVolume()
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
		try
		{
			const auto endIter = frnPfrnNameMap.end();
			for (auto iter = frnPfrnNameMap.begin(); iter != endIter; ++iter)
			{
				auto name = iter->second.filename;
				const auto ascii = getAscIISum(to_utf8(wstring(name)));
				CString path = _T("\0");
				getPath(iter->first, path);
				CString record = vol + path;
				auto fullPath = to_utf8(wstring(record));
				if (!isIgnore(fullPath))
				{
					collectResult(ascii, fullPath);
				}
			}
		}
		catch (exception& e)
		{
			cout << e.what() << endl;
		}
#ifdef TEST
		cout << "start copy disk " << this->getDiskPath() << " to shared memory" << endl;
#endif
		copyResultsToSharedMemory();
		++*completeTaskCount;
		setCompleteSignal();
		const std::time_t startWaitTime = std::time(nullptr);
		//阻塞等待其他线程全部完成共享内存的复制
		while (allTaskCount->load() != completeTaskCount->load())
		{
			if (std::time(nullptr) - startWaitTime > 10000)
			{
				//等待超过十秒
				break;
			}
			Sleep(10);
		}
		saveAllResultsToDb();
	}
#ifdef TEST
	cout << "disk " << this->getDiskPath() << " complete" << endl;
#endif
}

void volume::copyResultsToSharedMemory()
{
	for (int i = 0; i <= 40; ++i)
	{
		for (const auto& eachPriority : *priorityMap)
		{
			static const char* listNamePrefix = "list";
			static const char* prefix = "sharedMemory:";
			size_t memorySize = 0;
			string eachListName = string(listNamePrefix) + to_string(i);
			const auto& sharedMemoryName = string(prefix) + getDiskPath() + ":" + eachListName + ":" + to_string(
				eachPriority.second); // 共享内存名为 sharedMemory:[path]:list[num]:[priority]
			createSharedMemoryAndCopy(eachListName, eachPriority.second, &memorySize, sharedMemoryName);
		}
	}
}

void volume::collectResult(const int ascii, const string& fullPath)
{
	const int asciiGroup = ascii / 100;
	if (asciiGroup > 40)
	{
		return;
	}
	string listName("list");
	listName += to_string(asciiGroup);
	CONCURRENT_MAP<int, CONCURRENT_QUEUE<string>&>* tmpResults;
	CONCURRENT_QUEUE<string>* priorityStrList;
	const int priority = getPriorityByPath(fullPath);
	try
	{
		tmpResults = allResultsMap.at(listName);
		try
		{
			priorityStrList = &tmpResults->at(priority);
		}
		catch (exception&)
		{
			priorityStrList = new CONCURRENT_QUEUE<string>();
			tmpResults->insert(pair<int, CONCURRENT_QUEUE<string>&>(priority, *priorityStrList));
		}
	}
	catch (exception&)
	{
		priorityStrList = new CONCURRENT_QUEUE<string>();
		tmpResults = new CONCURRENT_MAP<int, CONCURRENT_QUEUE<string>&>();
		tmpResults->insert(pair<int, CONCURRENT_QUEUE<string>&>(priority, *priorityStrList));
		allResultsMap.insert(
			pair<string, CONCURRENT_MAP<int, CONCURRENT_QUEUE<string>&>*>(listName, tmpResults));
	}
	priorityStrList->push(fullPath);
}

void volume::setCompleteSignal()
{
	const BOOL isAllDone = allTaskCount->load() == completeTaskCount->load();
	memcpy_s(isCompletePtr.load(), sizeof(BOOL), &isAllDone, sizeof(BOOL));
}


void volume::createSharedMemoryAndCopy(const string& listName, const int priority, size_t* size,
                                       const string& sharedMemoryName)
{
	if (allResultsMap.find(listName) == allResultsMap.end())
	{
		return;
	}
	const auto& table = allResultsMap.at(listName);
	if (table->find(priority) == table->end())
	{
		return;
	}
	CONCURRENT_QUEUE<string>& result = table->at(priority);
	size_t _size = result.unsafe_size();
	_size = _size > MAX_RECORD_COUNT ? MAX_RECORD_COUNT : _size;
	const size_t memorySize = _size * RECORD_MAX_PATH;

	// 创建共享文件句柄
	HANDLE hMapFile;
	LPVOID pBuf = nullptr;
	createFileMapping(hMapFile, pBuf, memorySize, sharedMemoryName.c_str());
	*size = memorySize;
	if (pBuf == nullptr)
	{
		return;
	}
	long long count = 0;
	for (auto iterator = result.unsafe_begin(); iterator != result.unsafe_end() && count <= MAX_RECORD_COUNT; ++iterator)
	{
		memcpy_s(reinterpret_cast<void*>(reinterpret_cast<long long>(pBuf) + count * RECORD_MAX_PATH), RECORD_MAX_PATH,
		         iterator->c_str(), iterator->length());
		++count;
	}
	// 保存该结果的大小信息
	createFileMapping(hMapFile, pBuf, sizeof size_t, (sharedMemoryName + "size").c_str());
	memcpy_s(pBuf, sizeof size_t, &memorySize, sizeof size_t);
}

void volume::saveAllResultsToDb()
{
	initAllPrepareStatement();
	for (auto& eachTable : allResultsMap)
	{
		for (auto& eachResult : *eachTable.second)
		{
			for (auto iter = eachResult.second.unsafe_begin(); iter != eachResult.second.unsafe_end(); ++iter)
			{
				saveResult(*iter, getAscIISum(getFileName(*iter)));
			}
		}
	}
	finalizeAllStatement();
}

int volume::getPriorityBySuffix(const string& suffix) const
{
	const auto& iter = priorityMap->find(suffix);
	if (iter == priorityMap->end())
	{
		if (suffix.find('\\') != string::npos)
		{
			return getPriorityBySuffix("dirPriority");
		}
		return getPriorityBySuffix("defaultPriority");
	}
	return iter->second;
}


int volume::getPriorityByPath(const string& _path) const
{
	auto suffix = _path.substr(_path.find_last_of('.') + 1);
	transform(suffix.begin(), suffix.end(), suffix.begin(), tolower);
	return getPriorityBySuffix(suffix);
}

void volume::initSinglePrepareStatement(sqlite3_stmt** statement, const char* init) const
{
	const size_t ret = sqlite3_prepare_v2(db, init, static_cast<long>(strlen(init)), statement, nullptr);
	if (SQLITE_OK != ret)
	{
		cout << "error preparing stmt \"" << init << "\"" << endl;
		cout << "disk: " << this->getDiskPath() << endl;
	}
}

void volume::finalizeAllStatement() const
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

void volume::saveSingleRecordToDB(sqlite3_stmt* stmt, const string& record, const int ascii) const
{
	sqlite3_reset(stmt);
	sqlite3_bind_int(stmt, 1, ascii);
	sqlite3_bind_text(stmt, 2, record.c_str(), -1, SQLITE_STATIC);
	sqlite3_bind_int(stmt, 3, getPriorityByPath(record));
	sqlite3_step(stmt);
}

void volume::initAllPrepareStatement()
{
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

bool volume::isIgnore(const string& _path) const
{
	if (_path.find('$') != string::npos)
	{
		return true;
	}
	string _path0(_path);
	transform(_path0.begin(), _path0.end(), _path0.begin(), tolower);
	return std::any_of(ignorePathVector->begin(), ignorePathVector->end(), [_path0](const string& each)
	{
		return _path0.find(each) != string::npos;
	});
}

void volume::saveResult(const string& _path, const int ascII) const
{
	const int asciiGroup = ascII / 100;
	switch (asciiGroup)
	{
	case 0:
		saveSingleRecordToDB(stmt0, _path, ascII);
		break;
	case 1:
		saveSingleRecordToDB(stmt1, _path, ascII);
		break;
	case 2:
		saveSingleRecordToDB(stmt2, _path, ascII);
		break;
	case 3:
		saveSingleRecordToDB(stmt3, _path, ascII);
		break;
	case 4:
		saveSingleRecordToDB(stmt4, _path, ascII);
		break;
	case 5:
		saveSingleRecordToDB(stmt5, _path, ascII);
		break;
	case 6:
		saveSingleRecordToDB(stmt6, _path, ascII);
		break;
	case 7:
		saveSingleRecordToDB(stmt7, _path, ascII);
		break;
	case 8:
		saveSingleRecordToDB(stmt8, _path, ascII);
		break;
	case 9:
		saveSingleRecordToDB(stmt9, _path, ascII);
		break;
	case 10:
		saveSingleRecordToDB(stmt10, _path, ascII);
		break;
	case 11:
		saveSingleRecordToDB(stmt11, _path, ascII);
		break;
	case 12:
		saveSingleRecordToDB(stmt12, _path, ascII);
		break;
	case 13:
		saveSingleRecordToDB(stmt13, _path, ascII);
		break;
	case 14:
		saveSingleRecordToDB(stmt14, _path, ascII);
		break;
	case 15:
		saveSingleRecordToDB(stmt15, _path, ascII);
		break;
	case 16:
		saveSingleRecordToDB(stmt16, _path, ascII);
		break;
	case 17:
		saveSingleRecordToDB(stmt17, _path, ascII);
		break;
	case 18:
		saveSingleRecordToDB(stmt18, _path, ascII);
		break;
	case 19:
		saveSingleRecordToDB(stmt19, _path, ascII);
		break;
	case 20:
		saveSingleRecordToDB(stmt20, _path, ascII);
		break;
	case 21:
		saveSingleRecordToDB(stmt21, _path, ascII);
		break;
	case 22:
		saveSingleRecordToDB(stmt22, _path, ascII);
		break;
	case 23:
		saveSingleRecordToDB(stmt23, _path, ascII);
		break;
	case 24:
		saveSingleRecordToDB(stmt24, _path, ascII);
		break;
	case 25:
		saveSingleRecordToDB(stmt25, _path, ascII);
		break;
	case 26:
		saveSingleRecordToDB(stmt26, _path, ascII);
		break;
	case 27:
		saveSingleRecordToDB(stmt27, _path, ascII);
		break;
	case 28:
		saveSingleRecordToDB(stmt28, _path, ascII);
		break;
	case 29:
		saveSingleRecordToDB(stmt29, _path, ascII);
		break;
	case 30:
		saveSingleRecordToDB(stmt30, _path, ascII);
		break;
	case 31:
		saveSingleRecordToDB(stmt31, _path, ascII);
		break;
	case 32:
		saveSingleRecordToDB(stmt32, _path, ascII);
		break;
	case 33:
		saveSingleRecordToDB(stmt33, _path, ascII);
		break;
	case 34:
		saveSingleRecordToDB(stmt34, _path, ascII);
		break;
	case 35:
		saveSingleRecordToDB(stmt35, _path, ascII);
		break;
	case 36:
		saveSingleRecordToDB(stmt36, _path, ascII);
		break;
	case 37:
		saveSingleRecordToDB(stmt37, _path, ascII);
		break;
	case 38:
		saveSingleRecordToDB(stmt38, _path, ascII);
		break;
	case 39:
		saveSingleRecordToDB(stmt39, _path, ascII);
		break;
	case 40:
		saveSingleRecordToDB(stmt40, _path, ascII);
		break;
	default:
		break;
	}
}

int volume::getAscIISum(const string& name)
{
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

void volume::getPath(DWORDLONG frn, CString& _path)
{
	const auto end = frnPfrnNameMap.end();
	while (true)
	{
		auto it = frnPfrnNameMap.find(frn);
		if (it == end)
		{
			_path = L":" + _path;
			return;
		}
		_path = _T("\\") + it->second.filename + _path;
		frn = it->second.pfrn;
	}
}

bool volume::getHandle()
{
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


	if (INVALID_HANDLE_VALUE != hVol)
	{
		return true;
	}

	return false;
}

bool volume::createUSN()
{
	cujd.MaximumSize = 0; // 0表示使用默认值  
	cujd.AllocationDelta = 0; // 0表示使用默认值

	DWORD br;
	if (
		DeviceIoControl(hVol, // handle to volume
		                FSCTL_CREATE_USN_JOURNAL, // dwIoControlCode
		                &cujd, // input buffer
		                sizeof(cujd), // size of input buffer
		                nullptr, // lpOutBuffer
		                0, // nOutBufferSize
		                &br, // number of bytes returned
		                nullptr) // OVERLAPPED structure	
	)
	{
		return true;
	}
	return false;
}


bool volume::getUSNInfo()
{
	DWORD br;
	if (
		DeviceIoControl(hVol, // handle to volume
		                FSCTL_QUERY_USN_JOURNAL, // dwIoControlCode
		                nullptr, // lpInBuffer
		                0, // nInBufferSize
		                &ujd, // output buffer
		                sizeof(ujd), // size of output buffer
		                &br, // number of bytes returned
		                nullptr) // OVERLAPPED structure
	)
	{
		return true;
	}
	return false;
}

bool volume::getUSNJournal()
{
	MFT_ENUM_DATA med;
	med.StartFileReferenceNumber = 0;
	med.LowUsn = ujd.FirstUsn;
	med.HighUsn = ujd.NextUsn;

	// 根目录
	CString tmp(_T("C:"));
	tmp.SetAt(0, vol);
	frnPfrnNameMap[0x20000000000005].filename = tmp;
	frnPfrnNameMap[0x20000000000005].pfrn = 0;

	constexpr auto BUF_LEN = 0x3900; // 尽可能地大，提高效率;

	CHAR Buffer[BUF_LEN];
	DWORD usnDataSize;

	while (0 != DeviceIoControl(hVol,
	                            FSCTL_ENUM_USN_DATA,
	                            &med,
	                            sizeof(med),
	                            Buffer,
	                            BUF_LEN,
	                            &usnDataSize,
	                            nullptr))
	{
		DWORD dwRetBytes = usnDataSize - sizeof(USN);
		// 找到第一个 USN 记录  
		auto UsnRecord = reinterpret_cast<PUSN_RECORD>(static_cast<PCHAR>(Buffer) + sizeof(USN));

		while (dwRetBytes > 0)
		{
			// 获取到的信息  	
			const CString CfileName(UsnRecord->FileName, UsnRecord->FileNameLength / 2);

			pfrnName.filename = CfileName;
			pfrnName.pfrn = UsnRecord->ParentFileReferenceNumber;

			frnPfrnNameMap[UsnRecord->FileReferenceNumber] = pfrnName;
			// 获取下一个记录  
			const auto recordLen = UsnRecord->RecordLength;
			dwRetBytes -= recordLen;
			UsnRecord = reinterpret_cast<PUSN_RECORD>(reinterpret_cast<PCHAR>(UsnRecord) + recordLen);
		}
		// 获取下一页数据 
		med.StartFileReferenceNumber = *reinterpret_cast<USN*>(&Buffer);
	}
	return true;
}

bool volume::deleteUSN() const
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
	)
	{
		CloseHandle(hVol);
		return true;
	}
	CloseHandle(hVol);
	return false;
}

string to_utf8(const wstring& str)
{
	return to_utf8(str.c_str(), static_cast<int>(str.size()));
}

string to_utf8(const wchar_t* buffer, int len)
{
	const auto nChars = WideCharToMultiByte(
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
	WideCharToMultiByte(
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

string getFileName(const string& path)
{
	string fileName = path.substr(path.find_last_of('\\') + 1);
	return fileName;
}

bool initCompleteSignalMemory()
{
	HANDLE handle;
	LPVOID tmp;
	createFileMapping(handle, tmp, sizeof(bool), "sharedMemory:complete:status");
	if (tmp == nullptr)
	{
		cout << GetLastError() << endl;
		return false;
	}
	isCompletePtr.store(tmp);
	return true;
}

void closeSharedMemory()
{
	for (const auto& each : sharedMemoryMap)
	{
		UnmapViewOfFile(each.second);
		CloseHandle(each.first);
	}
}

void createFileMapping(HANDLE& hMapFile, LPVOID& pBuf, size_t memorySize, const char* sharedMemoryName)
{
	// 创建共享文件句柄
	hMapFile = CreateFileMappingA(
		INVALID_HANDLE_VALUE, // 物理文件句柄
		nullptr, // 默认安全级别
		PAGE_READWRITE, // 可读可写
		0, // 高位文件大小
		static_cast<DWORD>(memorySize), // 低位文件大小
		sharedMemoryName
	);

	pBuf = MapViewOfFile(
		hMapFile, // 共享内存的句柄
		FILE_MAP_ALL_ACCESS, // 可读写许可
		0,
		0,
		memorySize
	);
	sharedMemoryMap.insert(pair<HANDLE, LPVOID>(hMapFile, pBuf));
}
