#include "search.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include "sqlite3.h"
#include <concurrent_queue.h>
#include <concurrent_unordered_map.h>
#include <atomic>
#include <mutex>

CONCURRENT_MAP<HANDLE, LPVOID> shared_memory_map;
static std::atomic_int* complete_task_count = new std::atomic_int(0);
static std::atomic_int* all_task_count = new std::atomic_int(0);
static std::atomic<LPVOID> is_complete_ptr(nullptr);
constexpr int RECORD_MAX_PATH = 500;
constexpr int MAX_RECORD_COUNT = 1000;

using namespace std;

volume::volume(const char vol, sqlite3* database, std::vector<std::string>* ignorePaths, PriorityMap* priorityMap)
{
	this->vol = vol;
	this->priority_map_ = priorityMap;
	hVol = nullptr;
	path = "";
	db = database;
	ignore_path_vector_ = ignorePaths;
	++*all_task_count;
}

void volume::init_volume()
{
	if (
		// 2.获取驱动盘句柄
		get_handle() &&
		// 3.创建USN日志
		create_usn() &&
		// 4.获取USN日志信息
		get_usn_info() &&
		// 5.获取 USN Journal 文件的基本信息
		get_usn_journal() &&
		// 06. 删除 USN 日志文件 ( 也可以不删除 ) 
		delete_usn()
	)
	{
		auto collect_internal = [this](Frn_Pfrn_Name_Map::iterator iter)
		{
			const auto& name = iter->second.filename;
			const int ascii = get_asc_ii_sum(to_utf8(wstring(name)));
			CString result_path = _T("\0");
			get_path(iter->first, result_path);
			CString record = vol + result_path;
			const auto full_path = to_utf8(wstring(record));
			if (!is_ignore(full_path))
			{
				collect_result_to_result_map(ascii, full_path);
			}
		};
		try
		{
			SYSTEM_INFO sys_info;
			GetSystemInfo(&sys_info);
			const auto thread_num = sys_info.dwNumberOfProcessors;
			const auto map_size = frnPfrnNameMap.size();
			const auto split_size = map_size / thread_num;
			vector<thread> thread_vec;
			auto search_internal = [split_size, this, collect_internal](int pre_loop_count)
			{
				auto start_iter = frnPfrnNameMap.begin();
				for (size_t i = 0; i < split_size * pre_loop_count; ++i)
				{
					++start_iter;
				}
				for (size_t i = 0; i < split_size; ++i)
				{
					collect_internal(start_iter);
					++start_iter;
				}
			};
			for (DWORD i = 0; i < thread_num - 1; ++i)
			{
				thread_vec.emplace_back(thread(search_internal, i));
			}
			thread t([split_size, this, collect_internal, thread_num]
			{
				auto start_iter = frnPfrnNameMap.begin();
				auto end_iter = frnPfrnNameMap.end();
				for (size_t i = 0; i < split_size * (static_cast<size_t>(thread_num) - 1); ++i)
				{
					++start_iter;
				}
				while (start_iter != end_iter)
				{
					collect_internal(start_iter);
					++start_iter;
				}
			});
			for (auto& each : thread_vec)
			{
				if (each.joinable())
				{
					each.join();
				}
			}
			if (t.joinable())
			{
				t.join();
			}
		}
#ifdef TEST
		catch (exception& e)
		{
			cout << e.what() << endl;
		}
#else
		catch (exception&)
		{
		}
#endif
#ifdef TEST
		cout << "start copy disk " << this->getDiskPath() << " to shared memory" << endl;
#endif
		copy_results_to_shared_memory();
		++*complete_task_count;
		set_complete_signal();
		const std::time_t startWaitTime = std::time(nullptr);
		//阻塞等待其他线程全部完成共享内存的复制
		while (all_task_count->load() != complete_task_count->load())
		{
			if (std::time(nullptr) - startWaitTime > 10000)
			{
				//等待超过十秒
				break;
			}
			Sleep(10);
		}
		save_all_results_to_db();
	}
#ifdef TEST
	cout << "disk " << this->getDiskPath() << " complete" << endl;
#endif
}

void volume::copy_results_to_shared_memory()
{
	for (int i = 0; i <= 40; ++i)
	{
		for (const auto& eachPriority : *priority_map_)
		{
			static const char* listNamePrefix = "list";
			static const char* prefix = "sharedMemory:";
			size_t memorySize = 0;
			string eachListName = string(listNamePrefix) + to_string(i);
			const auto& sharedMemoryName = string(prefix) + getDiskPath() + ":" + eachListName + ":" + to_string(
				eachPriority.second); // 共享内存名为 sharedMemory:[path]:list[num]:[priority]
			create_shared_memory_and_copy(eachListName, eachPriority.second, &memorySize, sharedMemoryName);
		}
	}
}

void volume::collect_result_to_result_map(const int ascii, const string& full_path)
{
	static std::mutex add_priority_lock;
	int ascii_group = ascii / 100;
	if (ascii_group > 40)
	{
		ascii_group = 40;
	}
	string list_name("list");
	list_name += to_string(ascii_group);
	CONCURRENT_MAP<int, CONCURRENT_QUEUE<string>&>* tmp_results;
	CONCURRENT_QUEUE<string>* priority_str_list;
	const int priority = get_priority_by_path(full_path);
	try
	{
		tmp_results = all_results_map.at(list_name);
		try
		{
			priority_str_list = &tmp_results->at(priority);
		}
		catch (exception&)
		{
			std::lock_guard<std::mutex> lock_guard(add_priority_lock);
			auto iter = tmp_results->find(priority);
			if (iter == tmp_results->end())
			{
				priority_str_list = new CONCURRENT_QUEUE<string>();
				tmp_results->insert(pair<int, CONCURRENT_QUEUE<string>&>(priority, *priority_str_list));
			}
			else
			{
				priority_str_list = &iter->second;
			}
		}
	}
	catch (exception&)
	{
		static std::mutex add_list_lock;
		std::lock_guard<std::mutex> lock_guard(add_list_lock);
		auto iter = all_results_map.find(list_name);
		if (iter == all_results_map.end())
		{
			priority_str_list = new CONCURRENT_QUEUE<string>();
			tmp_results = new CONCURRENT_MAP<int, CONCURRENT_QUEUE<string>&>();
			tmp_results->insert(pair<int, CONCURRENT_QUEUE<string>&>(priority, *priority_str_list));
			all_results_map.insert(
				pair<string, CONCURRENT_MAP<int, CONCURRENT_QUEUE<string>&>*>(list_name, tmp_results));
		}
		else
		{
			tmp_results = iter->second;
			std::lock_guard<std::mutex> lock_guard2(add_priority_lock);
			auto iter2 = tmp_results->find(priority);
			if (iter2 == tmp_results->end())
			{
				priority_str_list = new CONCURRENT_QUEUE<string>();
				tmp_results->insert(pair<int, CONCURRENT_QUEUE<string>&>(priority, *priority_str_list));
			}
			else
			{
				priority_str_list = &iter2->second;
			}
		}
	}
	priority_str_list->push(full_path);
}

void volume::set_complete_signal()
{
	const BOOL isAllDone = all_task_count->load() == complete_task_count->load();
	memcpy_s(is_complete_ptr.load(), sizeof(BOOL), &isAllDone, sizeof(BOOL));
}


void volume::create_shared_memory_and_copy(const string& list_name, const int priority, size_t* size,
                                           const string& shared_memory_name)
{
	if (all_results_map.find(list_name) == all_results_map.end())
	{
		return;
	}
	const auto& table = all_results_map.at(list_name);
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
	create_file_mapping(hMapFile, pBuf, memorySize, shared_memory_name.c_str());
	*size = memorySize;
	if (pBuf == nullptr)
	{
		return;
	}
	long long count = 0;
	for (auto iterator = result.unsafe_begin(); iterator != result.unsafe_end() && count <= MAX_RECORD_COUNT; ++
	     iterator)
	{
		memcpy_s(reinterpret_cast<void*>(reinterpret_cast<long long>(pBuf) + count * RECORD_MAX_PATH), RECORD_MAX_PATH,
		         iterator->c_str(), iterator->length());
		++count;
	}
	// 保存该结果的大小信息
	create_file_mapping(hMapFile, pBuf, sizeof size_t, (shared_memory_name + "size").c_str());
	memcpy_s(pBuf, sizeof size_t, &memorySize, sizeof size_t);
}

void volume::save_all_results_to_db()
{
	init_all_prepare_statement();
	for (auto& eachTable : all_results_map)
	{
		const int asciiGroup = stoi(eachTable.first.substr(4));
		for (auto& eachResult : *eachTable.second)
		{
			const int priority = eachResult.first;
			for (auto iter = eachResult.second.unsafe_begin(); iter != eachResult.second.unsafe_end(); ++iter)
			{
				save_result(*iter, get_asc_ii_sum(get_file_name(*iter)), asciiGroup, priority);
			}
		}
	}
	finalize_all_statement();
}

int volume::get_priority_by_suffix(const string& suffix) const
{
	const auto& iter = priority_map_->find(suffix);
	if (iter == priority_map_->end())
	{
		if (suffix.find('\\') != string::npos)
		{
			return get_priority_by_suffix("dirPriority");
		}
		return get_priority_by_suffix("defaultPriority");
	}
	return iter->second;
}


int volume::get_priority_by_path(const string& _path) const
{
	auto suffix = _path.substr(_path.find_last_of('.') + 1);
	transform(suffix.begin(), suffix.end(), suffix.begin(), tolower);
	return get_priority_by_suffix(suffix);
}

void volume::init_single_prepare_statement(sqlite3_stmt** statement, const char* init) const
{
	const size_t ret = sqlite3_prepare_v2(db, init, static_cast<long>(strlen(init)), statement, nullptr);
	if (SQLITE_OK != ret)
	{
		cout << "error preparing stmt \"" << init << "\"" << endl;
		cout << "disk: " << this->getDiskPath() << endl;
	}
}

void volume::finalize_all_statement() const
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

void volume::save_single_record_to_db(sqlite3_stmt* stmt, const string& record, const int ascii, const int priority)
{
	sqlite3_reset(stmt);
	sqlite3_bind_int(stmt, 1, ascii);
	sqlite3_bind_text(stmt, 2, record.c_str(), -1, SQLITE_STATIC);
	sqlite3_bind_int(stmt, 3, priority);
	sqlite3_step(stmt);
}

void volume::init_all_prepare_statement()
{
	init_single_prepare_statement(&stmt0, "INSERT OR IGNORE INTO list0 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt1, "INSERT OR IGNORE INTO list1 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt2, "INSERT OR IGNORE INTO list2 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt3, "INSERT OR IGNORE INTO list3 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt4, "INSERT OR IGNORE INTO list4 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt5, "INSERT OR IGNORE INTO list5 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt6, "INSERT OR IGNORE INTO list6 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt7, "INSERT OR IGNORE INTO list7 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt8, "INSERT OR IGNORE INTO list8 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt9, "INSERT OR IGNORE INTO list9 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt10, "INSERT OR IGNORE INTO list10 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt11, "INSERT OR IGNORE INTO list11 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt12, "INSERT OR IGNORE INTO list12 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt13, "INSERT OR IGNORE INTO list13 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt14, "INSERT OR IGNORE INTO list14 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt15, "INSERT OR IGNORE INTO list15 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt16, "INSERT OR IGNORE INTO list16 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt17, "INSERT OR IGNORE INTO list17 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt18, "INSERT OR IGNORE INTO list18 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt19, "INSERT OR IGNORE INTO list19 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt20, "INSERT OR IGNORE INTO list20 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt21, "INSERT OR IGNORE INTO list21 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt22, "INSERT OR IGNORE INTO list22 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt23, "INSERT OR IGNORE INTO list23 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt24, "INSERT OR IGNORE INTO list24 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt25, "INSERT OR IGNORE INTO list25 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt26, "INSERT OR IGNORE INTO list26 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt27, "INSERT OR IGNORE INTO list27 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt28, "INSERT OR IGNORE INTO list28 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt29, "INSERT OR IGNORE INTO list29 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt30, "INSERT OR IGNORE INTO list30 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt31, "INSERT OR IGNORE INTO list31 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt32, "INSERT OR IGNORE INTO list32 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt33, "INSERT OR IGNORE INTO list33 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt34, "INSERT OR IGNORE INTO list34 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt35, "INSERT OR IGNORE INTO list35 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt36, "INSERT OR IGNORE INTO list36 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt37, "INSERT OR IGNORE INTO list37 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt38, "INSERT OR IGNORE INTO list38 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt39, "INSERT OR IGNORE INTO list39 VALUES(?, ?, ?);");
	init_single_prepare_statement(&stmt40, "INSERT OR IGNORE INTO list40 VALUES(?, ?, ?);");
}

bool volume::is_ignore(const string& _path) const
{
	if (_path.find('$') != string::npos)
	{
		return true;
	}
	string _path0(_path);
	transform(_path0.begin(), _path0.end(), _path0.begin(), tolower);
	return std::any_of(ignore_path_vector_->begin(), ignore_path_vector_->end(), [_path0](const string& each)
	{
		return _path0.find(each) != string::npos;
	});
}

void volume::save_result(const string& _path, const int ascii, const int ascii_group, const int priority) const
{
	switch (ascii_group)
	{
	case 0:
		save_single_record_to_db(stmt0, _path, ascii, priority);
		break;
	case 1:
		save_single_record_to_db(stmt1, _path, ascii, priority);
		break;
	case 2:
		save_single_record_to_db(stmt2, _path, ascii, priority);
		break;
	case 3:
		save_single_record_to_db(stmt3, _path, ascii, priority);
		break;
	case 4:
		save_single_record_to_db(stmt4, _path, ascii, priority);
		break;
	case 5:
		save_single_record_to_db(stmt5, _path, ascii, priority);
		break;
	case 6:
		save_single_record_to_db(stmt6, _path, ascii, priority);
		break;
	case 7:
		save_single_record_to_db(stmt7, _path, ascii, priority);
		break;
	case 8:
		save_single_record_to_db(stmt8, _path, ascii, priority);
		break;
	case 9:
		save_single_record_to_db(stmt9, _path, ascii, priority);
		break;
	case 10:
		save_single_record_to_db(stmt10, _path, ascii, priority);
		break;
	case 11:
		save_single_record_to_db(stmt11, _path, ascii, priority);
		break;
	case 12:
		save_single_record_to_db(stmt12, _path, ascii, priority);
		break;
	case 13:
		save_single_record_to_db(stmt13, _path, ascii, priority);
		break;
	case 14:
		save_single_record_to_db(stmt14, _path, ascii, priority);
		break;
	case 15:
		save_single_record_to_db(stmt15, _path, ascii, priority);
		break;
	case 16:
		save_single_record_to_db(stmt16, _path, ascii, priority);
		break;
	case 17:
		save_single_record_to_db(stmt17, _path, ascii, priority);
		break;
	case 18:
		save_single_record_to_db(stmt18, _path, ascii, priority);
		break;
	case 19:
		save_single_record_to_db(stmt19, _path, ascii, priority);
		break;
	case 20:
		save_single_record_to_db(stmt20, _path, ascii, priority);
		break;
	case 21:
		save_single_record_to_db(stmt21, _path, ascii, priority);
		break;
	case 22:
		save_single_record_to_db(stmt22, _path, ascii, priority);
		break;
	case 23:
		save_single_record_to_db(stmt23, _path, ascii, priority);
		break;
	case 24:
		save_single_record_to_db(stmt24, _path, ascii, priority);
		break;
	case 25:
		save_single_record_to_db(stmt25, _path, ascii, priority);
		break;
	case 26:
		save_single_record_to_db(stmt26, _path, ascii, priority);
		break;
	case 27:
		save_single_record_to_db(stmt27, _path, ascii, priority);
		break;
	case 28:
		save_single_record_to_db(stmt28, _path, ascii, priority);
		break;
	case 29:
		save_single_record_to_db(stmt29, _path, ascii, priority);
		break;
	case 30:
		save_single_record_to_db(stmt30, _path, ascii, priority);
		break;
	case 31:
		save_single_record_to_db(stmt31, _path, ascii, priority);
		break;
	case 32:
		save_single_record_to_db(stmt32, _path, ascii, priority);
		break;
	case 33:
		save_single_record_to_db(stmt33, _path, ascii, priority);
		break;
	case 34:
		save_single_record_to_db(stmt34, _path, ascii, priority);
		break;
	case 35:
		save_single_record_to_db(stmt35, _path, ascii, priority);
		break;
	case 36:
		save_single_record_to_db(stmt36, _path, ascii, priority);
		break;
	case 37:
		save_single_record_to_db(stmt37, _path, ascii, priority);
		break;
	case 38:
		save_single_record_to_db(stmt38, _path, ascii, priority);
		break;
	case 39:
		save_single_record_to_db(stmt39, _path, ascii, priority);
		break;
	case 40:
		save_single_record_to_db(stmt40, _path, ascii, priority);
		break;
	default:
		break;
	}
}

int volume::get_asc_ii_sum(const string& name)
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

void volume::get_path(DWORDLONG frn, CString& output_path)
{
	const auto end = frnPfrnNameMap.end();
	while (true)
	{
		auto it = frnPfrnNameMap.find(frn);
		if (it == end)
		{
			output_path = L":" + output_path;
			return;
		}
		output_path = _T("\\") + it->second.filename + output_path;
		frn = it->second.pfrn;
	}
}

bool volume::get_handle()
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

bool volume::create_usn()
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


bool volume::get_usn_info()
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

bool volume::get_usn_journal()
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

bool volume::delete_usn() const
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

string get_file_name(const string& path)
{
	string fileName = path.substr(path.find_last_of('\\') + 1);
	return fileName;
}

bool init_complete_signal_memory()
{
	HANDLE handle;
	LPVOID tmp;
	create_file_mapping(handle, tmp, sizeof(bool), "sharedMemory:complete:status");
	if (tmp == nullptr)
	{
		cout << GetLastError() << endl;
		return false;
	}
	is_complete_ptr.store(tmp);
	return true;
}

void close_shared_memory()
{
	for (const auto& each : shared_memory_map)
	{
		UnmapViewOfFile(each.second);
		CloseHandle(each.first);
	}
}

void create_file_mapping(HANDLE& hMapFile, LPVOID& pBuf, size_t memorySize, const char* sharedMemoryName)
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
	shared_memory_map.insert(pair<HANDLE, LPVOID>(hMapFile, pBuf));
}
