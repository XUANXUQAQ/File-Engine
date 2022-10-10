#include "search.h"
#ifdef TEST
#include <iostream>
#endif
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include "sqlite3.h"
#include <concurrent_unordered_set.h>
#include <concurrent_unordered_map.h>
#include <atomic>
#include <mutex>
#include "constants.h"

CONCURRENT_MAP<HANDLE, LPVOID> shared_memory_map;
static std::atomic_int* complete_task_count = new std::atomic_int(0);
static std::atomic_int* all_task_count = new std::atomic_int(0);
static std::atomic<LPVOID> is_complete_ptr(nullptr);

volume::volume(const char vol, sqlite3* database, std::vector<std::string>* ignore_paths, PriorityMap* priority_map)
{
	this->vol = vol;
	this->priority_map_ = priority_map;
	hVol = nullptr;
	path = "";
	db = database;
	ignore_path_vector_ = ignore_paths;
	++* all_task_count;
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
		using namespace std;
#ifdef TEST
		cout << "starting " << thread_num << " threads to collect results" << endl;
		cout << "fileMap size: " << map_size << endl;
#endif
		auto collect_internal = [this](const Frn_Pfrn_Name_Map::iterator& map_iterator)
		{
			const auto& name = map_iterator->second.filename;
			const int ascii = get_asc_ii_sum(to_utf8(wstring(name)));
			CString result_path = _T("\0");
			get_path(map_iterator->first, result_path);
			const CString record = vol + result_path;
			if (const auto full_path = to_utf8(wstring(record)); !is_ignore(full_path))
			{
				collect_result_to_result_map(ascii, full_path);
				const string tmp_path(full_path);
				const size_t pos = tmp_path.find_last_of('\\');
				if (pos != string::npos)
				{
					if (auto&& parent_path = tmp_path.substr(0, pos); parent_path.length() > 2)
					{
						collect_result_to_result_map(get_asc_ii_sum(get_file_name(parent_path)), parent_path);
					}
				}
			}
		};
		auto search_internal = [this, &collect_internal]
		{
			auto&& start_iter = frnPfrnNameMap.begin();
			auto&& end_iter = frnPfrnNameMap.end();
			while (start_iter != end_iter)
			{
				collect_internal(start_iter);
				++start_iter;
			}
		};
		try
		{
			search_internal();
			printf("collect complete.\n");
			thread save_results_to_db_thread([this]
				{
					save_all_results_to_db();
				});
			auto&& info = string("start to copy disk ") + this->getDiskPath() + " to shared memory";
			printf("%s\n", info.c_str());
			copy_results_to_shared_memory();
			set_complete_signal();
			++* complete_task_count;
			printf("copy to shared memory complete. save to database.\n");
			if (save_results_to_db_thread.joinable())
				save_results_to_db_thread.join();
		}
		catch (exception& e)
		{
			fprintf(stderr, "fileSearcherUSN: %s\n", e.what());
		}
	}
	else
	{
		fprintf(stderr, "fileSearcherUSN: init usn journal failed.");
	}
	auto&& info = std::string("disk ") + this->getDiskPath() + " complete";
	printf("%s\n", info.c_str());
}

void volume::copy_results_to_shared_memory()
{
	for (int i = 0; i <= 40; ++i)
	{
		for (const auto& [fst, snd] : *priority_map_)
		{
			static auto list_name_prefix = "list";
			static auto prefix = "sharedMemory:";
			std::string each_list_name = std::string(list_name_prefix) + std::to_string(i);
			const auto& shared_memory_name =
				// 共享内存名为 sharedMemory:[path]:list[num]:[priority]
				std::string(prefix) + getDiskPath() + ":" + each_list_name + ":" + std::to_string(snd);
			create_shared_memory_and_copy(each_list_name, snd, shared_memory_name);
		}
	}
}

void volume::collect_result_to_result_map(const int ascii, const std::string& full_path)
{
	static std::mutex add_priority_lock;
	int ascii_group = ascii / 100;
	if (ascii_group > 40)
	{
		ascii_group = 40;
	}
	std::string list_name("list");
	list_name += std::to_string(ascii_group);
	CONCURRENT_MAP<int, CONCURRENT_SET<std::string>&>* tmp_results;
	CONCURRENT_SET<std::string>* priority_str_list = nullptr;
	const int priority = get_priority_by_path(full_path);
	try
	{
		tmp_results = all_results_map.at(list_name);
		try
		{
			priority_str_list = &tmp_results->at(priority);
		}
		catch (std::exception&)
		{
			std::lock_guard lock_guard(add_priority_lock);
			auto&& iter = tmp_results->find(priority);
			if (iter == tmp_results->end())
			{
				priority_str_list = new CONCURRENT_SET<std::string>();
				tmp_results->insert(std::pair<int, CONCURRENT_SET<std::string>&>(priority, *priority_str_list));
			}
			else
			{
				priority_str_list = &iter->second;
			}
		}
	}
	catch (std::out_of_range&)
	{
		static std::mutex add_list_lock;
		std::lock_guard lock_guard(add_list_lock);
		auto&& iter = all_results_map.find(list_name);
		if (iter == all_results_map.end())
		{
			priority_str_list = new CONCURRENT_SET<std::string>();
			tmp_results = new CONCURRENT_MAP<int, CONCURRENT_SET<std::string>&>();
			tmp_results->insert(std::pair<int, CONCURRENT_SET<std::string>&>(priority, *priority_str_list));
			all_results_map.insert(
				std::pair(list_name, tmp_results));
		}
		else
		{
			tmp_results = iter->second;
			std::lock_guard lock_guard2(add_priority_lock);
			auto&& iter2 = tmp_results->find(priority);
			if (iter2 == tmp_results->end())
			{
				priority_str_list = new CONCURRENT_SET<std::string>();
				tmp_results->insert(std::pair<int, CONCURRENT_SET<std::string>&>(priority, *priority_str_list));
			}
			else
			{
				priority_str_list = &iter2->second;
			}
		}
	}
	catch (std::exception& e)
	{
		fprintf(stderr, "fileSearcherUSN: %s\n", e.what());
	}
	if (priority_str_list != nullptr)
	{
		priority_str_list->insert(full_path);
	}
}

void volume::set_complete_signal()
{
	const BOOL is_all_done = all_task_count->load() == complete_task_count->load();
	memcpy_s(is_complete_ptr.load(), sizeof(BOOL), &is_all_done, sizeof(BOOL));
}

void volume::create_shared_memory_and_copy(const std::string& list_name, const int priority,
	const std::string& shared_memory_name)
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
	CONCURRENT_SET<std::string>& result = table->at(priority);
	size_t result_size = result.size();
	result_size = result_size > MAX_RECORD_COUNT ? MAX_RECORD_COUNT : result_size;
	const size_t memory_size = result_size * RECORD_MAX_PATH;

	// 创建共享文件句柄
	HANDLE h_map_file;
	LPVOID p_buf = nullptr;
	create_file_mapping(h_map_file, p_buf, memory_size, shared_memory_name.c_str());
	if (p_buf == nullptr)
	{
		return;
	}
	size_t count = 0;
	for (auto iterator = result.begin();
		iterator != result.end() && count < MAX_RECORD_COUNT;
		++iterator)
	{
		if (iterator->length() >= RECORD_MAX_PATH)
		{
			continue;
		}
		memcpy_s(reinterpret_cast<void*>(reinterpret_cast<size_t>(p_buf) + count * RECORD_MAX_PATH),
			RECORD_MAX_PATH,
			iterator->c_str(), iterator->length());
		++count;
	}
	// 保存该结果的大小信息
	create_file_mapping(h_map_file, p_buf, sizeof size_t, (shared_memory_name + "size").c_str());
	memcpy_s(p_buf, sizeof size_t, &memory_size, sizeof size_t);
}

void volume::save_all_results_to_db()
{
	init_all_prepare_statement();
	for (auto& [fst, snd] : all_results_map)
	{
		const int ascii_group = stoi(fst.substr(4));
		for (auto& [priority, result_container] : *snd)
		{
			for (auto&& iter = result_container.begin(); iter != result_container.end(); ++iter)
			{
				save_result(*iter, get_asc_ii_sum(get_file_name(*iter)), ascii_group, priority);
			}
		}
	}
	finalize_all_statement();
}

int volume::get_priority_by_suffix(const std::string& suffix) const
{
	auto&& iter = priority_map_->find(suffix);
	if (iter == priority_map_->end())
	{
		if (suffix.find('\\') != std::string::npos)
		{
			return get_priority_by_suffix("dirPriority");
		}
		return get_priority_by_suffix("defaultPriority");
	}
	return iter->second;
}


int volume::get_priority_by_path(const std::string& _path) const
{
	auto suffix = _path.substr(_path.find_last_of('.') + 1);
	transform(suffix.begin(), suffix.end(), suffix.begin(), tolower);
	return get_priority_by_suffix(suffix);
}

void volume::init_single_prepare_statement(sqlite3_stmt** statement, const char* init) const
{
	if (const size_t ret = sqlite3_prepare_v2(db, init, static_cast<long>(strlen(init)), statement, nullptr); SQLITE_OK != ret)
	{
		auto&& err_info = std::string("error preparing stmt \"") + init + "\"  disk: " + this->getDiskPath();
		fprintf(stderr, "fileSearcherUSN: %s\n", err_info.c_str());
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

void volume::save_single_record_to_db(sqlite3_stmt* stmt, const std::string& record, const int ascii, const int priority)
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

bool volume::is_ignore(const std::string& _path) const
{
	if (_path.find('$') != std::string::npos)
	{
		return true;
	}
	std::string path0(_path);
	transform(path0.begin(), path0.end(), path0.begin(), tolower);
	return std::any_of(ignore_path_vector_->begin(), ignore_path_vector_->end(), [path0](const std::string& each)
		{
			return path0.find(each) != std::string::npos;
		});
}

void volume::save_result(const std::string& _path, const int ascii, const int ascii_group, const int priority) const
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

int volume::get_asc_ii_sum(const std::string& name)
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
	CString lp_file_name(_T("\\\\.\\c:"));
	lp_file_name.SetAt(4, vol);


	hVol = CreateFile(lp_file_name,
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
	auto&& info = std::wstring(L"create file handle failed. ") + lp_file_name.GetString() +
		L"error code: " + std::to_wstring(GetLastError());
	fprintf(stderr, "fileSearcherUSN: %ls", info.c_str());
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
	auto&& info = "create usn error. Error code: " + std::to_string(GetLastError());
	fprintf(stderr, "fileSearcherUSN: %s\n", info.c_str());
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
	auto&& info = "query usn error. Error code: " + std::to_string(GetLastError());
	fprintf(stderr, "fileSearcherUSN: %s\n", info.c_str());
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

	constexpr auto BUF_LEN = sizeof(USN) + 0x100000; // 尽可能地大，提高效率;

	CHAR* buffer = new CHAR[BUF_LEN];
	DWORD usn_data_size;
	pfrn_name pfrn_name;

	while (0 != DeviceIoControl(hVol,
		FSCTL_ENUM_USN_DATA,
		&med,
		sizeof med,
		buffer,
		BUF_LEN,
		&usn_data_size,
		nullptr))
	{
		DWORD dw_ret_bytes = usn_data_size - sizeof(USN);
		// 找到第一个 USN 记录  
		auto usn_record = reinterpret_cast<PUSN_RECORD>(buffer + sizeof(USN));

		while (dw_ret_bytes > 0)
		{
			// 获取到的信息  	
			const CString cfile_name(usn_record->FileName, usn_record->FileNameLength / 2);
			pfrn_name.filename = cfile_name;
			pfrn_name.pfrn = usn_record->ParentFileReferenceNumber;
			// frnPfrnNameMap[UsnRecord->FileReferenceNumber] = pfrnName;
			frnPfrnNameMap.insert(std::make_pair(usn_record->FileReferenceNumber, pfrn_name));
			// 获取下一个记录  
			const auto record_len = usn_record->RecordLength;
			dw_ret_bytes -= record_len;
			usn_record = reinterpret_cast<PUSN_RECORD>(reinterpret_cast<PCHAR>(usn_record) + record_len);
		}
		// 获取下一页数据 
		med.StartFileReferenceNumber = *reinterpret_cast<DWORDLONG*>(buffer);
	}
	delete[] buffer;
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

std::string get_file_name(const std::string& path)
{
	std::string file_name = path.substr(path.find_last_of('\\') + 1);
	return file_name;
}

bool init_complete_signal_memory()
{
	HANDLE handle;
	LPVOID tmp;
	create_file_mapping(handle, tmp, sizeof(bool), "sharedMemory:complete:status");
	if (tmp == nullptr)
	{
		fprintf(stderr, "fileSearcherUSN: %ld\n", GetLastError());
		return false;
	}
	is_complete_ptr.store(tmp);
	return true;
}

void close_shared_memory()
{
	for (const auto& [fst, snd] : shared_memory_map)
	{
		UnmapViewOfFile(snd);
		CloseHandle(fst);
	}
}

void create_file_mapping(HANDLE& h_map_file, LPVOID& p_buf, const size_t memory_size, const char* shared_memory_name)
{
	// 创建共享文件句柄
	h_map_file = CreateFileMappingA(
		INVALID_HANDLE_VALUE, // 物理文件句柄
		nullptr, // 默认安全级别
		PAGE_READWRITE, // 可读可写
		0, // 高位文件大小
		static_cast<DWORD>(memory_size), // 低位文件大小
		shared_memory_name
	);

	p_buf = MapViewOfFile(
		h_map_file, // 共享内存的句柄
		FILE_MAP_ALL_ACCESS, // 可读写许可
		0,
		0,
		memory_size
	);
	shared_memory_map.insert(std::pair(h_map_file, p_buf));
}
