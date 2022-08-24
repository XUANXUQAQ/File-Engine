#include "file_search_usn.h"
#include "string_to_utf8.h"
#include <iostream>
#include <thread>
#include "search.h"
#include <fstream>
#include "constants.h"

static PriorityMap suffix_priority_map;

/**
 * 创建数据库表 list0-list40
 */
void init_tables(sqlite3* db)
{
	using namespace std;
	sqlite3_exec(db, "BEGIN;", nullptr, nullptr, nullptr);
	for (int i = 0; i < 41; i++)
	{
		string sql = "CREATE TABLE IF NOT EXISTS list" + to_string(i) + "(ASCII INT, PATH TEXT, PRIORITY INT)";
		sqlite3_exec(db, sql.c_str(), nullptr, nullptr, nullptr);
	}
	sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
}

/**
 * 开始搜索
 */
void init_usn(parameter p)
{
	init_tables(p.db);
	sqlite3_exec(p.db, "BEGIN;", nullptr, nullptr, nullptr);
	volume volumeInstance(p.disk, p.db, &p.ignorePath, &suffix_priority_map);
	volumeInstance.init_volume();
	sqlite3_exec(p.db, "COMMIT;", nullptr, nullptr, nullptr);
	sqlite3_close(p.db);
#ifdef TEST
	std::cout << "path : " << p.disk << std::endl;
	std::cout << "Initialize done " << p.disk << std::endl;
#endif
}

/**
 * 获取后缀优先级表，构造成Map
 */
inline bool init_priority_map(PriorityMap& priority_map, const char* priorityDbPath)
{
	using namespace std;
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
	for (const auto total = column * row + 2; i < total; i += 2)
	{
		const string suffix(pResult[i]);
		const string priorityVal(pResult[i + 1]);
		auto pairPriority = pair<string, int>{suffix, stoi(priorityVal)};
		priority_map.insert(pairPriority);
	}
	priority_map.insert(pair<string, int>("dirPriority", -1));
	sqlite3_free_table(pResult);
	sqlite3_close(cacheDb);
	return true;
}

/**
 * 通过逗号（,）分割字符串，并存入vector
 */
void split_string(const char* str, std::vector<std::string>& vec)
{
	char* remain_disk = nullptr;
	char disk_path[5000];
	strcpy_s(disk_path, str);
	char* _diskPath = disk_path;
	char* p = strtok_s(_diskPath, ",", &remain_disk);
	if (p != nullptr)
	{
		vec.emplace_back(p);
	}
	while (p != nullptr)
	{
		p = strtok_s(nullptr, ",", &remain_disk);
		if (p != nullptr)
		{
			vec.emplace_back(p);
		}
	}
}

int main()
{
	using namespace std;
	char disk_path[500];
	char output[500];
	char ignore_path[500];

	vector<string> disk_vector;
	vector<string> ignore_paths_vector;

	ifstream input("MFTSearchInfo.dat", ios::in);
	if (!input)
	{
		cerr << "open MFTSearchInfo.dat failed";
		return 1;
	}
	input.getline(disk_path, 500);
	input.getline(output, 500);
	input.getline(ignore_path, 500);
	input.close();

#ifdef TEST
	cout << "disk path: " << disk_path << endl;
	cout << "output: " << output << endl;
	cout << "ignore_path" << ignore_path << endl;
#endif
	disk_vector.reserve(26);

	sqlite3_config(SQLITE_CONFIG_MULTITHREAD);
	sqlite3_config(SQLITE_CONFIG_MEMSTATUS, 0);

	split_string(disk_path, disk_vector);
	split_string(ignore_path, ignore_paths_vector);

	bool is_priority_map_initialized = false;
	vector<thread> threads;
	if (!init_complete_signal_memory())
	{
		close_shared_memory();
		return 1;
	}
	// 创建线程
	for (auto& iter : disk_vector)
	{
		const auto disk = iter[0];
		if ('A' <= disk && disk <= 'Z')
		{
			parameter p;
			p.disk = disk;
			p.ignorePath = ignore_paths_vector;
			char tmp_db_path[1000];
			strcpy_s(tmp_db_path, output);
			strcat_s(tmp_db_path, "\\");
			const size_t length = strlen(tmp_db_path);
			tmp_db_path[length] = disk;
			tmp_db_path[length + 1] = '\0';
			strcat_s(tmp_db_path, ".db");
			const size_t ret = sqlite3_open(tmp_db_path, &p.db);
#ifdef TEST
			cout << "database path: " << tmp_db_path << endl;
#endif
			if (SQLITE_OK != ret)
			{
				cout << "error opening database" << endl;
				return 1;
			}
			tmp_db_path[strlen(tmp_db_path) - 4] = '\0';
			strcat_s(tmp_db_path, "cache.db");

			if (!is_priority_map_initialized)
			{
				is_priority_map_initialized = true;
				init_priority_map(suffix_priority_map, tmp_db_path);
			}
			sqlite3_exec(p.db, "PRAGMA TEMP_STORE=MEMORY;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA cache_size=262144;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA page_size=65535;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA auto_vacuum=0;", nullptr, nullptr, nullptr);
			sqlite3_exec(p.db, "PRAGMA mmap_size=4096;", nullptr, nullptr, nullptr);
			threads.emplace_back(thread(init_usn, p));
		}
	}

	for (auto& each_thread : threads)
	{
		each_thread.join();
	}
	close_shared_memory();
	return 0;
}
