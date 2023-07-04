#pragma once

#include <string>
#include "stdafx.h"
#include <unordered_map>
#include "sqlite3.h"
#include <winioctl.h>
#include <concurrent_unordered_map.h>
#include <concurrent_unordered_set.h>

#define CONCURRENT_MAP concurrency::concurrent_unordered_map
#define CONCURRENT_SET concurrency::concurrent_unordered_set

typedef struct pfrn_name
{
    DWORDLONG pfrn = 0;
    CString filename;
} pfrn_name;

typedef std::unordered_map<std::string, int> PriorityMap;
typedef std::unordered_map<DWORDLONG, pfrn_name> Frn_Pfrn_Name_Map;

class volume
{
public:
    volume(char vol, sqlite3* database, std::vector<std::string>* ignore_paths, PriorityMap* priority_map);

    volume(volume&) = delete;

    volume(volume&&) = delete;

    ~volume();

    char getDiskPath() const
    {
        return vol;
    }

    void collect_result_to_result_map(int ascii, const std::string& full_path);

    void init_volume();

private:
    char vol;
    HANDLE hVol;
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

    USN_JOURNAL_DATA ujd{};

    std::vector<std::string>* ignore_path_vector_ = nullptr;
    PriorityMap* priority_map_ = nullptr;
    CONCURRENT_MAP<std::string, CONCURRENT_MAP<int, CONCURRENT_SET<std::string>*>*> all_results_map;

    bool get_handle();
    bool create_usn() const;
    bool get_usn_info();
    bool get_usn_journal();
    bool delete_usn() const;
    void save_result(const std::string& _path, int ascii, int ascii_group, int priority) const;
    void get_path(DWORDLONG frn, CString& output_path);
    static int get_asc_ii_sum(const std::string& name);
    bool is_ignore(const std::string& path) const;
    void finalize_all_statement() const;
    static void save_single_record_to_db(sqlite3_stmt* stmt, const std::string& record, int ascii, int priority);
    int get_priority_by_suffix(const std::string& suffix) const;
    int get_priority_by_path(const std::string& _path) const;
    void init_all_prepare_statement();
    void init_single_prepare_statement(sqlite3_stmt** statement, const char* init) const;
    void save_all_results_to_db();
};


std::string to_utf8(const wchar_t* buffer, int len);

std::string to_utf8(const std::wstring& str);

std::string get_file_name(const std::string& path);
