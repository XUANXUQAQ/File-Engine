#pragma once
#include <atomic>
#include <mutex>

typedef struct cache_data
{
	char* dev_cache_str = nullptr;
	std::atomic_uint64_t remain_blank_num;
	std::atomic_uint64_t record_num;
	std::mutex lock;
} cache_data;

typedef struct cache_struct
{
	cache_data str_data;
	char* dev_output = nullptr;
	bool is_cache_valid = false;
	std::atomic_bool is_match_done;
	std::atomic_bool is_output_done;
} list_cache;

std::string get_cache_info(const std::string& key, const list_cache* cache);
