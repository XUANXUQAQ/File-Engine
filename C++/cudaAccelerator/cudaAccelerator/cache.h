#pragma once
#include <atomic>
#include <mutex>
#include <vector>

typedef struct cache_data
{
	unsigned long long* dev_cache_str_ptr = nullptr;
	size_t* str_length_array = nullptr;
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
