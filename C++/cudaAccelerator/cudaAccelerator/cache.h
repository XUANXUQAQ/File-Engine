#pragma once
#include <atomic>
#include <vector>

typedef struct cache_data
{
	unsigned long long* dev_cache_str_ptr = nullptr;
	std::vector<unsigned long long> str_length;
	std::atomic_uint64_t remain_blank_num;
	std::atomic_uint64_t record_num;
} cache_data;

typedef struct cache_struct
{
	cache_data str_data;
	char* dev_output = nullptr;
	bool is_cache_valid = false;
	std::atomic_bool is_match_done;
	std::atomic_bool is_output_done;
} list_cache;
