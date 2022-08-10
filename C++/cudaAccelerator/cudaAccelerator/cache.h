#pragma once
#include <atomic>

typedef struct cache_struct
{
	char* dev_cache = nullptr;
	std::atomic_uint64_t remain_blank_num;
	std::atomic_uint64_t record_num;
	char* dev_output = nullptr;
	bool is_cache_valid = false;
	std::atomic_bool is_match_done;
} list_cache;
