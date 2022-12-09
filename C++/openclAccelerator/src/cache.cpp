#include "cache.h"
#include <string>

std::string get_cache_info(const std::string& key, const list_cache* cache)
{
    std::string str;
    str.append("cache key: ").append(key)
       .append("cache record num: ").append(std::to_string(cache->str_data.record_num)).append("  ")
       .append("cache remain blank num: ").append(std::to_string(cache->str_data.remain_blank_num)).append("  ")
       .append("is cache valid: ").append(std::to_string(cache->is_cache_valid));
    return str;
}
