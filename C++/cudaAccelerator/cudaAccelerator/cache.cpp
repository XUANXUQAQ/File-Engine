#include "pch.h"
#include "cache.h"
#include <cuda_runtime_api.h>
#include <string>
#include "kernels.cuh"

static stop_signal is_stop_collect;

std::string get_cache_info(const std::string& key, const list_cache* cache)
{
    std::string str;
    str.append("cache key: ").append(key)
       .append("cache record num: ").append(std::to_string(cache->str_data.record_num)).append("  ")
       .append("is cache valid: ").append(std::to_string(cache->is_cache_valid));
    return str;
}

bool is_stop()
{
    return is_stop_collect.is_stop_collect;
}

bool* get_dev_stop_signal()
{
    return is_stop_collect.dev_is_stop_collect;
}

void set_stop(const bool b)
{
    is_stop_collect.is_stop_collect = b;
    gpuErrchk(cudaMemcpy(is_stop_collect.dev_is_stop_collect, &b, sizeof(bool), cudaMemcpyHostToDevice), true, nullptr);
}

void init_stop_signal()
{
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&is_stop_collect.dev_is_stop_collect), sizeof(bool)), true, nullptr);
}

void free_stop_signal()
{
    gpuErrchk(cudaFree(is_stop_collect.dev_is_stop_collect), false, nullptr);
}
