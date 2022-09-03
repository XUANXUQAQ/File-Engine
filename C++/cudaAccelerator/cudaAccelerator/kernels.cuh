#pragma once
#include <string>
#include <vector>
#include "cache.h"
#include <concurrent_unordered_map.h>
#include "cuda_runtime.h"

#define GET_TID() ((gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z) + blockDim.x * blockDim.y * threadIdx. z + blockDim.x * threadIdx.y + threadIdx.x)
#define gpuErrchk(ans, is_exit, info) gpuAssert((ans), __FILE__, __LINE__, __FUNCTION__, (is_exit), (info))

void CUDART_CB set_match_done_flag_callback(cudaStream_t, cudaError_t, void* data);

inline void gpuAssert(cudaError_t code, const char* file, int line, const char* function, bool is_exit,
	const char* info)
{
	if (code != cudaSuccess)
	{
		if (info == nullptr)
		{
			fprintf(stderr, "GPU assert: %s %s %d %s\n", cudaGetErrorString(code), file, line, function);
		}
		else
		{
			fprintf(stderr, "GPU assert: %s %s %d %s %s\n", cudaGetErrorString(code), file, line, function, info);
		}
		if (is_exit)
		{
			std::quick_exit(code);
		}
	}
}

void start_kernel(concurrency::concurrent_unordered_map<std::string, list_cache*>& cache_map,
	const std::vector<std::string>& search_case,
	bool is_ignore_case,
	const char* search_text,
	const std::vector<std::string>& keywords,
	const std::vector<std::string>& keywords_lower_case,
	const bool* is_keyword_path,
	cudaStream_t* streams,
	size_t stream_count);

__device__ bool not_matched(const char* path,
	bool is_ignore_case,
	char* keywords,
	char* keywords_lower_case,
	int keywords_length,
	const bool* is_keyword_path);


__global__ void check(const char(*str_address_ptr_array)[MAX_PATH_LENGTH],
	const size_t* total_num,
	const int* search_case,
	const bool* is_ignore_case,
	char* search_text,
	char* keywords,
	char* keywords_lower_case,
	const size_t* keywords_length,
	const bool* is_keyword_path,
	char* output,
	const bool* is_stop_collect_var);

__device__ void convert_to_pinyin(const char* chinese_str, char* output_str);
void free_cuda_search_memory();
void init_cuda_search_memory();
bool set_using_device(int device_number);
size_t find_table_sizeof2(size_t target);
