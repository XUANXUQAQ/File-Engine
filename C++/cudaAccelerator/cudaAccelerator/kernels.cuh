#pragma once
#include <string>
#include <vector>
#include "cache.h"
#include "cuda_runtime.h"

#define GET_TID() ((gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z) + blockDim.x * blockDim.y * threadIdx. z + blockDim.x * threadIdx.y + threadIdx.x)
#define gpuErrchk(ans, is_exit) { gpuAssert((ans), __FILE__, __LINE__, (is_exit)); }

void CUDART_CB set_match_done_flag_callback(cudaStream_t, cudaError_t, void* data);

inline void gpuAssert(cudaError_t code, const char* file, int line, bool is_exit);

void start_kernel(concurrency::concurrent_unordered_map<std::string, list_cache*>& cache_map,
                  const std::vector<std::string>& search_case,
                  bool is_ignore_case,
                  const char* search_text,
                  const std::vector<std::string>& keywords,
                  const std::vector<std::string>& keywords_lower_case,
                  const bool* is_keyword_path);

__device__ bool not_matched(const char* path,
                            bool is_ignore_case,
                            char* keywords,
                            char* keywords_lower_case,
                            int keywords_length,
                            const bool* is_keyword_path);


__global__ void check(const unsigned long long* str_address_ptr_array,
                      const int* search_case,
                      const bool* is_ignore_case,
                      char* search_text,
                      char* keywords,
                      char* keywords_lower_case,
                      const size_t* keywords_length,
                      const bool* is_keyword_path,
                      char* output);

__device__ int strcmp_cuda(const char* str1, const char* str2);
__device__ char* strlwr_cuda(char* src);
__device__ char* strstr_cuda(char* s1, char* s2);
__device__ char* strrchr_cuda(const char* s, int c);
__device__ char* strcpy_cuda(char* dst, const char* src);
__device__ void get_file_name(const char* path, char* output);
__device__ void get_parent_path(const char* path, char* output);
