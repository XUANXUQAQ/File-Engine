#pragma once
#include <string>
#include <vector>
#include "cache.h"
#include "cuda_runtime.h"

void CUDART_CB set_match_done_flag_callback(cudaStream_t, cudaError_t, void* data);

void start_kernel(concurrency::concurrent_unordered_map<std::string, list_cache*>& cache_map,
                  const std::vector<std::string>& search_case,
                  bool is_ignore_case,
                  const char* search_text,
                  const std::vector<std::string>& keywords,
                  const std::vector<std::string>& keywords_lower_case,
                  const bool* is_keyword_path);

__device__ bool not_matched(char* path,
                            bool is_ignore_case,
                            char* keywords,
                            char* keywords_lower_case,
                            int keywords_length,
                            bool* is_keyword_path);


__global__ void check(char* paths,
                      int* search_case,
                      bool* is_ignore_case,
                      char* search_text,
                      char* keywords,
                      char* keywords_lower_case,
                      size_t* keywords_length,
                      bool* is_keyword_path,
                      char* output);

__device__ int strcmp_cuda(const char* str1, const char* str2);
__device__ char* strlwr_cuda(char* src);
__device__ char* strstr_cuda(char* s1, char* s2);
__device__ char* strrchr_cuda(const char* s, int c);
__device__ char* strcpy_cuda(char* dst, const char* src);
__device__ void get_file_name(const char* path, char* output);
__device__ void get_parent_path(const char* path, char* output);
