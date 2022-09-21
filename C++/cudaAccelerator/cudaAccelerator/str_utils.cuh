#pragma once

__device__ int strcmp_cuda(const char* str1, const char* str2);
__device__ char* strlwr_cuda(char* src);
__device__ const char* strstr_cuda(const char* str1, const char* str2);
__device__ char* strrchr_cuda(const char* s, int c);
__device__ char* strcpy_cuda(char* dst, const char* src);
__device__ void get_file_name(const char* path, char* output);
__device__ void get_parent_path(const char* path, char* output);
__device__ size_t strlen_cuda(const char* str);
__device__ char* strcat_cuda(char* dst, char const* src);
__device__ void str_add_single(char* dst, char c);
__device__ bool is_str_contains_chinese(const char* source);
