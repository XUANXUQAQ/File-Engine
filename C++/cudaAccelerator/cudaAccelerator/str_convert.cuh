#pragma once
#include <cuda_runtime.h>

void init_str_convert();
// __device__ int gbk_to_utf8(const char* from, unsigned int from_len, char** to, unsigned int* to_len);
__device__ int utf8_to_gbk(const char* from, unsigned int from_len, char** to, unsigned int* to_len);
