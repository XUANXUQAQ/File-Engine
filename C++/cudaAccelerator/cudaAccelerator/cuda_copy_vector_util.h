#pragma once
#include "cuda_runtime.h"
#include <string>
#include <vector>

cudaError_t vector_to_cuda_char_array(const std::vector<std::string>& vec, void** cuda_mem);
