#include "pch.h"
#include "cuda_copy_vector_util.h"
#include "constans.h"

/**
 * 每个字符串分配max_path字节，存入连续内存
 * max_path定义在constants中
 */
cudaError_t vector_to_cuda_char_array(const std::vector<std::string>& vec, void** cuda_mem)
{
    cudaError_t cudaStatus;
    const auto vec_size = vec.size();
    const auto bytes = MAX_PATH_LENGTH * sizeof(char) * vec_size;
    cudaStatus = cudaMemset(*cuda_mem, 0, bytes);
    if (cudaStatus != cudaSuccess)
    {
        return cudaStatus;
    }
    for (size_t i = 0; i < vec_size; ++i)
    {
        auto& str = vec[i];
        const auto address_num = reinterpret_cast<unsigned long long>(*cuda_mem) + i * MAX_PATH_LENGTH;
        cudaStatus = cudaMemcpy(reinterpret_cast<void*>(address_num),
                                str.c_str(), str.length(), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            return cudaStatus;
        }
    }
    return cudaStatus;
}
