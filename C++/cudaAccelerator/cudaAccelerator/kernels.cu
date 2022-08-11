#include "pch.h"
#include <memory>
#include <string>
#include <vector>
#include <concurrent_unordered_map.h>
#include "cuda_copy_vector_util.h"
#include "kernels.cuh"
#include "cache.h"
#include "constans.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

inline void gpuAssert(cudaError_t code, const char* file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		std::quick_exit(code);
	}
}

__device__ bool not_matched(char* path,
                            bool is_ignore_case,
                            char* keywords,
                            char* keywords_lower_case,
                            int keywords_length,
                            bool* is_keyword_path)
{
	for (int i = 0; i < keywords_length; ++i)
	{
		const bool is_keyword_path_val = is_keyword_path[i];
		char match_str[MAX_PATH_LENGTH];
		if (is_keyword_path_val)
		{
			get_parent_path(path, match_str);
		}
		else
		{
			get_file_name(path, match_str);
		}
		char* each_keyword;
		if (is_ignore_case)
		{
			each_keyword = keywords_lower_case + i * static_cast<unsigned long long>(MAX_PATH_LENGTH);
			// strlwr_cuda(each_keyword);
			strlwr_cuda(match_str);
		}
		else
		{
			each_keyword = keywords + i * static_cast<unsigned long long>(MAX_PATH_LENGTH);
		}
		if (strstr_cuda(match_str, each_keyword) == nullptr)
		{
			return true;
		}
	}
	return false;
}


__device__ int strcmp_cuda(const char* str1, const char* str2)
{
	while (*str1)
	{
		if (*str1 > *str2)return 1;
		if (*str1 < *str2)return -1;
		++str1;
		++str2;
	}
	if (*str1 < *str2)return -1;
	return 0;
}


__device__ char* strlwr_cuda(char* src)
{
	while (*src != '\0')
	{
		if (*src > 'A' && *src <= 'Z')
		{
			*src += 32;
		}
		++src;
	}
	return src;
}


__device__ char* strstr_cuda(char* s1, char* s2)
{
	int n;
	if (*s2) //两种情况考虑
	{
		while (*s1)
		{
			for (n = 0; *(s1 + n) == *(s2 + n); ++n)
			{
				if (!*(s2 + n + 1)) //查找的下一个字符是否为'\0'
				{
					return s1;
				}
			}
			++s1;
		}
		return nullptr;
	}
	return s1;
}

__device__ char* strrchr_cuda(const char* s, int c)
{
	if (s == nullptr)
	{
		return nullptr;
	}

	char* p_char = nullptr;
	while (*s != '\0')
	{
		if (*s == static_cast<char>(c))
		{
			p_char = const_cast<char*>(s);
		}
		++s;
	}

	return p_char;
}

__device__ char* strcpy_cuda(char* dst, const char* src)
{
	char* ret = dst;
	while ((*dst++ = *src++) != '\0')
	{
	}
	return ret;
}

__device__ void get_file_name(const char* path, char* output)
{
	const char* p = strrchr_cuda(path, '\\');
	strcpy_cuda(output, p + 1);
}

__device__ void get_parent_path(const char* path, char* output)
{
	strcpy_cuda(output, path);
	char* p = strrchr_cuda(output, '\\');
	*p = '\0';
}


__global__ void check(char* paths,
                      int* search_case,
                      bool* is_ignore_case,
                      char* search_text,
                      char* keywords,
                      char* keywords_lower_case,
                      size_t* keywords_length,
                      bool* is_keyword_path,
                      char* output)
{
	const int thread_id = GET_TID();
	output[thread_id] = 0;
	char* path = paths + thread_id * static_cast<unsigned long long>(MAX_PATH_LENGTH);
	if (path == nullptr || !path[0])
	{
		return;
	}
	if (not_matched(path, *is_ignore_case, keywords, keywords_lower_case, static_cast<int>(*keywords_length),
	                is_keyword_path))
	{
		return;
	}
	if (*search_case == 0)
	{
		output[thread_id] = 1;
		return;
	}
	// if ((*search_case & 1) == 1)
	// {
	// 	// TODO 判断是否为文件
	// }
	// if ((*search_case & 2) == 2)
	// {
	// 	// TODO 判断是否为文件夹
	// }
	if ((*search_case & 4) == 4)
	{
		// 全字匹配
		strlwr_cuda(search_text);
		char file_name[FILENAME_MAX];
		get_file_name(path, file_name);
		strlwr_cuda(file_name);
		if (strcmp_cuda(search_text, file_name) != 0)
		{
			return;
		}
	}
	output[thread_id] = 1;
}

void start_kernel(concurrency::concurrent_unordered_map<std::string, list_cache*>& cache_map,
                  const std::vector<std::string>& search_case,
                  bool is_ignore_case,
                  const char* search_text,
                  const std::vector<std::string>& keywords,
                  const std::vector<std::string>& keywords_lower_case,
                  const bool* is_keyword_path)
{
	int* dev_search_case = nullptr;
	char* dev_search_text = nullptr;
	char* dev_keywords = nullptr;
	char* dev_keywords_lower_case = nullptr;
	size_t* dev_keywords_length = nullptr;
	bool* dev_is_keyword_path = nullptr;
	bool* dev_is_ignore_case = nullptr;

	const auto keywords_num = keywords.size();
	const auto stream_count = cache_map.size();
	auto streams = new cudaStream_t[stream_count];
	//初始化流
	for (size_t i = 0; i < stream_count; ++i)
	{
		gpuErrchk(cudaStreamCreate(&streams[i]))
	}
	do
	{
		// 选择第一个GPU
		gpuErrchk(cudaSetDevice(0))

		// 复制search case
		// 第一位为1表示有F，第二位为1表示有D，第三位为1表示有FULL，CASE由File-Engine主程序进行判断，传入参数is_ignore_case为false表示有CASE
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_search_case), sizeof(int)))
		int search_case_num = 0;
		for (auto& each_case : search_case)
		{
			//TODO 支持文件和文件夹判断
			// if (each_case == "f")
			// {
			// 	search_case_num |= 1;
			// }
			// if (each_case == "d")
			// {
			// 	search_case_num |= 2;
			// }
			if (each_case == "full")
			{
				search_case_num |= 4;
			}
		}
		gpuErrchk(cudaMemcpy(dev_search_case, &search_case_num, sizeof(int), cudaMemcpyHostToDevice))

		// 复制search text
		const auto search_text_len = strlen(search_text);
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_search_text), (search_text_len + 1) * sizeof(char)))
		gpuErrchk(cudaMemset(dev_search_text, 0, search_text_len + 1))
		gpuErrchk(cudaMemcpy(dev_search_text, search_text, search_text_len, cudaMemcpyHostToDevice))

		// 复制keywords
		gpuErrchk(vector_to_cuda_char_array(keywords, reinterpret_cast<void**>(&dev_keywords), 0))

		// 复制keywords_lower_case
		gpuErrchk(vector_to_cuda_char_array(keywords_lower_case, reinterpret_cast<void**>(&dev_keywords_lower_case),0))

		//复制keywords_length
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_keywords_length), sizeof(size_t)))
		gpuErrchk(cudaMemcpy(dev_keywords_length, &keywords_num, sizeof(size_t), cudaMemcpyHostToDevice))

		// 复制is_keyword_path
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_is_keyword_path), sizeof(bool) * keywords_num))
		gpuErrchk(cudaMemcpy(dev_is_keyword_path, is_keyword_path, sizeof(bool) * keywords_num, cudaMemcpyHostToDevice))

		// 复制is_ignore_case
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_is_ignore_case), sizeof(bool)))
		gpuErrchk(cudaMemcpy(dev_is_ignore_case, &is_ignore_case, sizeof(bool), cudaMemcpyHostToDevice))
		int count = 0;
		for (auto& each : cache_map)
		{
			int block_num, thread_num;
			const auto& cache = each.second;
			if (cache->record_num > MAX_THREAD_PER_BLOCK)
			{
				thread_num = MAX_THREAD_PER_BLOCK;
				block_num = static_cast<int>(cache->record_num / thread_num);
			}
			else
			{
				thread_num = static_cast<int>(cache->record_num.load());
				block_num = 1;
			}
			cache->is_match_done = false;
			//注册回调
			cudaStreamAddCallback(streams[count], set_match_done_flag_callback, cache, 0);

			check<<<block_num, thread_num, 0, streams[count]>>>
			(cache->dev_cache,
			 dev_search_case,
			 dev_is_ignore_case,
			 dev_search_text,
			 dev_keywords,
			 dev_keywords_lower_case,
			 dev_keywords_length,
			 dev_is_keyword_path,
			 cache->dev_output);
			++count;
		}

		// 检查启动错误
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "check launch failed: %s\n", cudaGetErrorString(cudaStatus));
			break;
		}

		// 等待执行完成
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launch!\n", cudaStatus);
		}
	}
	while (false);
	delete[] streams;
	cudaFree(dev_search_case);
	cudaFree(dev_search_text);
	cudaFree(dev_keywords);
	cudaFree(dev_keywords_lower_case);
	cudaFree(dev_is_keyword_path);
	cudaFree(dev_is_ignore_case);
	cudaFree(dev_keywords_length);
}

void CUDART_CB set_match_done_flag_callback(cudaStream_t, cudaError_t, void* data)
{
	const auto cache = static_cast<list_cache*>(data);
	cache->is_match_done = true;
}
