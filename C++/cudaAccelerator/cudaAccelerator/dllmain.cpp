// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <concurrent_unordered_map.h>
#include "kernels.cuh"
#include "file_engine_dllInterface_CudaAccelerator.h"
#include "cache.h"
#include "constans.h"
#include "cuda_copy_vector_util.h"
#include "str_convert.cuh"
#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

#pragma comment(lib, "cudart.lib")

void clear_cache(const std::string& key);
void clear_all_cache();
bool has_cache(const std::string& key);
void add_records_to_cache(const std::string& key, const std::vector<std::string>& records);
void remove_records_from_cache(const std::string& key, const std::vector<std::string>& records);
void generate_search_case(JNIEnv* env, std::vector<std::string>& search_case_vec, jobjectArray search_case);
void collect_results(JNIEnv* thread_env, jobject result_collector, std::atomic_uint& result_counter,
	unsigned max_results, const std::vector<std::string>& search_case_vec);
bool is_record_repeat(const std::string& record, const list_cache* cache);
void release_all();
int is_dir_or_file(const char* path);
std::wstring string2wstring(const std::string& str);

//lock
inline void wait_for_clear_cache();
inline void wait_for_add_or_remove_record();
inline void lock_clear_cache(std::atomic_uint& thread_counter);
inline void free_clear_cache(std::atomic_uint& thread_counter);
inline void lock_add_or_remove_result();
inline void free_add_or_remove_results();

concurrency::concurrent_unordered_map<std::string, list_cache*> cache_map;
concurrency::concurrent_unordered_map<std::string, unsigned> matched_result_number_map;
std::atomic_bool clear_cache_flag(false);
std::atomic_uint add_record_thread_count(0);
std::atomic_uint remove_record_thread_count(0);
std::mutex modify_cache_lock;
std::hash<std::string> hasher;
std::atomic_int task_complete_count;

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    getDevices
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_CudaAccelerator_getDevices
(JNIEnv* env, jobject)
{
	int device_count = 0;
	std::string device_string;
	std::string ret;
	gpuErrchk(cudaGetDeviceCount(&device_count), false, "get device number failed.");
	for (int i = 0; i < device_count; ++i)
	{
		cudaDeviceProp prop;
		gpuErrchk(cudaGetDeviceProperties(&prop, i), false, "get device info failed.");
		device_string.append(prop.name).append(",").append(std::to_string(i)).append(";");
	}
	if (device_count)
	{
		ret = device_string.substr(0, device_string.length() - 1);
	}
	return env->NewStringUTF(ret.c_str());
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    setDevice
 * Signature: (I)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_CudaAccelerator_setDevice
(JNIEnv*, jobject, jint device_number_jint)
{
	return set_using_device(device_number_jint);
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_release
(JNIEnv*, jobject)
{
	release_all();
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    initialize
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_initialize
(JNIEnv*, jobject)
{
	init_stop_signal();
	init_cuda_search_memory();
	init_str_convert();
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    resetAllResultStatus
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_resetAllResultStatus
(JNIEnv*, jobject)
{
	set_stop(false);
	// 初始化is_match_done is_output_done为false
	for (const auto& [key, val] : cache_map)
	{
		val->is_match_done = false;
		val->is_output_done = 0;
		gpuErrchk(cudaMemset(val->dev_output, 0,
			val->str_data.record_num + val->str_data.remain_blank_num), true, nullptr);
	}
	matched_result_number_map.clear();
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    stopCollectResults
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_stopCollectResults
(JNIEnv*, jobject)
{
	set_stop(true);
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    match
 * Signature: ([Ljava/lang/String;ZLjava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[ZILjava/util/function/BiConsumer;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_match
(JNIEnv* env, jobject, jobjectArray search_case, jboolean is_ignore_case, jstring search_text,
	jobjectArray keywords, jobjectArray keywords_lower, jbooleanArray is_keyword_path, jint max_results, jobject result_collector)
{
	if (cache_map.empty())
	{
		return;
	}
	wait_for_clear_cache();
	std::lock_guard lock_guard(modify_cache_lock);
	//生成搜索条件 search_case_vec
	std::vector<std::string> search_case_vec;
	if (search_case != nullptr)
	{
		generate_search_case(env, search_case_vec, search_case);
	}
	//生成搜索关键字 keywords_vec keywords_lower_vec is_keyword_path_ptr
	std::vector<std::string> keywords_vec;
	std::vector<std::string> keywords_lower_vec;
	const auto keywords_length = env->GetArrayLength(keywords);
	if (keywords_length > MAX_KEYWORDS_NUMBER)
	{
		fprintf(stderr, "too many keywords.\n");
		return;
	}
	bool is_keyword_path_ptr[MAX_KEYWORDS_NUMBER]{ false };
	const auto is_keyword_path_ptr_bool_array = env->GetBooleanArrayElements(is_keyword_path, nullptr);
	for (jsize i = 0; i < keywords_length; ++i)
	{
		auto tmp_keywords_str = reinterpret_cast<jstring>(env->GetObjectArrayElement(keywords, i));
		auto keywords_chars = env->GetStringUTFChars(tmp_keywords_str, nullptr);
#ifdef DEBUG_OUTPUT
		std::cout << "keywords: " << keywords_chars << std::endl;
#endif
		keywords_vec.emplace_back(keywords_chars);
		env->ReleaseStringUTFChars(tmp_keywords_str, keywords_chars);
		env->DeleteLocalRef(tmp_keywords_str);

		tmp_keywords_str = reinterpret_cast<jstring>(env->GetObjectArrayElement(keywords_lower, i));
		keywords_chars = env->GetStringUTFChars(tmp_keywords_str, nullptr);
		keywords_lower_vec.emplace_back(keywords_chars);
		env->ReleaseStringUTFChars(tmp_keywords_str, keywords_chars);
		env->DeleteLocalRef(tmp_keywords_str);

#ifdef DEBUG_OUTPUT
		std::cout << "is keyword path: " << static_cast<bool>(is_keyword_path_ptr_bool_array[i]) << std::endl;
#endif
		is_keyword_path_ptr[i] = is_keyword_path_ptr_bool_array[i];
	}
	env->ReleaseBooleanArrayElements(is_keyword_path, is_keyword_path_ptr_bool_array, JNI_ABORT);
	//复制全字匹配字符串 search_text
	const auto search_text_chars = env->GetStringUTFChars(search_text, nullptr);
	const auto stream_count = cache_map.size();
	const auto streams = new cudaStream_t[stream_count];
	//初始化流
	for (size_t i = 0; i < stream_count; ++i)
	{
		gpuErrchk(cudaStreamCreate(&streams[i]), true, nullptr);
	}
	//GPU并行计算
	start_kernel(cache_map, search_case_vec, is_ignore_case, search_text_chars, keywords_vec, keywords_lower_vec,
		is_keyword_path_ptr, streams, stream_count);
	std::atomic_uint result_counter = 0;
	task_complete_count = 0;
	collect_results(env, result_collector, result_counter, max_results, search_case_vec);
	env->ReleaseStringUTFChars(search_text, search_text_chars);
	// 等待执行完成
	const auto cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launch!\n", cudaStatus);
	for (size_t i = 0; i < stream_count; ++i)
	{
		gpuErrchk(cudaStreamDestroy(streams[i]), true, nullptr);
	}
	delete[] streams;
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    isMatchDone
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_CudaAccelerator_isMatchDone
(JNIEnv* env, jobject, jstring key_jstring)
{
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	auto iter = cache_map.find(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
	if (iter == cache_map.end())
	{
		return FALSE;
	}
	return iter->second->is_output_done.load() == 2;
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    matchedNumber
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_CudaAccelerator_matchedNumber
(JNIEnv* env, jobject, jstring key_jstring)
{
	const auto key = env->GetStringUTFChars(key_jstring, nullptr);
	unsigned matched_number;
	try
	{
		matched_number = matched_result_number_map.at(key);
	}
	catch (std::out_of_range&)
	{
		matched_number = 0;
	}
	env->ReleaseStringUTFChars(key_jstring, key);
	return matched_number;
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    hasCache
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_CudaAccelerator_hasCache
(JNIEnv*, jobject)
{
	return !cache_map.empty();
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    isCacheExist
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_CudaAccelerator_isCacheExist
(JNIEnv* env, jobject, jstring key_jstring)
{
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	const std::string key(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
	const bool ret = has_cache(key);
	return ret;
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    initCache
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_initCache
(JNIEnv* env, jobject, jstring key_jstring, jobjectArray records)
{
	const jsize _length = env->GetArrayLength(records);
	std::vector<std::string> records_vec;
	unsigned record_count = 0;
	for (jsize i = 0; i < _length; ++i)
	{
		const auto jstring_val = reinterpret_cast<jstring>(env->GetObjectArrayElement(records, i));
		const auto record = env->GetStringUTFChars(jstring_val, nullptr);
		if (const auto record_len = strlen(record); record_len < MAX_PATH_LENGTH)
		{
			records_vec.emplace_back(record);
			++record_count;
		}
		env->ReleaseStringUTFChars(jstring_val, record);
		env->DeleteLocalRef(jstring_val);
	}
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	std::string key(_key);
	auto cache = new list_cache;
	cache->str_data.record_num = record_count;
	cache->str_data.remain_blank_num = MAX_RECORD_ADD_COUNT;
	const size_t total_results_size = static_cast<size_t>(record_count) + MAX_RECORD_ADD_COUNT;
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cache->str_data.dev_total_number), sizeof(size_t)), true, nullptr);
	gpuErrchk(cudaMemcpy(cache->str_data.dev_total_number, &total_results_size, sizeof(size_t), cudaMemcpyHostToDevice), true, nullptr);
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cache->str_data.dev_cache_str),
		total_results_size * MAX_PATH_LENGTH), true, get_cache_info(key, cache).c_str());
	gpuErrchk(cudaMemset(cache->str_data.dev_cache_str, 0, MAX_PATH_LENGTH), true, nullptr);
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cache->dev_output), total_results_size), true,
		get_cache_info(key, cache).c_str());
	cache->is_cache_valid = true;
	cache->is_match_done = false;
	cache->is_output_done = 0;

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream), true, nullptr);
	unsigned address_offset = 0;
	for (const std::string& record : records_vec)
	{
		const auto target_address = cache->str_data.dev_cache_str + address_offset;
		gpuErrchk(cudaMemcpyAsync(target_address, record.c_str(), record.length(), cudaMemcpyHostToDevice, stream),
			true, nullptr);
		cache->str_data.record_hash.insert(hasher(record));
		++address_offset;
	}
	gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
	gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
	cache_map.insert(std::make_pair(key, cache));
	env->ReleaseStringUTFChars(key_jstring, _key);
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    addRecordsToCache
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_addRecordsToCache
(JNIEnv* env, jobject, jstring key_jstring, jobjectArray records)
{
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	const std::string key(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
	const auto records_num = env->GetArrayLength(records);
	std::vector<std::string> records_vec;
	for (int i = 0; i < records_num; ++i)
	{
		const auto record_jstring = reinterpret_cast<jstring>(env->GetObjectArrayElement(records, i));
		const auto record = env->GetStringUTFChars(record_jstring, nullptr);
		records_vec.emplace_back(record);
		env->ReleaseStringUTFChars(record_jstring, record);
		env->DeleteLocalRef(record_jstring);
	}
	add_records_to_cache(key, records_vec);
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    removeRecordsFromCache
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_removeRecordsFromCache
(JNIEnv* env, jobject, jstring key_jstring, jobjectArray records)
{
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	const std::string key(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
	const auto records_num = env->GetArrayLength(records);
	std::vector<std::string> records_vec;
	for (int i = 0; i < records_num; ++i)
	{
		const auto record_jstring = reinterpret_cast<jstring>(env->GetObjectArrayElement(records, i));
		const auto record = env->GetStringUTFChars(record_jstring, nullptr);
		records_vec.emplace_back(record);
		env->ReleaseStringUTFChars(record_jstring, record);
		env->DeleteLocalRef(record_jstring);
	}
	remove_records_from_cache(key, records_vec);
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    clearCache
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_clearCache
(JNIEnv* env, jobject, jstring key_jstring)
{
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	clear_cache(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    isCudaAvailable
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_CudaAccelerator_isCudaAvailable
(JNIEnv*, jobject)
{
	cudaError_t cudaStatus;
	int device_num = 0;
	cudaStatus = cudaGetDeviceCount(&device_num);
	if (cudaStatus != cudaSuccess)
	{
		return FALSE;
	}
	if (device_num > 0)
	{
		void* test_ptr;
		bool ret = FALSE;
		// 测试分配内存
		do
		{
			cudaStatus = cudaMalloc(&test_ptr, 1);
			if (cudaStatus == cudaSuccess)
			{
				ret = TRUE;
			}
		} while (false);
		cudaFree(test_ptr);
		return ret;
	}
	return FALSE;
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    isCacheValid
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_CudaAccelerator_isCacheValid
(JNIEnv* env, jobject, jstring key_jstring)
{
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	auto iter = cache_map.find(_key);
	if (iter == cache_map.end())
	{
		return FALSE;
	}
	return iter->second->is_cache_valid;
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    clearAllCache
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_clearAllCache
(JNIEnv*, jobject)
{
	clear_all_cache();
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    getCudaMemUsage
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_CudaAccelerator_getCudaMemUsage
(JNIEnv*, jobject)
{
	size_t avail;
	size_t total;
	cudaMemGetInfo(&avail, &total);
	const size_t used = total - avail;
	return static_cast<long>(used * 100 / total);
}

/**
 * \brief 将显卡计算的结果保存到hashmap中
 */
void collect_results(JNIEnv* thread_env, jobject result_collector, std::atomic_uint& result_counter,
	const unsigned max_results, const std::vector<std::string>& search_case_vec)
{
	jclass biconsumer_class = thread_env->GetObjectClass(result_collector);
	jmethodID collector = thread_env->GetMethodID(biconsumer_class, "accept", "(Ljava/lang/Object;Ljava/lang/Object;)V");
	bool all_complete;
	const auto stop_func = [&]
	{
		return is_stop() || result_counter.load() >= max_results;
	};
	auto _collect_func = [&](const std::string& _key, char _matched_record_str[MAX_PATH_LENGTH], unsigned* matched_number)
	{
		++result_counter;
		auto record_jstring = thread_env->NewStringUTF(_matched_record_str);
		auto key_jstring = thread_env->NewStringUTF(_key.c_str());
		thread_env->CallVoidMethod(result_collector, collector, key_jstring, record_jstring);
		thread_env->DeleteLocalRef(record_jstring);
		thread_env->DeleteLocalRef(key_jstring);
		++* matched_number;
	};
	do
	{
		//尝试退出
		all_complete = true;
		for (const auto& [key, val] : cache_map)
		{
			if (stop_func())
			{
				break;
			}
			if (!val->is_match_done.load())
			{
				//发现仍然有结果未计算完，设置退出标志为false，跳到下一个计算结果
				all_complete = false;
				continue;
			}
			if (int expected = 0; !val->is_output_done.compare_exchange_strong(expected, 1))
			{
				continue;
			}
			unsigned matched_number = 0;
			//复制结果数组到host，dev_output下标对应dev_cache中的下标，若dev_output[i]中的值为1，则对应dev_cache[i]字符串匹配成功
			const auto output_ptr = new char[val->str_data.record_num + val->str_data.remain_blank_num];
			//将dev_output拷贝到output_ptr
			gpuErrchk(cudaMemcpy(output_ptr, val->dev_output, val->str_data.record_num, cudaMemcpyDeviceToHost), false,
				"collect results failed");
			for (size_t i = 0; i < val->str_data.record_num.load(); ++i)
			{
				if (stop_func())
				{
					break;
				}
				//dev_cache[i]字符串匹配成功
				if (static_cast<bool>(output_ptr[i]))
				{
					char matched_record_str[MAX_PATH_LENGTH]{ 0 };
					const auto str_address = val->str_data.dev_cache_str + i;
					//拷贝GPU中的字符串到host
					gpuErrchk(cudaMemcpy(matched_record_str, str_address, MAX_PATH_LENGTH, cudaMemcpyDeviceToHost), false, "collect results failed");
					// 判断文件和文件夹
					if (search_case_vec.empty())
					{
						_collect_func(key, matched_record_str, &matched_number);
					}
					else
					{
						if (std::find(search_case_vec.begin(), search_case_vec.end(), "f") != search_case_vec.end())
						{
							if (is_dir_or_file(matched_record_str) == 1)
							{
								_collect_func(key, matched_record_str, &matched_number);
							}
						}
						else if (std::find(search_case_vec.begin(), search_case_vec.end(), "d") != search_case_vec.end())
						{
							if (is_dir_or_file(matched_record_str) == 0)
							{
								_collect_func(key, matched_record_str, &matched_number);
							}
						}
						else
						{
							_collect_func(key, matched_record_str, &matched_number);
						}
					}
				}
			}
			matched_result_number_map.insert(std::make_pair(key, matched_number));
			val->is_output_done = 2;
			delete[] output_ptr;
		}
	} while (!all_complete && !is_stop() && result_counter.load() < max_results);
	thread_env->DeleteLocalRef(biconsumer_class);
}

/**
 * \brief 检查是文件还是文件夹
 * \param path path
 * \return 如果是文件返回1，文件夹返回0，错误返回-1
 */
int is_dir_or_file(const char* path)
{
	const auto w_path = string2wstring(path);
	DWORD dwAttrib = GetFileAttributes(w_path.c_str());
	if (dwAttrib != INVALID_FILE_ATTRIBUTES)
	{
		if (dwAttrib & FILE_ATTRIBUTE_DIRECTORY)
		{
			return 0;
		}
		return 1;
	}
	return -1;
}

std::wstring string2wstring(const std::string& str)
{
	std::wstring result;
	//获取缓冲区大小，并申请空间，缓冲区大小按字符计算  
	const int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
	const auto buffer = new TCHAR[len + 1];
	//多字节编码转换成宽字节编码  
	MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), buffer, len);
	buffer[len] = '\0';
	//删除缓冲区并返回值  
	result.append(buffer);
	delete[] buffer;
	return result;
}

/**
 * \brief 等待clear_cache执行完成，用于add_one_record_to_cache以及remove_one_record_from_cache方法，防止在缓存被删除后读取到空指针
 */
void wait_for_clear_cache()
{
	while (clear_cache_flag.load())
		std::this_thread::yield();
}

/**
 * \brief 根据java数组search_case构建search_case_vector
 * \param env java运行环境指针
 * \param search_case_vec output
 * \param search_case search_case，有f，d，case，full，分别代表仅匹配文件，仅匹配文件夹，不忽略大小写匹配，全字匹配（CUDA不支持文件和文件夹判断，f和d将被忽略）
 */
void generate_search_case(JNIEnv* env, std::vector<std::string>& search_case_vec, const jobjectArray search_case)
{
	const auto search_case_len = env->GetArrayLength(search_case);
	for (jsize i = 0; i < search_case_len; ++i)
	{
		if (const auto search_case_str = env->GetObjectArrayElement(search_case, i); search_case_str != nullptr)
		{
			const auto tmp_search_case_str = reinterpret_cast<jstring>(search_case_str);
			auto search_case_chars = env->GetStringUTFChars(tmp_search_case_str, nullptr);
			search_case_vec.emplace_back(search_case_chars);
			env->ReleaseStringUTFChars(tmp_search_case_str, search_case_chars);
			env->DeleteLocalRef(tmp_search_case_str);
		}
	}
}

/**
 * \brief 检查是否已经存在该缓存
 * \param key key
 * \return true如果已经存在
 */
bool has_cache(const std::string& key)
{
	wait_for_clear_cache();
	return cache_map.find(key) != cache_map.end();
}

/**
 * \brief 锁住clear_cache方法，用于all_one_record_to_cache和remove_one_record_from_cache中
 * \param thread_counter 记录当前有多少线程调用该方法
 */
void lock_clear_cache(std::atomic_uint& thread_counter)
{
	std::lock_guard lock_guard(modify_cache_lock);
	wait_for_clear_cache();
	++thread_counter;
}

/**
 * \brief 方法最后调用，代表当前线程退出
 * \param thread_counter 记录线程数量
 */
void free_clear_cache(std::atomic_uint& thread_counter)
{
	--thread_counter;
}

bool is_record_repeat(const std::string& record, const list_cache* cache)
{
	const auto record_hash = hasher(record);
	return cache->str_data.record_hash.find(record_hash) != cache->str_data.record_hash.end();
}

/**
 * \brief 添加一个record到cache
 * \param key key
 * \param records records
 */
void add_records_to_cache(const std::string& key, const std::vector<std::string>& records)
{
	//锁住clear_cache，防止添加时缓存被清除
	lock_clear_cache(add_record_thread_count);
	if (cache_map.find(key) != cache_map.end())
	{
		const auto& cache = cache_map.at(key);
		//对当前cache加锁，防止一边add一边remove导致脏数据产生
		std::lock_guard lock_guard(cache->str_data.lock);
		cudaStream_t stream;
		gpuErrchk(cudaStreamCreate(&stream), true, nullptr);
		for (auto&& record : records)
		{
			const auto record_len = record.length();
			if (record_len >= MAX_PATH_LENGTH)
			{
				continue;
			}
			if (cache->is_cache_valid)
			{
				if (const bool is_repeat = is_record_repeat(record, cache); !is_repeat)
				{
					if (cache->str_data.remain_blank_num.load() > 0)
					{
						const auto index = cache->str_data.record_num.load();
						//计算下一个空位的内存地址
						const auto target_address = cache->str_data.dev_cache_str + index;
						//记录到下一空位内存地址target_address
						gpuErrchk(cudaMemcpyAsync(target_address, record.c_str(), record_len,
							cudaMemcpyHostToDevice, stream), true, get_cache_info(key, cache).c_str());
#ifdef DEBUG_OUTPUT
						std::cout << "successfully add record: " << record << "    key: " << key << std::endl;
#endif
						++cache->str_data.record_num;
						--cache->str_data.remain_blank_num;
						cache->str_data.record_hash.insert(hasher(record));
					}
					else
					{
						// 无空位，有文件丢失，使cache无效
						cache->is_cache_valid = false;
						break;
					}
				}
			}
		}
		gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
		gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
	}
	//释放clear_cache锁
	free_clear_cache(add_record_thread_count);
}

/**
 * \brief 从key对应的record中删除一个记录
 * \param key key
 * \param records records
 */
void remove_records_from_cache(const std::string& key, const std::vector<std::string>& records)
{
	lock_clear_cache(remove_record_thread_count);
	if (cache_map.find(key) != cache_map.end())
	{
		const auto& cache = cache_map.at(key);
		std::lock_guard lock_guard(cache->str_data.lock);
		cudaStream_t stream;
		gpuErrchk(cudaStreamCreate(&stream), true, nullptr);
		for (auto&& record : records)
		{
			if (const auto record_len = record.length(); record_len >= MAX_PATH_LENGTH)
			{
				continue;
			}
			if (cache->is_cache_valid)
			{
				char tmp[MAX_PATH_LENGTH]{ 0 };
				for (size_t i = 0; i < cache->str_data.record_num; ++i)
				{
					const auto str_address = cache->str_data.dev_cache_str + i;
					//拷贝GPU中的字符串到tmp
					gpuErrchk(cudaMemcpy(tmp, str_address,
						MAX_PATH_LENGTH, cudaMemcpyDeviceToHost), true,
						get_cache_info(key, cache).c_str());
					if (record == tmp)
					{
#ifdef DEBUG_OUTPUT
						printf("removing record: %s\n", record.c_str());
#endif
						//成功找到字符串
						const auto last_index = cache->str_data.record_num - 1;
						const auto last_str_address = cache->str_data.dev_cache_str + last_index;
						//复制最后一个结果到当前空位
						gpuErrchk(cudaMemcpyAsync(str_address, last_str_address,
							MAX_PATH_LENGTH, cudaMemcpyDeviceToDevice, stream), true, get_cache_info(key, cache).c_str());
						gpuErrchk(cudaMemsetAsync(last_str_address, 0, MAX_PATH_LENGTH, stream), true, nullptr);
						--cache->str_data.record_num;
						++cache->str_data.remain_blank_num;
						cache->str_data.record_hash.unsafe_erase(hasher(record));
						break;
					}
				}
			}
		}
		gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
		gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
	}
	//释放clear_cache锁
	free_clear_cache(remove_record_thread_count);
}

/**
 * \brief 等待add_one_record_to_cache和remove_one_record_from_cache方法全部退出
 */
void wait_for_add_or_remove_record()
{
	while (add_record_thread_count.load() != 0 || remove_record_thread_count.load() != 0)
		std::this_thread::yield();
}

/**
 * \brief 对add_one_record_to_cache和remove_one_record_from_cache方法加锁
 */
void lock_add_or_remove_result()
{
	std::lock_guard lock_guard(modify_cache_lock);
	wait_for_add_or_remove_record();
	clear_cache_flag = true;
}

void free_add_or_remove_results()
{
	clear_cache_flag = false;
}

void clear_cache(const std::string& key)
{
	//对add_one_record_to_cache和remove_one_record_from_cache方法加锁
	lock_add_or_remove_result();
	try
	{
		//对自身加锁，防止多个线程同时清除cache
		static std::mutex clear_cache_lock;
		std::lock_guard clear_cache_lock_guard(clear_cache_lock);
		const auto cache = cache_map.at(key);
		cache->is_cache_valid = false;
		{
			//对当前cache加锁
			std::lock_guard lock_guard(cache->str_data.lock);
			gpuErrchk(cudaFree(cache->str_data.dev_cache_str), false, get_cache_info(key, cache).c_str());
			gpuErrchk(cudaFree(cache->dev_output), false, get_cache_info(key, cache).c_str());
			gpuErrchk(cudaFree(cache->str_data.dev_total_number), true, nullptr);
		}
		delete cache;
		cache_map.unsafe_erase(key);
	}
	catch (std::out_of_range&)
	{
	}
	catch (std::exception& e)
	{
		fprintf(stderr, "clear cache failed: %s\n", e.what());
	}
	free_add_or_remove_results();
}

void clear_all_cache()
{
	std::vector<std::string> all_keys_vec;
	for (auto& [key, _] : cache_map)
	{
		all_keys_vec.emplace_back(key);
	}
	for (auto& key : all_keys_vec)
	{
		clear_cache(key);
	}
}

void release_all()
{
	clear_all_cache();
	free_cuda_search_memory();
	free_stop_signal();
}
