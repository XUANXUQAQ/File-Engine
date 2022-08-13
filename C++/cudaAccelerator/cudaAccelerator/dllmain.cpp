// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <concurrent_unordered_map.h>
#include "kernels.cuh"
#include "file_engine_dllInterface_CudaAccelerator.h"
#include "cache.h"
#include "constans.h"
#include "cuda_copy_vector_util.h"

#pragma comment(lib, "cudart.lib")

static concurrency::concurrent_unordered_map<std::string, list_cache*> cache_map;

void clear_cache(const std::string& key);
void init_cache(std::string& key, const std::vector<std::string>& records);
bool has_cache(const std::string& key);
void add_one_record_to_cache(const std::string& key, const std::string& record);
void generate_search_case(JNIEnv* env, std::vector<std::string>& search_case_vec, jobjectArray search_case);
void collect_results(JNIEnv* env, jobject output);

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    resetAllResultStatus
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_resetAllResultStatus
(JNIEnv*, jobject)
{
	// 初始化is_match_done is_output_done为false
	for (const auto& each : cache_map)
	{
		each.second->is_match_done = false;
		each.second->is_output_done = false;
		cudaMemset(each.second->dev_output, 0,
		           each.second->str_data.record_num + each.second->str_data.remain_blank_num);
	}
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    match
 * Signature: ([Ljava/lang/String;ZLjava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[ZLjava/util/concurrent/ConcurrentSkipListSet;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_match
(JNIEnv* env, jobject, jobjectArray search_case, jboolean is_ignore_case, jstring search_text,
 jobjectArray keywords, jobjectArray keywords_lower, jbooleanArray is_keyword_path, jobject output)
{
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
	const auto is_keyword_path_ptr = new bool[keywords_length];
	for (jsize i = 0; i < keywords_length; ++i)
	{
		auto keywords_str = env->GetObjectArrayElement(keywords, i);
		auto tmp_keywords_str = static_cast<jstring>(keywords_str);
		auto keywords_chars = env->GetStringUTFChars(tmp_keywords_str, nullptr);
		keywords_vec.emplace_back(keywords_chars);
		env->ReleaseStringUTFChars(tmp_keywords_str, keywords_chars);

		keywords_str = env->GetObjectArrayElement(keywords_lower, i);
		tmp_keywords_str = static_cast<jstring>(keywords_str);
		keywords_chars = env->GetStringUTFChars(tmp_keywords_str, nullptr);
		keywords_lower_vec.emplace_back(keywords_chars);
		env->ReleaseStringUTFChars(tmp_keywords_str, keywords_chars);

		const auto val = env->GetBooleanArrayElements(is_keyword_path, nullptr);
		is_keyword_path_ptr[i] = *val == JNI_TRUE;
	}
	//复制全字匹配字符串 search_text
	const auto search_text_chars = env->GetStringUTFChars(search_text, nullptr);
	//GPU并行计算
	start_kernel(cache_map, search_case_vec, is_ignore_case, search_text_chars, keywords_vec, keywords_lower_vec,
	             is_keyword_path_ptr);
	collect_results(env, output);
	env->ReleaseStringUTFChars(search_text, search_text_chars);
	delete[] is_keyword_path_ptr;
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
	if (iter == cache_map.end())
	{
		return FALSE;
	}
	return iter->second->is_output_done.load();
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    hasCache
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_CudaAccelerator_hasCache
(JNIEnv* env, jobject, jstring key_jstring)
{
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	const std::string key(_key);
	const bool ret = has_cache(key);
	env->ReleaseStringUTFChars(key_jstring, _key);
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
	std::vector<std::string> vec;
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	std::string key(_key);
	const auto length = env->GetArrayLength(records);
	for (jsize i = 0; i < length; ++i)
	{
		const auto val = env->GetObjectArrayElement(records, i);
		const auto jstring_val = static_cast<jstring>(val);
		const auto str_len = env->GetStringLength(jstring_val);
		if (str_len > MAX_PATH_LENGTH)
		{
			return;
		}
		const auto record = env->GetStringUTFChars(jstring_val, nullptr);
		vec.emplace_back(record);
		env->ReleaseStringUTFChars(jstring_val, record);
	}
	init_cache(key, vec);
	env->ReleaseStringUTFChars(key_jstring, _key);
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    addOneRecordToCache
 * Signature: (Ljava/lang/String;Ljava/lang/String;)Z
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_addOneRecordToCache
(JNIEnv* env, jobject, jstring key_jstring, jstring record_jstring)
{
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	const std::string key(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
	const auto _record = env->GetStringUTFChars(record_jstring, nullptr);
	const std::string record(_record);
	env->ReleaseStringUTFChars(record_jstring, _record);
	if (record.length() > MAX_PATH_LENGTH)
	{
		return;
	}
	add_one_record_to_cache(key, record);
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
		}
		while (false);
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
	for (auto& each : cache_map)
	{
		clear_cache(each.first);
	}
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

void collect_results(JNIEnv* env, jobject output)
{
	//获取java中ConcurrentHashMap和ConcurrentLinkedQueue的类和方法
	const auto concurrent_hash_map_class = env->GetObjectClass(output);
	const auto contains_key_func = env->GetMethodID(concurrent_hash_map_class, "containsKey", "(Ljava/lang/Object;)Z");
	const auto get_func = env->GetMethodID(concurrent_hash_map_class, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
	const auto put_func = env->GetMethodID(concurrent_hash_map_class, "put",
	                                       "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
	const auto concurrent_linked_queue_class = env->FindClass("java/util/concurrent/ConcurrentLinkedQueue");
	const auto add_func = env->GetMethodID(concurrent_linked_queue_class, "add", "(Ljava/lang/Object;)Z");
	const auto concurrent_linked_queue_constructor = env->GetMethodID(concurrent_linked_queue_class, "<init>", "()V");

	bool all_complete;
	do
	{
		all_complete = true;
		for (const auto& each : cache_map)
		{
			if (!each.second->is_match_done.load())
			{
				all_complete = false;
				continue;
			}
			//复制结果数组到host，dev_output下标对应dev_cache中的下标，若dev_output[i]中的值为1，则对应dev_cache[i]字符串匹配成功
			const auto output_ptr = new char[each.second->str_data.record_num];
			//将dev_output拷贝到output_ptr
			cudaMemcpy(output_ptr, each.second->dev_output, each.second->str_data.record_num, cudaMemcpyDeviceToHost);
			for (unsigned long i = 0; i < each.second->str_data.record_num; ++i)
			{
				//dev_cache[i]字符串匹配成功
				if (static_cast<bool>(output_ptr[i]))
				{
					char tmp_matched_record_str[MAX_PATH_LENGTH]{0};
					//计算GPU内存偏移
					const auto str_address_ptr = reinterpret_cast<unsigned long long>
						(each.second->str_data.dev_cache_str_ptr) + i * sizeof(unsigned long long);
					unsigned long long str_address;
					cudaMemcpy(&str_address, reinterpret_cast<void*>(str_address_ptr),
					           sizeof(unsigned long long), cudaMemcpyDeviceToHost);
					//拷贝GPU中的字符串到host
					cudaMemcpy(tmp_matched_record_str, reinterpret_cast<void*>(str_address),
					           each.second->str_data.str_length[i], cudaMemcpyDeviceToHost);
					const auto matched_record = env->NewStringUTF(tmp_matched_record_str);
					const auto key_jstring = env->NewStringUTF(each.first.c_str());
					if (env->CallBooleanMethod(output, contains_key_func, key_jstring))
					{
						//已经存在该key容器
						const jobject container = env->CallObjectMethod(output, get_func, key_jstring);
						env->CallBooleanMethod(container, add_func, matched_record);
					}
					else
					{
						//不存在该key的容器
						const jobject container = env->NewObject(concurrent_linked_queue_class,
						                                         concurrent_linked_queue_constructor);
						env->CallBooleanMethod(container, add_func, matched_record);
						env->CallObjectMethod(output, put_func, key_jstring, container);
					}
				}
			}
			each.second->is_output_done = true;
			delete[] output_ptr;
		}
	}
	while (!all_complete);
}

void generate_search_case(JNIEnv* env, std::vector<std::string>& search_case_vec, const jobjectArray search_case)
{
	const auto search_case_len = env->GetArrayLength(search_case);
	for (jsize i = 0; i < search_case_len; ++i)
	{
		const auto search_case_str = env->GetObjectArrayElement(search_case, i);
		if (search_case_str != nullptr)
		{
			const auto tmp_search_case_str = static_cast<jstring>(search_case_str);
			auto search_case_chars = env->GetStringUTFChars(tmp_search_case_str, nullptr);
			search_case_vec.emplace_back(search_case_chars);
			env->ReleaseStringUTFChars(tmp_search_case_str, search_case_chars);
		}
	}
}

void init_cache(std::string& key, const std::vector<std::string>& records)
{
	constexpr int blank_num = 100;
	auto cache = new list_cache;
	const auto total_size = records.size();
	cache->str_data.record_num = total_size;
	cache->str_data.remain_blank_num = blank_num;
	gpuErrchk(
		cudaMalloc(reinterpret_cast<void**>(&cache->str_data.dev_cache_str_ptr),
			(total_size + blank_num) * sizeof(unsigned long long)), true)
	gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cache->dev_output), total_size + blank_num), true)
	cache->is_cache_valid = true;
	cache->is_match_done = false;
	cache->is_output_done = false;
	char* tmp_ptr;
	int count = 0;
	for (auto& each_record : records)
	{
		const auto record_len = each_record.length();
		if (record_len < MAX_PATH_LENGTH)
		{
			const unsigned long long current_address = reinterpret_cast<unsigned long long>
				(cache->str_data.dev_cache_str_ptr)
				+ count * sizeof(unsigned long long);
			//复制字符串到tmp_ptr
			const auto bytes = record_len + 1;
			gpuErrchk(cudaMalloc(&tmp_ptr, bytes), true)
			gpuErrchk(cudaMemset(tmp_ptr, 0, bytes), true)
			gpuErrchk(cudaMemcpy(tmp_ptr, each_record.c_str(), record_len, cudaMemcpyHostToDevice), true)
			//记录tmp_ptr地址并存入cache->str_data.dev_cache_str_ptr
			const auto str_address = reinterpret_cast<unsigned long long>(tmp_ptr);
			gpuErrchk(cudaMemcpy(reinterpret_cast<void*>(current_address), &str_address, sizeof(unsigned long long),
				          cudaMemcpyHostToDevice), true)
			cache->str_data.str_length.emplace_back(bytes);
		}
		++count;
	}
	cache_map.insert(std::make_pair(key, cache));
}


bool has_cache(const std::string& key)
{
	return cache_map.find(key) != cache_map.end();
}

void add_one_record_to_cache(const std::string& key, const std::string& record)
{
	try
	{
		const auto record_len = record.length();
		if (record_len >= MAX_PATH_LENGTH)
		{
			return;
		}
		const auto& cache = cache_map.at(key);
		if (cache->str_data.remain_blank_num > 0)
		{
			const unsigned long long current_address = reinterpret_cast<unsigned long long>
				(cache->str_data.dev_cache_str_ptr)
				+ cache->str_data.record_num * sizeof(unsigned long long);

			++cache->str_data.record_num;
			--cache->str_data.remain_blank_num;

			char* tmp_ptr;
			const auto bytes = record_len + 1;
			gpuErrchk(cudaMalloc(&tmp_ptr, bytes), true)
			gpuErrchk(cudaMemset(tmp_ptr, 0, bytes), true)
			gpuErrchk(cudaMemcpy(tmp_ptr, record.c_str(), record_len, cudaMemcpyHostToDevice), true)

			const auto str_address = reinterpret_cast<unsigned long long>(tmp_ptr);

			gpuErrchk(cudaMemcpy(reinterpret_cast<void*>(current_address), &str_address, sizeof(unsigned long long),
				          cudaMemcpyHostToDevice), true)
			cache->str_data.str_length.emplace_back(bytes);
		}
		else
		{
			cache->is_cache_valid = false;
		}
	}
	catch (std::exception&)
	{
	}
}

void clear_cache(const std::string& key)
{
	try
	{
		const auto& cache = cache_map.at(key);
		for (unsigned long long i = 0; i < cache->str_data.record_num; ++i)
		{
			const auto str_address_ptr = reinterpret_cast<unsigned long long>(cache->str_data.dev_cache_str_ptr)
				+ i * sizeof(unsigned long long);
			unsigned long long str_address = 0;
			gpuErrchk(cudaMemcpy(&str_address, reinterpret_cast<const void*>(str_address_ptr),
				          sizeof(unsigned long long), cudaMemcpyDeviceToHost), false)
			gpuErrchk(cudaFree(reinterpret_cast<void*>(str_address)), false)
		}
		gpuErrchk(cudaFree(cache->str_data.dev_cache_str_ptr), false)
		gpuErrchk(cudaFree(cache->dev_output), false)
		delete cache;
		cache_map.unsafe_erase(key);
	}
	catch (std::exception&)
	{
	}
}
