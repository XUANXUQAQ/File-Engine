﻿// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <concurrent_unordered_map.h>
#include "kernels.cuh"
#include "file_engine_dllInterface_CudaAccelerator.h"
#include "cache.h"
#include "constans.h"
#include "cuda_copy_vector_util.h"

#pragma comment(lib, "cudart.lib")

void clear_cache(const std::string& key);
bool has_cache(const std::string& key);
void add_one_record_to_cache(const std::string& key, const std::string& record);
void remove_one_record_from_cache(const std::string& key, const std::string& record);
void generate_search_case(JNIEnv* env, std::vector<std::string>& search_case_vec, jobjectArray search_case);
void collect_results(JNIEnv* env, jobject output, jint max_results);
bool is_record_repeat(const std::string& record, const list_cache* cache);

//lock
inline void wait_for_clear_cache();
inline void wait_for_add_or_remove_record();
inline void lock_clear_cache(std::atomic_uint& thread_counter);
inline void free_clear_cache(std::atomic_uint& thread_counter);
inline void lock_add_or_remove_result();

static concurrency::concurrent_unordered_map<std::string, list_cache*> cache_map;
static std::atomic_bool clear_cache_flag(false);
static std::atomic_uint add_record_thread_count(0);
static std::atomic_uint remove_record_thread_count(0);
static std::mutex modify_cache_lock;
static std::hash<std::string> hasher;

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    initialize
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_initialize
(JNIEnv* env, jobject)
{
	init_stop_signal();
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
		val->is_output_done = false;
		cudaMemset(val->dev_output, 0,
			val->str_data.record_num + val->str_data.remain_blank_num);
	}
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
 * Signature: ([Ljava/lang/String;ZLjava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[ZLjava/util/concurrent/ConcurrentHashMap;I)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_match
(JNIEnv* env, jobject, jobjectArray search_case, jboolean is_ignore_case, jstring search_text,
	jobjectArray keywords, jobjectArray keywords_lower, jbooleanArray is_keyword_path, jobject output, jint max_results)
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
		auto tmp_keywords_str = reinterpret_cast<jstring>(keywords_str);
		auto keywords_chars = env->GetStringUTFChars(tmp_keywords_str, nullptr);
		keywords_vec.emplace_back(keywords_chars);
		env->ReleaseStringUTFChars(tmp_keywords_str, keywords_chars);

		keywords_str = env->GetObjectArrayElement(keywords_lower, i);
		tmp_keywords_str = reinterpret_cast<jstring>(keywords_str);
		keywords_chars = env->GetStringUTFChars(tmp_keywords_str, nullptr);
		keywords_lower_vec.emplace_back(keywords_chars);
		env->ReleaseStringUTFChars(tmp_keywords_str, keywords_chars);

		const auto val = env->GetBooleanArrayElements(is_keyword_path, nullptr);
		is_keyword_path_ptr[i] = *val == JNI_TRUE;
	}
	//复制全字匹配字符串 search_text
	const auto search_text_chars = env->GetStringUTFChars(search_text, nullptr);
	const auto streamCount = cache_map.size();
	const auto streams = new cudaStream_t[streamCount];
	//GPU并行计算
	start_kernel(cache_map, search_case_vec, is_ignore_case, search_text_chars, keywords_vec, keywords_lower_vec,
		is_keyword_path_ptr, streams, streamCount);
	collect_results(env, output, max_results);
	env->ReleaseStringUTFChars(search_text, search_text_chars);
	// 等待执行完成
	auto cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launch!\n", cudaStatus);
	delete[] is_keyword_path_ptr;
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
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	std::string key(_key);
	const jsize length = env->GetArrayLength(records);
	auto cache = new list_cache;
	cache->str_data.record_num = length;
	cache->str_data.remain_blank_num = MAX_RECORD_ADD_COUNT;
	const size_t total_results_size = static_cast<size_t>(length) + MAX_RECORD_ADD_COUNT;
	gpuErrchk(
		cudaMalloc(reinterpret_cast<void**>(&cache->str_data.dev_cache_str),
			total_results_size * MAX_PATH_LENGTH), true, get_cache_info(key, cache).c_str())
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cache->dev_output), total_results_size), true,
			get_cache_info(key, cache).c_str())
		cache->is_cache_valid = true;
	cache->is_match_done = false;
	cache->is_output_done = false;

	for (jsize i = 0; i < length; ++i)
	{
		const auto jstring_val = reinterpret_cast<jstring>(env->GetObjectArrayElement(records, i));
		const auto record = env->GetStringUTFChars(jstring_val, nullptr);
		const auto record_len = strlen(record);
		if (record_len < MAX_PATH_LENGTH)
		{
			const auto target_address = cache->str_data.dev_cache_str + i;
			gpuErrchk(cudaMemset(target_address, 0, MAX_PATH_LENGTH), true, nullptr)
				gpuErrchk(cudaMemcpy(target_address, record, record_len, cudaMemcpyHostToDevice),
					true, nullptr)
				cache->str_data.record_hash.insert(hasher(record));
		}
		env->ReleaseStringUTFChars(jstring_val, record);
	}
	cache_map.insert(std::make_pair(key, cache));
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
	add_one_record_to_cache(key, record);
}

/*
 * Class:     file_engine_dllInterface_CudaAccelerator
 * Method:    removeOneRecordFromCache
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_CudaAccelerator_removeOneRecordFromCache
(JNIEnv* env, jobject, jstring key_jstring, jstring record_jstring)
{
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	const std::string key(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
	const auto _record = env->GetStringUTFChars(record_jstring, nullptr);
	const std::string record(_record);
	env->ReleaseStringUTFChars(record_jstring, _record);
	remove_one_record_from_cache(key, record);
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
	std::vector<std::string> all_keys_vec;
	for (auto& each : cache_map)
	{
		all_keys_vec.emplace_back(each.first);
	}
	for (auto& key : all_keys_vec)
	{
		clear_cache(key);
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

/**
 * \brief 将显卡计算的结果保存到java的concurrent_hashmap中
 * \param env jni指针
 * \param output hashmap
 *				 output结构为ConcurrentHashMap<String, ConcurrentLinkedQueue<String>>，key为 disk,list[num],priority 格式，value为存放结果的容器
 */
void collect_results(JNIEnv* env, jobject output, jint max_resutls)
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
	int result_count = 0;
	do
	{
		//尝试退出
		all_complete = true;
		for (const auto& [key, val] : cache_map)
		{
			if (is_stop() || result_count >= max_resutls)
			{
				break;
			}
			if (!val->is_match_done.load())
			{
				//发现仍然有结果未计算完，设置退出标志为false，跳到下一个计算结果
				all_complete = false;
				continue;
			}
			if (val->is_output_done.load())
			{
				continue;
			}
#ifdef DEBUG_OUTPUT
			printf("collecting key: %s\n", key.c_str());
#endif
			//复制结果数组到host，dev_output下标对应dev_cache中的下标，若dev_output[i]中的值为1，则对应dev_cache[i]字符串匹配成功
			const auto output_ptr = new char[val->str_data.record_num + val->str_data.remain_blank_num];
			//将dev_output拷贝到output_ptr
			gpuErrchk(cudaMemcpy(output_ptr, val->dev_output, val->str_data.record_num, cudaMemcpyDeviceToHost), false,
				"collect results failed")
				for (size_t i = 0; i < val->str_data.record_num.load(); ++i)
				{
					if (is_stop())
					{
						break;
					}
					//dev_cache[i]字符串匹配成功
					if (static_cast<bool>(output_ptr[i]))
					{
						char tmp_matched_record_str[MAX_PATH_LENGTH]{ 0 };
						const auto str_address = val->str_data.dev_cache_str + i;
						//拷贝GPU中的字符串到host
						gpuErrchk(cudaMemcpy(tmp_matched_record_str, str_address, MAX_PATH_LENGTH, cudaMemcpyDeviceToHost), false, "collect results failed")
							const auto matched_record = env->NewStringUTF(tmp_matched_record_str);
						++result_count;
						const auto key_jstring = env->NewStringUTF(key.c_str());
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
			val->is_output_done = true;
			delete[] output_ptr;
		}
	} while (!all_complete && !is_stop());
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
inline void lock_clear_cache(std::atomic_uint& thread_counter)
{
	std::lock_guard lock_guard(modify_cache_lock);
	wait_for_clear_cache();
	++thread_counter;
}

/**
 * \brief 方法最后调用，代表当前线程退出
 * \param thread_counter 记录线程数量
 */
inline void free_clear_cache(std::atomic_uint& thread_counter)
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
 * \param record record
 */
void add_one_record_to_cache(const std::string& key, const std::string& record)
{
	const auto record_len = record.length();
	if (record_len >= MAX_PATH_LENGTH)
	{
		return;
	}
	//锁住clear_cache，防止添加时缓存被清除
	lock_clear_cache(add_record_thread_count);
	try
	{
		const auto& cache = cache_map.at(key);
		//对当前cache加锁，防止一边add一边remove导致脏数据产生
		std::lock_guard lock_guard(cache->str_data.lock);
		if (cache->is_cache_valid)
		{
			if (const bool is_repeat = is_record_repeat(record, cache); !is_repeat)
			{
				if (cache->str_data.remain_blank_num.load() > 0)
				{
#ifdef DEBUG_OUTPUT
					printf("adding record: %s\n", record.c_str());
#endif
					const auto index = cache->str_data.record_num.load();
					//计算下一个空位的内存地址
					const auto target_address = cache->str_data.dev_cache_str + index;
					//记录到下一空位内存地址target_address
					gpuErrchk(
						cudaMemcpy(target_address, record.c_str(), record_len,
							cudaMemcpyHostToDevice),
						true, get_cache_info(key, cache).c_str())
						++cache->str_data.record_num;
					--cache->str_data.remain_blank_num;
					cache->str_data.record_hash.insert(hasher(record));
				}
				else
				{
					// 无空位，有文件丢失，使cache无效
					cache->is_cache_valid = false;
				}
			}
		}
	}
	catch (std::out_of_range&)
	{
	}
	catch (std::exception& e)
	{
		printf("add record failed: %s\n", e.what());
	}
	//释放clear_cache锁
	free_clear_cache(add_record_thread_count);
}

/**
 * \brief 从key对应的record中删除一个记录
 * \param key key
 * \param record record
 */
void remove_one_record_from_cache(const std::string& key, const std::string& record)
{
	const auto record_len = record.length();
	if (record_len >= MAX_PATH_LENGTH)
	{
		return;
	}
	lock_clear_cache(remove_record_thread_count);
	try
	{
		const auto& cache = cache_map.at(key);
		std::lock_guard lock_guard(cache->str_data.lock);
		if (cache->is_cache_valid)
		{
			char tmp[MAX_PATH_LENGTH]{ 0 };
			for (size_t i = 0; i < cache->str_data.record_num; ++i)
			{
				const auto str_address = cache->str_data.dev_cache_str + i;
				//拷贝GPU中的字符串到tmp
				gpuErrchk(cudaMemcpy(tmp, str_address,
					MAX_PATH_LENGTH, cudaMemcpyDeviceToHost), true,
					get_cache_info(key, cache).c_str())
					if (record == tmp)
					{
#ifdef DEBUG_OUTPUT
						printf("removing record: %s\n", record.c_str());
#endif
						//成功找到字符串
						const auto last_index = cache->str_data.record_num - 1;
						const auto last_str_address = cache->str_data.dev_cache_str + last_index;
						//复制最后一个结果到当前空位
						gpuErrchk(
							cudaMemcpy(str_address, last_str_address,
								MAX_PATH_LENGTH, cudaMemcpyDeviceToDevice), true, get_cache_info(key, cache).c_str())
							--cache->str_data.record_num;
						++cache->str_data.remain_blank_num;
						cache->str_data.record_hash.unsafe_erase(hasher(record));
						break;
					}
			}
		}
	}
	catch (std::out_of_range&)
	{
	}
	catch (std::exception& e)
	{
		printf("remove record failed: %s\n", e.what());
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
inline void lock_add_or_remove_result()
{
	std::lock_guard lock_guard(modify_cache_lock);
	wait_for_add_or_remove_record();
	clear_cache_flag = true;
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
		{
			//对当前cache加锁
			std::lock_guard lock_guard(cache->str_data.lock);
			gpuErrchk(cudaFree(cache->str_data.dev_cache_str), false, get_cache_info(key, cache).c_str())
				gpuErrchk(cudaFree(cache->dev_output), false, get_cache_info(key, cache).c_str())
		}
		delete cache;
		cache_map.unsafe_erase(key);
	}
	catch (std::out_of_range&)
	{
	}
	catch (std::exception& e)
	{
		printf("clear cache failed: %s\n", e.what());
	}
	clear_cache_flag = false;
}
