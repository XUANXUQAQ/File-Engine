#include <concurrent_unordered_map.h>
#include "file_engine_dllInterface_gpu_OpenclAccelerator.h"
#include "cache.h"
#include "constans.h"
#include <Windows.h>
#include <string>
#include "file_utils.h"
#ifdef DEBUG_OUTPUT
#include <iostream>
#endif
#include "utf162gbk_val.h"
#include <thread>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <shared_mutex>
#include <d3d.h>
#include <dxgi.h>
#include <pdh.h>
#include <unordered_map>
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "pdh.lib")


void clear_cache(const std::string& key);
void clear_all_cache();
bool has_cache(const std::string& key);
void add_records_to_cache(const std::string& key, const std::vector<std::string>& records);
void remove_records_from_cache(const std::string& key, std::vector<std::string>& records);
void generate_search_case(JNIEnv* env, std::vector<std::string>& search_case_vec, jobjectArray search_case);
void collect_results(JNIEnv* thread_env, jobject result_collector, std::atomic_uint& result_counter,
	unsigned max_results, const std::vector<std::string>& search_case_vec);
bool is_record_repeat(const std::string& record, const list_cache* cache);
void release_all();
bool set_using_device(unsigned device_num);
std::vector<cl::Event*> start_kernel(const std::vector<std::string>& search_case,
	bool is_ignore_case,
	const char* search_text,
	const std::vector<std::string>& keywords,
	const std::vector<std::string>& keywords_lower_case,
	const bool* is_keyword_path);
size_t get_device_memory_used();
void send_restart_event();

concurrency::concurrent_unordered_map<std::string, list_cache*> cache_map;
std::shared_mutex cache_lock;
std::hash<std::string> hasher;
std::atomic_bool exit_flag = false;
Device current_device;
Memory<char> p_stop_signal;
JavaVM* jvm;
std::atomic_bool is_results_number_exceed = false;
IDXGIFactory* p_dxgi_factory = nullptr;
concurrency::concurrent_unordered_map<std::wstring, DXGI_ADAPTER_DESC> gpu_name_adapter_map;

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    getDevices
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_getDevices
(JNIEnv* env, jobject)
{
	std::vector<string> device_vec;
	auto&& devices = get_devices();
	const auto device_count = devices.size();
	for (size_t i = 0; i < device_count; ++i)
	{
		// 是GPU
		if (devices[i].is_gpu)
		{
			device_vec.emplace_back(devices[i].board_name);
		}
	}
	const auto gpu_device_count = device_vec.size();
	auto&& string_clazz = env->FindClass("java/lang/String");
	auto&& object_arr = env->NewObjectArray(static_cast<jsize>(gpu_device_count), string_clazz, nullptr);
	for (uint i = 0; i < gpu_device_count; ++i)
	{
		auto&& each_device_name = device_vec[i];
		env->SetObjectArrayElement(object_arr, i, env->NewStringUTF(each_device_name.c_str()));
	}
	env->DeleteLocalRef(string_clazz);
	return object_arr;
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    setDevice
 * Signature: (I)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_setDevice
(JNIEnv*, jobject, jint device_number_jint)
{
	std::unique_lock ulck(cache_lock);
	release_all();
	return set_using_device(device_number_jint);
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_release
(JNIEnv*, jobject)
{
	std::unique_lock ulck(cache_lock);
	exit_flag = true;
	p_stop_signal[0] = 1;
	p_stop_signal.write_to_device();
	release_all();
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    initialize
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_initialize
(JNIEnv* env, jobject)
{
	if (env->GetJavaVM(&jvm) != JNI_OK)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), "get JavaVM ptr failed.");
		return;
	}
	current_device = Device(select_device_with_id(0));
	p_stop_signal = Memory<char>(current_device, 1);
	p_stop_signal[0] = 0;
	p_stop_signal.write_to_device();
	if (CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&p_dxgi_factory)) != S_OK)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), "create dxgi factory failed.");
		return;
	}
	IDXGIAdapter* p_adapter = nullptr;
	for (UINT i = 0;
		p_dxgi_factory->EnumAdapters(i, &p_adapter) != DXGI_ERROR_NOT_FOUND;
		++i)
	{
		DXGI_ADAPTER_DESC adapter_desc;
		p_adapter->GetDesc(&adapter_desc);
		gpu_name_adapter_map.insert(std::make_pair(adapter_desc.Description, adapter_desc));
	}
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    resetAllResultStatus
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_resetAllResultStatus
(JNIEnv*, jobject)
{
	std::unique_lock ulck(cache_lock);
	p_stop_signal[0] = 0;
	p_stop_signal.write_to_device();
	// 初始化is_match_done is_output_done为false
	for (auto& [_, cache_ptr] : cache_map)
	{
		cache_ptr->is_match_done = false;
		cache_ptr->matched_number = 0;
		cache_ptr->is_output_done = 0;
		cache_ptr->dev_output->reset();
	}
	is_results_number_exceed = false;
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    stopCollectResults
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_stopCollectResults
(JNIEnv*, jobject)
{
	p_stop_signal[0] = 1;
	p_stop_signal.write_to_device();
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    match
 * Signature: ([Ljava/lang/String;ZLjava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[ZIILjava/util/function/BiConsumer;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_match
(JNIEnv* env, jobject, jobjectArray search_case, jboolean is_ignore_case, jstring search_text,
	jobjectArray keywords, jobjectArray keywords_lower, jbooleanArray is_keyword_path, jint max_results,
	jint result_collect_thread_num, jobject result_collector)
{
	if (cache_map.empty())
	{
		return;
	}
	std::unique_lock ulck(cache_lock);
	//生成搜索条件 search_case_vec
	std::vector<std::string> search_case_vec;
	if (search_case != nullptr)
	{
		generate_search_case(env, search_case_vec, search_case);
	}
	//生成搜索关键字 keywords_vec keywords_lower_vec is_keyword_path_ptr
	const auto keywords_length = env->GetArrayLength(keywords);
	if (keywords_length > MAX_KEYWORDS_NUMBER)
	{
		fprintf(stderr, "too many keywords.\n");
		return;
	}
	std::vector<std::string> keywords_vec;
	std::vector<std::string> keywords_lower_vec;
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
	std::atomic_uint result_counter = 0;
	std::vector<std::thread> collect_threads_vec;
	collect_threads_vec.reserve(result_collect_thread_num);
	for (int i = 0; i < result_collect_thread_num; ++i)
	{
		auto&& collect_func = [&]
		{
			JNIEnv* thread_env = nullptr;
			JavaVMAttachArgs args{ JNI_VERSION_10, nullptr, nullptr };
			if (jvm->AttachCurrentThread(reinterpret_cast<void**>(&thread_env), &args) != JNI_OK)
			{
				fprintf(stderr, "get thread JNIEnv ptr failed");
				return;
			}
			collect_results(thread_env, result_collector, result_counter, max_results, search_case_vec);
			jvm->DetachCurrentThread();
		};
		collect_threads_vec.emplace_back(collect_func);
	}
	//GPU并行计算
	auto&& events_vec = start_kernel(search_case_vec,
		is_ignore_case, 
		search_text_chars,
		keywords_vec,
		keywords_lower_vec,
		is_keyword_path_ptr);
	collect_results(env, result_collector, result_counter, max_results, search_case_vec);
	for (auto&& each : collect_threads_vec)
	{
		if (each.joinable())
		{
			each.join();
		}
	}
	env->ReleaseStringUTFChars(search_text, search_text_chars);
	for (const auto event_ptr : events_vec)
	{
		delete event_ptr;
	}
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    isMatchDone
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_isMatchDone
(JNIEnv* env, jobject, jstring key_jstring)
{
	if (is_results_number_exceed.load())
	{
		return JNI_TRUE;
	}
	std::shared_lock lck(cache_lock);
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	auto iter = cache_map.find(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
	if (iter == cache_map.end())
	{
		return JNI_FALSE;
	}
	return iter->second->is_output_done.load() == 2;
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    matchedNumber
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_matchedNumber
(JNIEnv* env, jobject, jstring key_jstring)
{
	std::shared_lock lck(cache_lock);
	const auto key = env->GetStringUTFChars(key_jstring, nullptr);
	auto&& matched_number_iter = cache_map.find(key);
	env->ReleaseStringUTFChars(key_jstring, key);
	if (matched_number_iter == cache_map.end())
	{
		return 0;
	}
	auto&& cache_ptr = matched_number_iter->second;
	return static_cast<jint>(cache_ptr->matched_number);
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    hasCache
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_hasCache
(JNIEnv*, jobject)
{
	return !cache_map.empty();
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    isCacheExist
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_isCacheExist
(JNIEnv* env, jobject, jstring key_jstring)
{
	std::shared_lock lck(cache_lock);
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	const std::string key(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
	return has_cache(key);
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    initCache
 * Signature: (Ljava/lang/String;Ljava/util/function/Supplier;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_initCache
(JNIEnv* env, jobject, jstring key_jstring, jobject record_supplier)
{
	const jclass supplier_class = env->GetObjectClass(record_supplier);
	const jmethodID get_function = env->GetMethodID(supplier_class, "get", "()Ljava/lang/Object;");
	std::vector<std::string> records_vec;
	unsigned record_count = 0;
	while (true)
	{
		const jobject record_from_supplier = env->CallObjectMethod(record_supplier, get_function);
		if (record_from_supplier == nullptr)
		{
			break;
		}
		const auto jstring_val = reinterpret_cast<jstring>(record_from_supplier);
		const auto record = env->GetStringUTFChars(jstring_val, nullptr);
		if (const auto record_len = strlen(record); record_len < MAX_PATH_LENGTH)
		{
			records_vec.emplace_back(record);
			++record_count;
		}
		env->ReleaseStringUTFChars(jstring_val, record);
		env->DeleteLocalRef(record_from_supplier);
	}
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	std::string key(_key);
	auto cache = new list_cache;
	cache->str_data.record_num = record_count;
	cache->str_data.remain_blank_num = MAX_RECORD_ADD_COUNT;
	const size_t total_results_size = static_cast<size_t>(record_count) + MAX_RECORD_ADD_COUNT;

	cache->str_data.dev_total_number = new Memory<size_t>(current_device, 1);
	(*cache->str_data.dev_total_number)[0] = total_results_size;
	cache->str_data.dev_total_number->write_to_device();

	const auto alloc_bytes = total_results_size * MAX_PATH_LENGTH;
	cache->str_data.dev_cache_str = new Memory<char>(current_device, alloc_bytes);

	cache->dev_output = new Memory<char>(current_device, total_results_size);
	cache->is_cache_valid = true;
	cache->is_match_done = false;
	cache->is_output_done = 0;

	size_t address_offset = 0;
	for (const std::string& record : records_vec)
	{
		strcpy_s(&(*cache->str_data.dev_cache_str)[address_offset * MAX_PATH_LENGTH], MAX_PATH_LENGTH, record.c_str());
		cache->str_data.record_hash.insert(hasher(record));
		++address_offset;
	}
	cache->str_data.dev_cache_str->write_to_device();
	cache->str_data.dev_cache_str->delete_host_buffer();
	cache_map.insert(std::make_pair(key, cache));
#ifdef DEBUG_OUTPUT
	std::cout << "cache initialized. key: " << key << " cache size: " << records_vec.size() << std::endl;
#endif // DEBUG_OUTPUT
	env->ReleaseStringUTFChars(key_jstring, _key);
	env->DeleteLocalRef(supplier_class);
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    addRecordsToCache
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_addRecordsToCache
(JNIEnv* env, jobject, jstring key_jstring, jobjectArray records)
{
	std::unique_lock ulck(cache_lock);
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
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    removeRecordsFromCache
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_removeRecordsFromCache
(JNIEnv* env, jobject, jstring key_jstring, jobjectArray records)
{
	std::unique_lock ulck(cache_lock);
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
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    clearCache
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_clearCache
(JNIEnv* env, jobject, jstring key_jstring)
{
	std::unique_lock ulck(cache_lock);
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	clear_cache(_key);
	env->ReleaseStringUTFChars(key_jstring, _key);
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    isOpenCLAvailable
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_isOpenCLAvailable
(JNIEnv*, jobject)
{
	auto&& devices = get_devices();
	if (devices.empty())
	{
		return JNI_FALSE;
	}
	jboolean ret = JNI_FALSE;
	for (auto&& each : devices)
	{
		if (each.is_gpu)
		{
			ret = JNI_TRUE;
			break;
		}
	}
	return ret;
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    isCacheValid
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_isCacheValid
(JNIEnv* env, jobject, jstring key_jstring)
{
	std::shared_lock lck(cache_lock);
	const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
	auto iter = cache_map.find(_key);
	if (iter == cache_map.end())
	{
		return JNI_FALSE;
	}
	return iter->second->is_cache_valid;
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    clearAllCache
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_clearAllCache
(JNIEnv*, jobject)
{
	std::unique_lock ulck(cache_lock);
	clear_all_cache();
}

/*
 * Class:     file_engine_dllInterface_gpu_OpenclAccelerator
 * Method:    getGPUMemUsage
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_gpu_OpenclAccelerator_getGPUMemUsage
(JNIEnv*, jobject)
{
	// const auto mem_free = current_device.info.cl_device.getInfo<CL_DEVICE_GLOBAL_FREE_MEMORY_AMD>();
	// if (!mem_free.empty())
	// {
	//     const auto total_mem = current_device.info.memory;
	//     const auto mem_used = total_mem - mem_free[0] * 1024;
	//     return static_cast<jint>(mem_used * 100 / total_mem);
	// }
	const auto total_mem = current_device.info.memory;
	auto&& memory_used = get_device_memory_used();
	if (memory_used == INFINITE)
	{
		return 100;
	}
	return static_cast<jint>(memory_used * 100 / total_mem);
}

/**
 * \brief 将显卡计算的结果保存到hashmap中
 */
void collect_results(JNIEnv* thread_env, jobject result_collector, std::atomic_uint& result_counter,
	const unsigned max_results, const std::vector<std::string>& search_case_vec)
{
	const jclass biconsumer_class = thread_env->GetObjectClass(result_collector);
	const jmethodID collector = thread_env->GetMethodID(biconsumer_class, "accept",
		"(Ljava/lang/Object;Ljava/lang/Object;)V");
	bool all_complete;
	const auto stop_func = [&]
	{
		return p_stop_signal[0] || result_counter.load() > max_results;
	};
	auto _collect_func = [&](const std::string& _key, char _matched_record_str[MAX_PATH_LENGTH],
		unsigned* matched_number)
	{
		auto expect = result_counter.load();
		while (!result_counter.compare_exchange_weak(expect, expect + 1))
		{
			expect = result_counter.load();
		}
		if (result_counter.load() > max_results)
		{
			is_results_number_exceed = true;
		}
		auto record_jstring = thread_env->NewStringUTF(_matched_record_str);
		auto key_jstring = thread_env->NewStringUTF(_key.c_str());
		thread_env->CallVoidMethod(result_collector, collector, key_jstring, record_jstring);
		thread_env->DeleteLocalRef(record_jstring);
		thread_env->DeleteLocalRef(key_jstring);
		++* matched_number;
	};
	const auto cl_queue = current_device.get_cl_queue();
#ifdef DEBUG_OUTPUT
	std::cout << "start to collect results. " << std::endl;
#endif
	do
	{
		//尝试退出
		all_complete = true;
		for (const auto& [key, cache_struct] : cache_map)
		{
			if (stop_func())
			{
				break;
			}
			if (!cache_struct->is_cache_valid)
			{
				continue;
			}
			if (!cache_struct->is_match_done.load())
			{
				//发现仍然有结果未计算完，设置退出标志为false，跳到下一个计算结果
				all_complete = false;
				continue;
			}
			if (int expected = 0; 
				!cache_struct->is_output_done.compare_exchange_weak(expected, 1))
			{
				continue;
			}
			unsigned matched_number = 0;
			cache_struct->dev_output->read_from_device();
#ifdef DEBUG_OUTPUT
			std::cout << "collecting key: " << key << std::endl;
#endif
			for (size_t i = 0; i < cache_struct->str_data.record_num.load() + 
				cache_struct->str_data.remain_blank_num.load(); ++i)
			{
				if (stop_func())
				{
					break;
				}
				//dev_cache[i]字符串匹配成功
				if (static_cast<bool>((*(cache_struct->dev_output))[i]))
				{
					char matched_record_str[MAX_PATH_LENGTH]{ 0 };
					auto&& cl_buffer = cache_struct->str_data.dev_cache_str->get_cl_buffer();
					cl_queue.enqueueReadBuffer(cl_buffer, true,
						i * MAX_PATH_LENGTH, MAX_PATH_LENGTH, matched_record_str);
					cl_queue.finish();
#ifdef DEBUG_OUTPUT
					std::cout << "collecting: " << matched_record_str << std::endl;
#endif
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
						else if (std::find(search_case_vec.begin(), search_case_vec.end(), "d") != search_case_vec.
							end())
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
			cache_struct->matched_number = matched_number;
			cache_struct->is_output_done = 2;
		}
	} while (!all_complete && !stop_func());
#ifdef DEBUG_OUTPUT
	std::cout << "stop collect." << std::endl;
#endif
	thread_env->DeleteLocalRef(biconsumer_class);
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
	return cache_map.find(key) != cache_map.end();
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
	auto&& cache_iter = cache_map.find(key);
	if (cache_iter != cache_map.end())
	{
		const auto& cache = cache_iter->second;
		const auto cl_queue = current_device.get_cl_queue();
		for (auto&& record : records)
		{
			if (exit_flag.load())
			{
				break;
			}
			if (const auto record_len = record.length(); record_len >= MAX_PATH_LENGTH)
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
						cl_queue.enqueueWriteBuffer(cache->str_data.dev_cache_str->get_cl_buffer(),
							true, index * MAX_PATH_LENGTH, MAX_PATH_LENGTH, record.c_str());
#ifdef DEBUG_OUTPUT
						std::cout << "successfully add record: " << record << "    key: " << key << std::endl;
#endif
						++cache->str_data.record_num;
						--cache->str_data.remain_blank_num;
						cache->str_data.record_hash.insert(hasher(record));
						current_device.finish_queue();
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
	}
}

/**
 * \brief 从key对应的record中删除一个记录
 * \param key key
 * \param records records
 */
void remove_records_from_cache(const std::string& key, std::vector<std::string>& records)
{
	auto&& cache_iter = cache_map.find(key);
	if (cache_iter != cache_map.end())
	{
		if (const auto& cache = cache_iter->second; cache->is_cache_valid)
		{
			char tmp[MAX_PATH_LENGTH]{ 0 };
			const auto cl_queue = current_device.get_cl_queue();
			for (size_t i = 0; i < cache->str_data.record_num; ++i)
			{
				if (exit_flag.load())
				{
					break;
				}
				cl_queue.enqueueReadBuffer(cache->str_data.dev_cache_str->get_cl_buffer(),
					true, i * MAX_PATH_LENGTH, MAX_PATH_LENGTH, tmp);
				current_device.finish_queue();
				if (auto&& record = std::find(records.begin(), records.end(), tmp);
					record != records.end())
				{
#ifdef DEBUG_OUTPUT
					printf("removing record: %s\n", record->c_str());
#endif
					//成功找到字符串
					const auto last_index = cache->str_data.record_num - 1;
					if (last_index == i)
					{
						continue;
					}
					//复制最后一个结果到当前空位
					auto&& cl_buffer = cache->str_data.dev_cache_str->get_cl_buffer();
					cl_queue.enqueueCopyBuffer(cl_buffer, cl_buffer,
						last_index * MAX_PATH_LENGTH, i * MAX_PATH_LENGTH, MAX_PATH_LENGTH);
					--cache->str_data.record_num;
					++cache->str_data.remain_blank_num;
					cache->str_data.record_hash.unsafe_erase(hasher(*record));
					current_device.finish_queue();
				}
			}
		}
	}
}

void clear_cache(const std::string& key)
{
	try
	{
		const auto cache = cache_map.at(key);
		cache->str_data.dev_cache_str->add_host_buffer(false);
		delete cache->str_data.dev_cache_str;
		delete cache->str_data.dev_total_number;
		delete cache->dev_output;
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
}

bool set_using_device(const unsigned device_num)
{
	if (auto&& devices = get_devices(); device_num < devices.size())
	{
		current_device = Device(select_device_with_id(device_num));
		p_stop_signal = Memory<char>(current_device, 1);
		p_stop_signal[0] = 0;
		p_stop_signal.write_to_device();
		return true;
	}
	return false;
}

std::vector<cl::Event*> start_kernel(const std::vector<std::string>& search_case,
	bool is_ignore_case,
	const char* search_text,
	const std::vector<std::string>& keywords,
	const std::vector<std::string>& keywords_lower_case,
	const bool* is_keyword_path)
{
	vector<cl::Event*> events_vec;
	const auto utf162gbk = get_p_utf162gbk();
	cl_int error = 0;
	auto p_utf162gbk = cl::Buffer(current_device.get_cl_context(), CL_MEM_READ_ONLY, sizeof(unsigned short) * 0x10000,
		nullptr, &error);
	if (error)
	{
		fprintf(stderr, "init utf82gbk failed. Error code: %d", error);
		return events_vec;
	}
	auto task_queue = current_device.get_cl_queue();
	task_queue.enqueueWriteBuffer(p_utf162gbk, true, 0, 0x10000 * sizeof(unsigned short), utf162gbk);
	task_queue.finish();

	const auto keywords_num = keywords.size();
	// 复制search case
	// 第一位为1表示有F，第二位为1表示有D，第三位为1表示有FULL，CASE由File-Engine主程序进行判断，传入参数is_ignore_case为false表示有CASE
	int search_case_num = 0;
	for (auto& each_case : search_case)
	{
		if (each_case == "full")
		{
			search_case_num |= 4;
		}
	}
	Memory<int> dev_search_case(current_device, 1);
	dev_search_case[0] = search_case_num;
	dev_search_case.write_to_device();

	// 复制search text
	Memory<char> dev_search_text(current_device, MAX_PATH_LENGTH);
	dev_search_text.reset();
	strcpy_s(dev_search_text.data(), MAX_PATH_LENGTH, search_text);
	dev_search_text.write_to_device();

	// 复制keywords
	Memory<char> dev_keywords(current_device, keywords_num * MAX_PATH_LENGTH);
	dev_keywords.reset();
	for (size_t i = 0; i < keywords_num; ++i)
	{
		strcpy_s(&(dev_keywords[i * MAX_PATH_LENGTH]), MAX_PATH_LENGTH, keywords[i].c_str());
	}
	dev_keywords.write_to_device();

	// 复制keywords_lower_case
	Memory<char> dev_keywords_lower_case(current_device, keywords_num * MAX_PATH_LENGTH);
	dev_keywords_lower_case.reset();
	for (size_t i = 0; i < keywords_num; ++i)
	{
		strcpy_s(&(dev_keywords_lower_case[i * MAX_PATH_LENGTH]), MAX_PATH_LENGTH, keywords_lower_case[i].c_str());
	}
	dev_keywords_lower_case.write_to_device();

	//复制keywords_length
	Memory<size_t> dev_keywords_length(current_device, 1);
	dev_keywords_length[0] = keywords_num;
	dev_keywords_length.write_to_device();

	// 复制is_keyword_path
	Memory<char> dev_is_keyword_path(current_device, keywords_num);
	for (unsigned i = 0; i < keywords_num; ++i)
	{
		dev_is_keyword_path[i] = static_cast<char>(is_keyword_path[i]);
	}
	dev_is_keyword_path.write_to_device();

	// 复制is_ignore_case
	Memory<bool> dev_is_ignore_case(current_device, 1);
	dev_is_ignore_case[0] = is_ignore_case;
	dev_is_ignore_case.write_to_device();

	void CL_CALLBACK match_callback(cl_event, cl_int, void* user_data);
	for (auto& [_, cache] : cache_map)
	{
		if (!cache->is_cache_valid)
		{
			continue;
		}
		if (p_stop_signal[0])
		{
			break;
		}
		const auto total = cache->str_data.record_num.load() + cache->str_data.remain_blank_num.load();
		cl_int ret = 0;
		// kernel function
		Kernel search_kernel(current_device, total, "check", &ret,
			cache->str_data.dev_cache_str->get_cl_buffer(),
			cache->str_data.dev_total_number->get_cl_buffer(),
			dev_search_case,
			dev_is_ignore_case,
			dev_search_text,
			dev_keywords,
			dev_keywords_lower_case,
			dev_keywords_length,
			dev_is_keyword_path,
			cache->dev_output->get_cl_buffer(),
			p_stop_signal,
			p_utf162gbk);
		if (ret != CL_SUCCESS)
		{
			fprintf(stderr, "OpenCL: init kernel failed. Error code: %d\n", ret);
			send_restart_event();
			break;
		}
		auto&& each_event = new cl::Event();
		events_vec.emplace_back(each_event);
		search_kernel.enqueue_run(&ret, each_event);
		each_event->setCallback(CL_COMPLETE, match_callback, cache);
		if (ret != CL_SUCCESS)
		{
			fprintf(stderr, "OpenCL: start kernel function failed. Error code: %d\n", ret);
			send_restart_event();
			break;
		}
	}
	current_device.finish_queue();
#ifdef DEBUG_OUTPUT
	std::cout << "all finished" << std::endl;
#endif
	return events_vec;
}

void CL_CALLBACK match_callback(cl_event, cl_int, void* user_data)
{
	auto* cache = static_cast<list_cache*>(user_data);
	cache->is_match_done = true;
}

template <typename I>
std::string n2hexstr(I w, size_t hex_len = sizeof(I) << 1)
{
	static const char* digits = "0123456789ABCDEF";
	std::string rc(hex_len, '0');
	for (size_t i = 0, j = (hex_len - 1) * 4; i < hex_len; ++i, j -= 4)
		rc[i] = digits[w >> j & 0x0f];
	return rc;
}

std::unordered_map<std::wstring, LONGLONG> query_pdh_val(PDH_STATUS& ret);
size_t get_device_memory_used()
{
	auto&& device_name = current_device.info.board_name;
	auto&& device_name_wstr = string2wstring(device_name);
	auto&& dxgi_device = gpu_name_adapter_map.find(device_name_wstr);
	if (dxgi_device == gpu_name_adapter_map.end())
	{
		return INFINITE;
	}
	auto&& adapter_luid = dxgi_device->second.AdapterLuid;
	auto&& luid_str = "0x" + n2hexstr(adapter_luid.HighPart) + "_" + "0x" + n2hexstr(adapter_luid.LowPart);
	auto&& luid_wstr = string2wstring(luid_str);
	PDH_STATUS status;
	auto&& memory_map = query_pdh_val(status);
	for (auto& [gpu_name, memory_used] : memory_map)
	{
		if (gpu_name.find(luid_wstr) != std::wstring::npos)
		{
			return memory_used;
		}
	}
	return INFINITE;
}

std::unordered_map<std::wstring, LONGLONG> query_pdh_val(PDH_STATUS& ret)
{
	PDH_HQUERY query;
	std::unordered_map<std::wstring, LONGLONG> memory_usage_map;
	ret = PdhOpenQuery(nullptr, NULL, &query);
	if (ret != ERROR_SUCCESS)
	{
		return memory_usage_map;
	}
	PDH_HCOUNTER counter;
	ret = PdhAddCounter(query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", NULL, &counter);
	if (ret != ERROR_SUCCESS)
	{
		return memory_usage_map;
	}
	ret = PdhCollectQueryData(query);
	if (ret != ERROR_SUCCESS)
	{
		return memory_usage_map;
	}
	DWORD bufferSize = 0;
	DWORD itemCount = 0;
	PdhGetRawCounterArray(counter, &bufferSize, &itemCount, nullptr);
	auto&& lpItemBuffer = reinterpret_cast<PPDH_RAW_COUNTER_ITEM_W>(new char[bufferSize]);
	ret = PdhGetRawCounterArray(counter, &bufferSize, &itemCount, lpItemBuffer);
	if (ret != ERROR_SUCCESS)
	{
		delete[] lpItemBuffer;
		return memory_usage_map;
	}
	for (DWORD i = 0; i < itemCount; ++i)
	{
		auto& [szName, RawValue] = lpItemBuffer[i];
		memory_usage_map.insert(std::make_pair(szName, RawValue.FirstValue));
	}
	delete[] lpItemBuffer;
	ret = PdhCloseQuery(query);
	return memory_usage_map;
}

void send_restart_event()
{
	fprintf(stderr, "Send RestartEvent.");
	JNIEnv* env = nullptr;
	JavaVMAttachArgs args{ JNI_VERSION_10, nullptr, nullptr };
	if (jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), &args) != JNI_OK)
	{
		fprintf(stderr, "get thread JNIEnv ptr failed");
		return;
	}
	auto&& gpu_class = env->FindClass("file.engine.dllInterface.gpu.GPUAccelerator");
	auto&& restart_method = env->GetMethodID(gpu_class, "sendRestartOnError0", "()V");
	env->CallStaticVoidMethod(gpu_class, restart_method);
	env->DeleteLocalRef(gpu_class);
	jvm->DetachCurrentThread();
}