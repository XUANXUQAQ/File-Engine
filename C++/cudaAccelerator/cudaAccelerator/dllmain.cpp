// dllmain.cpp : 定义 DLL 应用程序的入口点。
#include "pch.h"
#include <concurrent_unordered_map.h>
#include "kernels.cuh"
#include "file_engine_dllInterface_gpu_CudaAccelerator.h"
#include "cache.h"
#include "constans.h"
#include "cuda_copy_vector_util.h"
#include "str_convert.cuh"
#include <thread>
#include <Shlwapi.h>
#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "Shlwapi.lib")

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
int is_dir_or_file(const char* path);
inline bool is_file_exist(const char* path);
std::wstring string2wstring(const std::string& str);
void handle_memory_fragmentation(const list_cache* cache);

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
std::atomic_bool exit_flag = false;
static int current_using_device = 0;
JavaVM* jvm;

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    getDevices
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_getDevices
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    setDevice
 * Signature: (I)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_setDevice
(JNIEnv*, jobject, jint device_number_jint)
{
    if (device_number_jint == current_using_device)
        return true;
    release_all();
    if (set_using_device(device_number_jint))
    {
        current_using_device = device_number_jint;
        init_stop_signal();
        init_cuda_search_memory();
        init_str_convert();
        return true;
    }
    return false;
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_release
(JNIEnv*, jobject)
{
    exit_flag = true;
    release_all();
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    initialize
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_initialize
(JNIEnv* env, jobject)
{
    init_stop_signal();
    init_cuda_search_memory();
    init_str_convert();
    //默认使用第一个设备,current_using_device=0
    set_using_device(current_using_device);
    if (env->GetJavaVM(&jvm) != JNI_OK)
    {
        env->ThrowNew(env->FindClass("java/lang/Exception"), "get JavaVM ptr failed.");
    }
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    resetAllResultStatus
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_resetAllResultStatus
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    stopCollectResults
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_stopCollectResults
(JNIEnv*, jobject)
{
    set_stop(true);
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    match
 * Signature: ([Ljava/lang/String;ZLjava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[ZILjava/util/function/BiConsumer;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_match
(JNIEnv* env, jobject, jobjectArray search_case, jboolean is_ignore_case, jstring search_text,
 jobjectArray keywords, jobjectArray keywords_lower, jbooleanArray is_keyword_path, jint max_results,
 jobject result_collector)
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
    bool is_keyword_path_ptr[MAX_KEYWORDS_NUMBER]{false};
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
    collect_threads_vec.reserve(COLLECT_RESULTS_THREADS);
    for (int i = 0; i < COLLECT_RESULTS_THREADS; ++i)
    {
        collect_threads_vec.emplace_back(std::thread([&]
        {
            JNIEnv* thread_env = nullptr;
            JavaVMAttachArgs args{JNI_VERSION_10, nullptr, nullptr};
            if (jvm->AttachCurrentThread(reinterpret_cast<void**>(&thread_env), &args) != JNI_OK)
            {
                fprintf(stderr, "get thread JNIEnv ptr failed");
                return;
            }
            collect_results(thread_env, result_collector, result_counter, max_results, search_case_vec);
            jvm->DetachCurrentThread();
        }));
    }
    //GPU并行计算
    start_kernel(cache_map, search_case_vec, is_ignore_case, search_text_chars,
                 keywords_vec, keywords_lower_vec, is_keyword_path_ptr);
    collect_results(env, result_collector, result_counter, max_results, search_case_vec);
    for (auto&& each_thread : collect_threads_vec)
    {
        if (each_thread.joinable())
        {
            each_thread.join();
        }
    }
    for (auto& [_, cache_val] : cache_map)
    {
        if (cache_val->is_output_done != 2)
     	{
     		cache_val->is_output_done = 2;
     	}
    }
    env->ReleaseStringUTFChars(search_text, search_text_chars);
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    isMatchDone
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_isMatchDone
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    matchedNumber
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_matchedNumber
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    hasCache
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_hasCache
(JNIEnv*, jobject)
{
    return !cache_map.empty();
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    isCacheExist
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_isCacheExist
(JNIEnv* env, jobject, jstring key_jstring)
{
    const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
    const std::string key(_key);
    env->ReleaseStringUTFChars(key_jstring, _key);
    const bool ret = has_cache(key);
    return ret;
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    initCache
 * Signature: (Ljava/lang/String;Ljava/util/function/Supplier;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_initCache
(JNIEnv* env, jobject, jstring key_jstring, jobject record_supplier)
{
    const jclass supplier_class = env->GetObjectClass(record_supplier);
    const jmethodID get_function = env->GetMethodID(supplier_class, "get", "()Ljava/lang/Object;");
    std::vector<std::string> records_vec;
    unsigned record_count = 0;
    size_t total_bytes = 0;
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
            total_bytes += record_len;
            ++total_bytes; // 每个字符串结尾 '\0'
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

    const auto alloc_bytes = total_bytes + MAX_RECORD_ADD_COUNT * MAX_PATH_LENGTH;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cache->str_data.dev_strs), alloc_bytes), true,
              get_cache_info(key, cache).c_str());
    gpuErrchk(cudaMemset(cache->str_data.dev_strs, 0, alloc_bytes), true, nullptr);

    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cache->str_data.dev_str_addr), total_results_size * sizeof(size_t)),
              true, nullptr);
    gpuErrchk(cudaMemset(cache->str_data.dev_str_addr, 0, total_results_size * sizeof(size_t)), true, nullptr);

    cache->str_data.str_length = new size_t[total_results_size];

    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&cache->dev_output), total_results_size), true,
              get_cache_info(key, cache).c_str());
    cache->is_cache_valid = true;
    cache->is_match_done = false;
    cache->is_output_done = 0;

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream), true, nullptr);
    auto target_addr = cache->str_data.dev_strs;
    auto save_str_addr_ptr = cache->str_data.dev_str_addr;
    unsigned i = 0;
    for (const std::string& record : records_vec)
    {
        const auto record_length = record.length();
        gpuErrchk(cudaMemcpyAsync(target_addr, record.c_str(), record_length, cudaMemcpyHostToDevice, stream), true,
                  nullptr);
        const auto str_address = reinterpret_cast<size_t>(target_addr);
        //保存字符串在显存上的地址
        gpuErrchk(cudaMemcpyAsync(save_str_addr_ptr, &str_address, sizeof(size_t), cudaMemcpyHostToDevice, stream),
                  true, nullptr);
        cache->str_data.str_length[i] = record_length;
        target_addr += record_length;
        ++target_addr;
        ++save_str_addr_ptr;
        ++i;
        cache->str_data.record_hash.insert(hasher(record));
    }
    gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
    gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
    cache_map.insert(std::make_pair(key, cache));
    env->ReleaseStringUTFChars(key_jstring, _key);
    env->DeleteLocalRef(supplier_class);
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    addRecordsToCache
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_addRecordsToCache
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    removeRecordsFromCache
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_removeRecordsFromCache
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    clearCache
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_clearCache
(JNIEnv* env, jobject, jstring key_jstring)
{
    const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
    clear_cache(_key);
    env->ReleaseStringUTFChars(key_jstring, _key);
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    isCudaAvailable
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_isCudaAvailable
(JNIEnv*, jobject)
{
    int device_num = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&device_num);
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    isCacheValid
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_isCacheValid
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    clearAllCache
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_clearAllCache
(JNIEnv*, jobject)
{
    clear_all_cache();
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    getGPUMemUsage
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_getGPUMemUsage
(JNIEnv*, jobject)
{
    size_t avail = 0;
    size_t total = 1;
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
    const jclass biconsumer_class = thread_env->GetObjectClass(result_collector);
    const jmethodID collector = thread_env->GetMethodID(biconsumer_class, "accept",
                                                        "(Ljava/lang/Object;Ljava/lang/Object;)V");
    bool all_complete;
    const auto stop_func = [&]
    {
        return is_stop() || result_counter.load() >= max_results;
    };
    auto _collect_func = [&](const std::string& _key, char _matched_record_str[MAX_PATH_LENGTH],
                             unsigned* matched_number)
    {
        ++result_counter;
        auto record_jstring = thread_env->NewStringUTF(_matched_record_str);
        auto key_jstring = thread_env->NewStringUTF(_key.c_str());
        thread_env->CallVoidMethod(result_collector, collector, key_jstring, record_jstring);
        thread_env->DeleteLocalRef(record_jstring);
        thread_env->DeleteLocalRef(key_jstring);
        ++*matched_number;
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
            if (!val->is_cache_valid)
            {
                continue;
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
                    char matched_record_str[MAX_PATH_LENGTH]{0};
                    char* str_address;
                    gpuErrchk(
                        cudaMemcpy(&str_address, val->str_data.dev_str_addr + i, sizeof(size_t), cudaMemcpyDeviceToHost
                        ), false, nullptr);
                    //拷贝GPU中的字符串到host
                    gpuErrchk(cudaMemcpy(matched_record_str, str_address, val->str_data.str_length[i],
                                  cudaMemcpyDeviceToHost), false, "collect results failed");
                    // 判断文件和文件夹
                    if (search_case_vec.empty())
                    {
                        if (is_file_exist(matched_record_str))
                        {
                            _collect_func(key, matched_record_str, &matched_number);
                        }
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
                            if (is_file_exist(matched_record_str))
                            {
                                _collect_func(key, matched_record_str, &matched_number);
                            }
                        }
                    }
                }
            }
            matched_result_number_map.insert(std::make_pair(key, matched_number));
            val->is_output_done = 2;
            delete[] output_ptr;
        }
    }
    while (!all_complete && !is_stop() && result_counter.load() < max_results);
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
    const DWORD dwAttrib = GetFileAttributes(w_path.c_str());
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

inline bool is_file_exist(const char* path)
{
    struct _stat64i32 buffer;
    return _wstat(string2wstring(path).c_str(), &buffer) == 0;
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
            if (exit_flag.load())
            {
                break;
            }
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
                        //计算上一个字符串的内存地址
                        char* last_str_addr;
                        gpuErrchk(cudaMemcpy(&last_str_addr, cache->str_data.dev_str_addr + index - 1,
                                      sizeof(size_t), cudaMemcpyDeviceToHost), true, nullptr);
#ifdef DEBUG_OUTPUT
						std::cout << "last str address: " << std::hex << "0x" << reinterpret_cast<size_t>(last_str_addr);
						std::cout << "  last str length: 0x" << cache->str_data.str_length[index - 1] << std::endl;
#endif
                        //字符串尾的下一位即为新字符串起始地址
                        char* target_address = last_str_addr + cache->str_data.str_length[index - 1] + 1;
                        //记录到下一空位内存地址target_address
                        gpuErrchk(cudaMemsetAsync(target_address, 0, record_len + 1, stream), true,
                                  get_cache_info(key, cache).c_str());
                        gpuErrchk(cudaMemcpyAsync(target_address, record.c_str(), record_len,
                                      cudaMemcpyHostToDevice, stream), true, get_cache_info(key, cache).c_str());
                        //记录内存地址到dev_str_addr
                        gpuErrchk(cudaMemcpyAsync(cache->str_data.dev_str_addr + index, &target_address,
                                      sizeof(size_t), cudaMemcpyHostToDevice, stream), true, nullptr);
                        //记录字符串长度
                        cache->str_data.str_length[index] = record.length();
#ifdef DEBUG_OUTPUT
						std::cout << "target address: " << std::hex << "0x" << reinterpret_cast<size_t>(target_address) << std::endl;
						std::cout << "successfully add record: " << record << "  key: " << key << std::endl;
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
void remove_records_from_cache(const std::string& key, std::vector<std::string>& records)
{
    lock_clear_cache(remove_record_thread_count);
    if (cache_map.find(key) != cache_map.end())
    {
        const auto& cache = cache_map.at(key);
        std::lock_guard lock_guard(cache->str_data.lock);
        bool has_memory_fragments = false;
        cudaStream_t stream;
        gpuErrchk(cudaStreamCreate(&stream), true, nullptr);
        if (cache->is_cache_valid)
        {
            char tmp[MAX_PATH_LENGTH]{0};
            for (size_t i = 0; i < cache->str_data.record_num; ++i)
            {
                if (exit_flag.load())
                {
                    break;
                }
                char* str_address;
                gpuErrchk(cudaMemcpy(&str_address, cache->str_data.dev_str_addr + i,
                              sizeof(size_t), cudaMemcpyDeviceToHost), true, nullptr);
                //拷贝GPU中的字符串到tmp
                gpuErrchk(cudaMemcpy(tmp, str_address,
                              cache->str_data.str_length[i], cudaMemcpyDeviceToHost), true,
                          get_cache_info(key, cache).c_str());
                if (auto&& record = std::find(records.begin(), records.end(), tmp);
                    record != records.end())
                {
#ifdef DEBUG_OUTPUT
					printf("removing record: %s\n", tmp);
#endif
                    //成功找到字符串
                    //判断是否已经是最后一个字符串
                    if (const auto last_index = cache->str_data.record_num - 1; last_index != i)
                    {
                        const char* last_str_address;
                        gpuErrchk(cudaMemcpy(&last_str_address, cache->str_data.dev_str_addr + last_index,
                                      sizeof(size_t), cudaMemcpyDeviceToHost), true, nullptr);
                        //判断字符串长度是否合适
                        if (const auto str_to_copy_len = cache->str_data.str_length[last_index];
                            cache->str_data.str_length[i] >= str_to_copy_len)
                        {
#ifdef DEBUG_OUTPUT
							std::cout << "str to remove address: " << std::hex << "0x" << reinterpret_cast<size_t>(str_address) << std::endl;
							std::cout << "str to copy address: " << std::hex << "0x" << reinterpret_cast<size_t>(last_str_address) << std::endl;
#endif
                            gpuErrchk(cudaMemsetAsync(str_address, 0, cache->str_data.str_length[i], stream), true,
                                      nullptr);
                            //复制最后一个结果到当前空位
                            gpuErrchk(cudaMemcpyAsync(str_address, last_str_address,
                                          str_to_copy_len, cudaMemcpyDeviceToDevice, stream),
                                      true, get_cache_info(key, cache).c_str());
                            //记录字符串长度
                            cache->str_data.str_length[i] = cache->str_data.str_length[last_index];
                        }
                        else
                        {
                            // 不删除字符串，只更新字符串地址
#ifdef DEBUG_OUTPUT
							std::cout << "str length not enough, copy address " << std::hex <<
								"0x" << reinterpret_cast<size_t>(str_address) << "  address removed: "
								"0x" << reinterpret_cast<size_t>(last_str_address)
								<< std::endl;
#endif
                            has_memory_fragments = true;
                            gpuErrchk(
                                cudaMemcpyAsync(cache->str_data.dev_str_addr + i, cache->str_data.dev_str_addr +
                                    last_index,
                                    sizeof(size_t), cudaMemcpyDeviceToDevice, stream), true, nullptr);
                            //记录字符串长度
                            cache->str_data.str_length[i] = cache->str_data.str_length[last_index];
                        }
                    }
                    --cache->str_data.record_num;
                    ++cache->str_data.remain_blank_num;
                    cache->str_data.record_hash.unsafe_erase(hasher(*record));
                }
            }
        }
        gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
        gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
        // 处理内存碎片
        if (has_memory_fragments)
            handle_memory_fragmentation(cache);
    }
    //释放clear_cache锁
    free_clear_cache(remove_record_thread_count);
}

/**
 * \brief 处理cache中缓存的内存碎片
 * \param cache 缓存
 */
void handle_memory_fragmentation(const list_cache* cache)
{
    std::vector<std::string> records;
#ifdef DEBUG_OUTPUT
	char* last_addr = nullptr;
#endif
    for (size_t i = 0; i < cache->str_data.record_num; ++i)
    {
        char* str_address_in_device;
        gpuErrchk(cudaMemcpy(&str_address_in_device, cache->str_data.dev_str_addr + i,
                      sizeof(size_t), cudaMemcpyDeviceToHost), true, nullptr);
#ifdef DEBUG_OUTPUT
		if (last_addr != nullptr)
		{
			if (last_addr > str_address_in_device)
			{
				std::cout << "address: " << std::hex << "0x" << reinterpret_cast<size_t>(last_addr) <<
					" is higher than " << "0x" << reinterpret_cast<size_t>(str_address_in_device) << std::endl;
			}
		}
		last_addr = str_address_in_device;
#endif
        char tmp[MAX_PATH_LENGTH]{0};
        gpuErrchk(cudaMemcpy(tmp, str_address_in_device, cache->str_data.str_length[i], cudaMemcpyDeviceToHost), true,
                  nullptr);
        records.emplace_back(tmp);
    }
    auto target_addr = cache->str_data.dev_strs;
    auto save_str_addr_ptr = cache->str_data.dev_str_addr;
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream), true, nullptr);
    unsigned i = 0;
    for (const std::string& record : records)
    {
        const auto record_length = record.length();
        gpuErrchk(cudaMemsetAsync(target_addr, 0, record_length + 1, stream), true, nullptr);
        gpuErrchk(cudaMemcpyAsync(target_addr, record.c_str(), record_length, cudaMemcpyHostToDevice, stream), true,
                  nullptr);
        const auto str_address = reinterpret_cast<size_t>(target_addr);
        //保存字符串在显存上的地址
        gpuErrchk(cudaMemcpyAsync(save_str_addr_ptr, &str_address, sizeof(size_t), cudaMemcpyHostToDevice, stream),
                  true, nullptr);
        //保存字符串长度
        cache->str_data.str_length[i] = record_length;
        target_addr += record_length;
        ++target_addr;
        ++save_str_addr_ptr;
        ++i;
    }
    gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
    gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
#ifdef DEBUG_OUTPUT
	{
		std::cout << "address check" << std::endl;
		char* p_last_addr = nullptr;
		for (size_t j = 0; j < cache->str_data.record_num; ++j)
		{
			char* str_address_in_device;
			gpuErrchk(cudaMemcpy(&str_address_in_device, cache->str_data.dev_str_addr + j,
				sizeof(size_t), cudaMemcpyDeviceToHost), true, nullptr);
			if (p_last_addr != nullptr)
			{
				if (p_last_addr > str_address_in_device)
				{
					std::cout << "address: " << std::hex << "0x" << reinterpret_cast<size_t>(p_last_addr) <<
						" is higher than " << "0x" << reinterpret_cast<size_t>(str_address_in_device) << std::endl;
					std::cout << "check failed." << std::endl;
					std::quick_exit(1);
				}
			}
			p_last_addr = str_address_in_device;
		}
}
#endif
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
            gpuErrchk(cudaFree(cache->str_data.dev_strs), false, get_cache_info(key, cache).c_str());
            gpuErrchk(cudaFree(cache->dev_output), false, get_cache_info(key, cache).c_str());
            gpuErrchk(cudaFree(cache->str_data.dev_str_addr), false, nullptr);
            delete cache->str_data.str_length;
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
