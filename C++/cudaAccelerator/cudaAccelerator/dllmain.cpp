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
#include <Pdh.h>
#include <unordered_map>
#include <d3d.h>
#include <dxgi.h>
#include <shared_mutex>
#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "Shlwapi.lib")
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
int is_dir_or_file(const char* path);
inline bool is_file_exist(const char* path);
std::wstring string2wstring(const std::string& str);
void recreate_cache_on_removing_records(const std::string& key, const list_cache* cache,
                                        const concurrency::concurrent_unordered_set<size_t>& str_need_to_remove);
void create_and_insert_cache(const std::vector<std::string>& records_vec, size_t total_bytes, const std::string& key);

std::shared_mutex cache_lock;
concurrency::concurrent_unordered_map<std::wstring, DXGI_ADAPTER_DESC> gpu_name_adapter_map;
concurrency::concurrent_unordered_map<std::string, list_cache*> cache_map;
std::hash<std::string> hasher;
std::atomic_bool exit_flag = false;
static int current_using_device = 0;
std::atomic_bool is_results_number_exceed = false;
static JavaVM* jvm;
IDXGIFactory* p_dxgi_factory = nullptr;


/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    getDevices
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_getDevices
(JNIEnv* env, jobject)
{
    int device_count = 0;
    std::vector<std::string> device_vec;
    gpuErrchk(cudaGetDeviceCount(&device_count), false, "get device number failed.");
    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp prop;
        gpuErrchk(cudaGetDeviceProperties(&prop, i), false, "get device info failed.");
        device_vec.emplace_back(prop.name);
    }
    auto&& string_clazz = env->FindClass("java/lang/String");
    auto&& gpu_device_count = device_vec.size();
    auto&& object_arr = env->NewObjectArray(static_cast<jsize>(gpu_device_count), string_clazz, nullptr);
    for (UINT i = 0; i < gpu_device_count; ++i)
    {
        auto&& device_name = device_vec[i];
        env->SetObjectArrayElement(object_arr, i, env->NewStringUTF(device_name.c_str()));
    }
    env->DeleteLocalRef(string_clazz);
    return object_arr;
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
    {
        return true;
    }
    std::unique_lock ulck(cache_lock);
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
    std::unique_lock ulck(cache_lock);
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
        return;
    }
    set_jvm_ptr_in_kernel(jvm);
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    resetAllResultStatus
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_resetAllResultStatus
(JNIEnv*, jobject)
{
    std::unique_lock ulck(cache_lock);
    set_stop(false);
    // 初始化is_match_done is_output_done为false
    for (auto& [_, cache_ptr] : cache_map)
    {
        cache_ptr->is_match_done = false;
        cache_ptr->is_output_done = 0;
        cache_ptr->matched_number = 0;
    }
    is_results_number_exceed = false;
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
 * Signature: ([Ljava/lang/String;ZLjava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[ZIILjava/util/function/BiConsumer;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_match
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
    collect_threads_vec.reserve(result_collect_thread_num);
    for (int i = 0; i < result_collect_thread_num; ++i)
    {
        collect_threads_vec.emplace_back([&]
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
        });
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
        if (cache_val->is_output_done.load() != 2)
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
    if (is_results_number_exceed.load())
    {
        return true;
    }
    std::shared_lock lck(cache_lock);
    const auto _key = env->GetStringUTFChars(key_jstring, nullptr);
    auto iter = cache_map.find(_key);
    env->ReleaseStringUTFChars(key_jstring, _key);
    if (iter == cache_map.end())
    {
        return false;
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
    std::shared_lock lck(cache_lock);
    const auto key = env->GetStringUTFChars(key_jstring, nullptr);
    auto&& matched_number_iter = cache_map.find(key);
    env->ReleaseStringUTFChars(key_jstring, key);
    if (matched_number_iter == cache_map.end())
    {
        return 0;
    }
    auto&& matched_number = matched_number_iter->second;
    return static_cast<jint>(matched_number->matched_number);
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
    std::shared_lock lck(cache_lock);
    const auto key = env->GetStringUTFChars(key_jstring, nullptr);
    const bool ret = has_cache(key);
    env->ReleaseStringUTFChars(key_jstring, key);
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
            total_bytes += record_len + 1; // 每个字符串结尾 '\0'
        }
        env->ReleaseStringUTFChars(jstring_val, record);
        env->DeleteLocalRef(record_from_supplier);
    }
    const auto key = env->GetStringUTFChars(key_jstring, nullptr);
    create_and_insert_cache(records_vec, total_bytes, key);
    env->ReleaseStringUTFChars(key_jstring, key);
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    removeRecordsFromCache
 * Signature: (Ljava/lang/String;[Ljava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_removeRecordsFromCache
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
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    clearCache
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_clearCache
(JNIEnv* env, jobject, jstring key_jstring)
{
    std::unique_lock ulck(cache_lock);
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
    std::shared_lock lck(cache_lock);
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
    std::unique_lock ulck(cache_lock);
    clear_all_cache();
}

/*
 * Class:     file_engine_dllInterface_gpu_CudaAccelerator
 * Method:    getGPUMemUsage
 * Signature: ()I
 */
size_t get_device_memory_used();
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_gpu_CudaAccelerator_getGPUMemUsage
(JNIEnv*, jobject)
{
    size_t free;
    size_t total;
    gpuErrchk(cudaMemGetInfo(&free, &total), true, "Get memory info failed");
    auto&& total_mem = total;
    auto&& memory_used = get_device_memory_used();
    if (memory_used == INFINITE)
    {
        return 100;
    }
    return static_cast<jint>(memory_used * 100 / total_mem);
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
    cudaDeviceProp prop;
    gpuErrchk(cudaGetDeviceProperties(&prop, current_using_device), true, "Get device info failed.");
    auto&& device_name = prop.name;
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
        return is_stop() || result_counter.load() > max_results;
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
        ++*matched_number;
    };
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
                !cache_struct->is_output_done.compare_exchange_strong(expected, 1))
            {
                continue;
            }
            unsigned matched_number = 0;
            //复制结果数组到host，dev_output下标对应dev_cache中的下标，若dev_output[i]中的值为1，则对应dev_cache[i]字符串匹配成功
            const auto output_ptr = new char[cache_struct->output_bitmap_size]();
            //将dev_output拷贝到output_ptr
            gpuErrchk(
                cudaMemcpy(output_ptr, cache_struct->dev_output_bitmap, cache_struct->output_bitmap_size,
                    cudaMemcpyDeviceToHost),
                true, nullptr);
            for (size_t i = 0; i < cache_struct->output_bitmap_size; ++i)
            {
                if (stop_func())
                {
                    break;
                }
                //dev_cache[i]字符串匹配成功
                if (static_cast<bool>(output_ptr[i]))
                {
                    char matched_record_str[MAX_PATH_LENGTH]{0};
                    char* str_address = nullptr;
                    gpuErrchk(cudaMemcpy(&str_address, cache_struct->str_data.dev_str_addr + i, sizeof(size_t),
                                  cudaMemcpyDeviceToHost), false, nullptr);
                    //拷贝GPU中的字符串到host
                    const auto str_len = cache_struct->str_data.str_length_array[i];
                    gpuErrchk(cudaMemcpy(matched_record_str, str_address, str_len,
                                  cudaMemcpyDeviceToHost), false, "collect results failed");
                    // 判断文件和文件夹
                    if (search_case_vec.empty())
                    {
                        _collect_func(key, matched_record_str, &matched_number);
                    }
                    else
                    {
                        if (std::find(search_case_vec.begin(),
                                      search_case_vec.end(),
                                      "f") != search_case_vec.end())
                        {
                            if (is_dir_or_file(matched_record_str) == 1)
                            {
                                _collect_func(key, matched_record_str, &matched_number);
                            }
                        }
                        else if (std::find(search_case_vec.begin(),
                                           search_case_vec.end(),
                                           "d") != search_case_vec.end())
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
            delete[] output_ptr;
        }
    }
    while (!all_complete && !stop_func());
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
        if (const auto& cache = cache_iter->second; cache->is_cache_valid)
        {
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

                if (const bool is_repeat = is_record_repeat(record, cache); !is_repeat)
                {
                    if (cache->str_data.str_remain_blank_bytes >= record_len + 1)
                    {
                        const auto index = cache->str_data.record_num.load();
                        //计算上一个字符串的内存地址
                        char* last_str_addr = nullptr;
                        if (index != 0)
                        {
                            gpuErrchk(cudaMemcpy(&last_str_addr,
                                          cache->str_data.dev_str_addr + index - 1,
                                          sizeof(size_t), cudaMemcpyDeviceToHost),
                                      true, nullptr);
                        }
                        char* target_address;
                        if (last_str_addr != nullptr)
                        {
                            //字符串尾的下一位即为新字符串起始地址
                            target_address = last_str_addr + cache->str_data.str_length_array[index - 1] + 1;
                        }
                        else
                        {
                            target_address = cache->str_data.dev_strs;
                        }
                        //记录到下一空位内存地址target_address
                        gpuErrchk(cudaMemsetAsync(target_address,
                                      0,
                                      record_len + 1,
                                      stream), true,
                                  get_cache_info(key, cache).c_str());
                        gpuErrchk(cudaMemcpyAsync(target_address,
                                      record.c_str(),
                                      record_len,
                                      cudaMemcpyHostToDevice, stream), true,
                                  get_cache_info(key, cache).c_str());
                        //记录内存地址到dev_str_addr
                        if (cache->str_data.str_addr_capacity <= index)
                        {
#ifdef DEBUG_OUTPUT
							std::cout << "start to enlarge dev_str_addr and str_lenght_array" << std::endl;
#endif
                            const auto new_str_addr_capacity = cache->str_data.str_addr_capacity + 10;
                            const auto new_str_addr_alloc_size = new_str_addr_capacity * sizeof(size_t);
                            size_t* new_dev_str_addr;
                            gpuErrchk(cudaMallocAsync(&new_dev_str_addr, new_str_addr_alloc_size, stream), true,
                                      nullptr);
                            gpuErrchk(cudaMemsetAsync(new_dev_str_addr, 0, new_str_addr_alloc_size, stream), true,
                                      nullptr);
                            gpuErrchk(cudaMemcpyAsync(new_dev_str_addr,
                                          cache->str_data.dev_str_addr,
                                          cache->str_data.str_addr_capacity * sizeof size_t,
                                          cudaMemcpyDeviceToDevice,
                                          stream), true, nullptr);
                            gpuErrchk(cudaFreeAsync(cache->str_data.dev_str_addr, stream), true, nullptr);
                            cache->str_data.dev_str_addr = new_dev_str_addr;
#ifdef DEBUG_OUTPUT
							std::cout << "copy dev_str_addr complete." << std::endl;
#endif
                            auto* new_str_length_array = new size_t[new_str_addr_capacity]();
                            for (size_t i = 0; i < cache->str_data.str_addr_capacity; ++i)
                            {
                                new_str_length_array[i] = cache->str_data.str_length_array[i];
                            }
                            delete[] cache->str_data.str_length_array;
                            cache->str_data.str_length_array = new_str_length_array;

                            cache->str_data.str_addr_capacity = new_str_addr_capacity;
#ifdef DEBUG_OUTPUT
							std::cout << "enlarge complete." << std::endl;
#endif
                        }
                        gpuErrchk(cudaMemcpyAsync(cache->str_data.dev_str_addr + index,
                                      &target_address,
                                      sizeof(size_t),
                                      cudaMemcpyHostToDevice,
                                      stream),
                                  true, nullptr);
                        //记录字符串长度
                        cache->str_data.str_length_array[index] = record.length();
#ifdef DEBUG_OUTPUT
						std::cout << "target address: " << std::hex << "0x" << reinterpret_cast<size_t>(target_address) << std::endl;
						std::cout << "successfully add record: " << record << "  key: " << key << std::endl;
#endif
                        ++cache->str_data.record_num;
                        cache->str_data.str_remain_blank_bytes -= record_len + 1;
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
            gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
            gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
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
            concurrency::concurrent_unordered_set<size_t> str_need_to_remove;
            const auto tmp_records = new char[MAX_PATH_LENGTH * cache->str_data.record_num.load()]();
            cudaStream_t stream;
            gpuErrchk(cudaStreamCreate(&stream), true, nullptr);
            for (size_t i = 0; i < cache->str_data.record_num.load(); ++i)
            {
                if (exit_flag.load())
                {
                    break;
                }
                char* str_address;
                char* tmp = tmp_records + MAX_PATH_LENGTH * i;
                gpuErrchk(cudaMemcpyAsync(&str_address,
                              cache->str_data.dev_str_addr + i,
                              sizeof(size_t),
                              cudaMemcpyDeviceToHost,
                              stream),
                          true,
                          get_cache_info(key, cache).c_str());
                //拷贝GPU中的字符串到tmp
                gpuErrchk(cudaMemcpyAsync(tmp,
                              str_address,
                              cache->str_data.str_length_array[i],
                              cudaMemcpyDeviceToHost,
                              stream),
                          true,
                          get_cache_info(key, cache).c_str());
                if (auto&& record = std::find(records.begin(), records.end(), tmp);
                    record == records.end())
                {
                    continue;
                }
#ifdef DEBUG_OUTPUT
				std::cout << "removing: " << tmp << "  index: " << i << std::endl;
#endif
                str_need_to_remove.insert(i);
            }
            gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
            gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
            delete[] tmp_records;
            recreate_cache_on_removing_records(key, cache, str_need_to_remove);
        }
    }
}

void create_and_insert_cache(const std::vector<std::string>& records_vec, const size_t total_bytes,
                             const std::string& key)
{
    const auto record_count = records_vec.size();
    auto cache = new list_cache;
    cache->str_data.record_num = record_count;

    const auto alloc_bytes = total_bytes + CACHE_REMAIN_BLANK_SIZE_IN_BYTES;
    gpuErrchk(cudaMalloc(&cache->str_data.dev_strs, alloc_bytes), true, nullptr);
    gpuErrchk(cudaMemset(cache->str_data.dev_strs, 0, alloc_bytes), true, nullptr);
    cache->str_data.str_total_bytes = alloc_bytes;
    cache->str_data.str_remain_blank_bytes = CACHE_REMAIN_BLANK_SIZE_IN_BYTES;

    const auto str_addr_capacity = record_count + CACHE_REMAIN_BLANK_SIZE_IN_BYTES / MAX_PATH_LENGTH;
    const auto str_addr_alloc_size = str_addr_capacity * sizeof(size_t);
    gpuErrchk(cudaMalloc(&cache->str_data.dev_str_addr, str_addr_alloc_size), true, nullptr);
    gpuErrchk(cudaMemset(cache->str_data.dev_str_addr, 0, str_addr_alloc_size), true, nullptr);
    cache->str_data.str_addr_capacity = str_addr_capacity;

    cache->str_data.str_length_array = new size_t[str_addr_capacity]();

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
        cache->str_data.str_length_array[i] = record_length;
        target_addr += record_length + 1;
        ++save_str_addr_ptr;
        ++i;
        cache->str_data.record_hash.insert(hasher(record));
    }
    gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
    gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
    cache_map.insert(std::make_pair(key, cache));
}

/**
 * \brief 重新创建缓存，但是忽略需要删除的数据
 * \param key cache key
 * \param cache 缓存
 * \param str_need_to_remove 需要删除数据的下标
 */
void recreate_cache_on_removing_records(const std::string& key, const list_cache* cache,
                                        const concurrency::concurrent_unordered_set<size_t>& str_need_to_remove)
{
    std::vector<std::string> records;
    size_t total_bytes = 0;
    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream), true, nullptr);
    const auto tmp_records = new char[MAX_PATH_LENGTH * cache->str_data.record_num.load()]();
    for (size_t i = 0; i < cache->str_data.record_num.load(); ++i)
    {
        if (auto&& find_val = std::find(str_need_to_remove.begin(), str_need_to_remove.end(), i);
            find_val != str_need_to_remove.end())
        {
#ifdef DEBUG_OUTPUT
			char* str_address_in_device_debug;
			gpuErrchk(cudaMemcpy(&str_address_in_device_debug, cache->str_data.dev_str_addr + i,
				sizeof(size_t), cudaMemcpyDeviceToHost), true, nullptr);
			gpuErrchk(cudaMemcpy(tmp, str_address_in_device_debug,
				cache->str_data.str_length_array[i],
				cudaMemcpyDeviceToHost),
				true,
				nullptr);
			std::cout << "Found removing record: " << tmp << "  index: " << i << "  Skipping." << std::endl;
#endif
            continue;
        }
        char* str_address_in_device;
        char* str_address = tmp_records + i * MAX_PATH_LENGTH;
        gpuErrchk(cudaMemcpyAsync(&str_address_in_device, cache->str_data.dev_str_addr + i,
                      sizeof(size_t), cudaMemcpyDeviceToHost, stream), true, nullptr);
        gpuErrchk(cudaMemcpyAsync(str_address, str_address_in_device, cache->str_data.str_length_array[i],
                      cudaMemcpyDeviceToHost, stream), true, nullptr);
        records.emplace_back(str_address);
        total_bytes += strlen(str_address) + 1;
    }
    gpuErrchk(cudaStreamSynchronize(stream), true, nullptr);
    gpuErrchk(cudaStreamDestroy(stream), true, nullptr);
    gpuErrchk(cudaFree(cache->str_data.dev_strs), false, nullptr);
    gpuErrchk(cudaFree(cache->str_data.dev_str_addr), false, nullptr);
    delete[] cache->str_data.str_length_array;
    delete cache;
    delete[] tmp_records;
    cache_map.unsafe_erase(key);
    create_and_insert_cache(records, total_bytes, key);
}

void clear_cache(const std::string& key)
{
    try
    {
        const auto cache = cache_map.at(key);
        cache->is_cache_valid = false;
        gpuErrchk(cudaFree(cache->str_data.dev_strs), false, get_cache_info(key, cache).c_str());
        gpuErrchk(cudaFree(cache->str_data.dev_str_addr), false, get_cache_info(key, cache).c_str());
        delete[] cache->str_data.str_length_array;
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
    free_cuda_search_memory();
    free_stop_signal();
}
