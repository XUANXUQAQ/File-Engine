// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "file_engine_dllInterface_PathMatcher.h"
#include <string>
#include <vector>
#include <concurrent_unordered_map.h>
#include "sqlite3.h"
#include <mutex>
#include "path_util.h"
#include "str_convert.h"
#pragma comment(lib, "sqlite3")

concurrency::concurrent_unordered_map<std::string, sqlite3*> connection_map;
std::mutex lock;

BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD ul_reason_for_call,
                      LPVOID lpReserved
)
{
    init_str_convert();
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

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

int callback(void* data, int col_count, char** res_value, char** res_col_name)
{
    search_task* task = static_cast<search_task*>(data);
    if (task->counter > task->max_result)
    {
        return 0;
    }
    for (int i = 0; i < col_count; i++)
    {
        if (strcmp(res_col_name[i], "PATH") == 0)
        {
            if (const auto res = res_value[i]; match_func(res, task))
            {
                ++task->counter;
                task->result_vec.emplace_back(res);
            }
        }
    }
    return 0;
}

/*
 * Class:     file_engine_dllInterface_PathMatcher
 * Method:    match
 * Signature: (Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;ZLjava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[ZI)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_file_engine_dllInterface_PathMatcher_match
(JNIEnv* env, jobject, jstring sql, jstring db_path, jobjectArray search_case, jboolean is_ignore_case,
 jstring search_text, jobjectArray keywords, jobjectArray keywords_lower, jbooleanArray is_keyword_path,
 jint max_results)
{
    std::vector<std::string> search_case_vec;
    if (search_case != nullptr)
    {
        generate_search_case(env, search_case_vec, search_case);
    }
    int search_case_num = 0;
    for (auto& each_case : search_case_vec)
    {
        if (each_case == "f")
        {
            search_case_num |= 1;
        }
        if (each_case == "d")
        {
            search_case_num |= 2;
        }
        if (each_case == "full")
        {
            search_case_num |= 4;
        }
    }
    std::vector<std::string> keywords_vec;
    std::vector<std::string> keywords_lower_vec;
    const auto keywords_length = env->GetArrayLength(keywords);
    if (keywords_length > MAX_KEYWORDS_NUMBER)
    {
        return nullptr;
    }
    bool is_keyword_path_ptr[MAX_KEYWORDS_NUMBER]{false};
    const auto is_keyword_path_ptr_bool_array = env->GetBooleanArrayElements(is_keyword_path, nullptr);
    for (jsize i = 0; i < keywords_length; ++i)
    {
        auto tmp_keywords_str = reinterpret_cast<jstring>(env->GetObjectArrayElement(keywords, i));
        auto keywords_chars = env->GetStringUTFChars(tmp_keywords_str, nullptr);
        keywords_vec.emplace_back(keywords_chars);
        env->ReleaseStringUTFChars(tmp_keywords_str, keywords_chars);
        env->DeleteLocalRef(tmp_keywords_str);

        tmp_keywords_str = reinterpret_cast<jstring>(env->GetObjectArrayElement(keywords_lower, i));
        keywords_chars = env->GetStringUTFChars(tmp_keywords_str, nullptr);
        keywords_lower_vec.emplace_back(keywords_chars);
        env->ReleaseStringUTFChars(tmp_keywords_str, keywords_chars);
        env->DeleteLocalRef(tmp_keywords_str);
        is_keyword_path_ptr[i] = is_keyword_path_ptr_bool_array[i];
    }
    env->ReleaseBooleanArrayElements(is_keyword_path, is_keyword_path_ptr_bool_array, JNI_ABORT);
    const auto search_text_chars = env->GetStringUTFChars(search_text, nullptr);
    sqlite3* db = nullptr;
    try
    {
        const auto db_path_str = env->GetStringUTFChars(db_path, nullptr);
        db = connection_map.at(db_path_str);
        env->ReleaseStringUTFChars(db_path, db_path_str);
    }
    catch (std::out_of_range& e)
    {
        fprintf(stderr, "Error: %s\n", e.what());
    }
    if (db == nullptr)
    {
        return nullptr;
    }
    const auto sql_str = env->GetStringUTFChars(sql, nullptr);
    search_task task;
    task.search_case_num = search_case_num;
    task.is_ignore_case = is_ignore_case;
    task.search_text = search_text_chars;
    task.keywords = &keywords_vec;
    task.keywords_lower_case = &keywords_lower_vec;
    task.is_keyword_path = is_keyword_path_ptr;
    task.max_result = max_results;
    char* error_str;
    const auto rc = sqlite3_exec(db, sql_str, callback, &task, &error_str);
    if (rc != SQLITE_OK)
    {
        fprintf(stderr, "Query sql failed, sql: %s\n", sql_str);
        sqlite3_free(error_str);
        return nullptr;
    }
    const auto string_class = env->FindClass("java/lang/String");
    const auto object_arr = env->NewObjectArray(static_cast<jsize>(task.result_vec.size()), string_class, nullptr);
    jsize count = 0;
    for (const auto& each_result : task.result_vec)
    {
        env->SetObjectArrayElement(object_arr, count, env->NewStringUTF(each_result.c_str()));
        ++count;
    }
    env->DeleteLocalRef(string_class);
    return object_arr;
}

/*
 * Class:     file_engine_dllInterface_PathMatcher
 * Method:    openConnection
 * Signature: (Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_PathMatcher_openConnection
(JNIEnv* env, jobject, jstring db_path)
{
    const auto db_path_str = env->GetStringUTFChars(db_path, nullptr);
    if (connection_map.find(db_path_str) != connection_map.end())
    {
        return;
    }
    lock.lock();
    if (connection_map.find(db_path_str) != connection_map.end())
    {
        lock.unlock();
        return;
    }
    sqlite3* db = nullptr;
    sqlite3_open(db_path_str, &db);
    connection_map.insert(std::make_pair(db_path_str, db));
    env->ReleaseStringUTFChars(db_path, db_path_str);
    lock.unlock();
}

/*
 * Class:     file_engine_dllInterface_PathMatcher
 * Method:    closeConnections
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_file_engine_dllInterface_PathMatcher_closeConnections
(JNIEnv*, jobject)
{
    lock.lock();
    for (const auto& each_connection : connection_map)
    {
        sqlite3_close(each_connection.second);
    }
    connection_map.clear();
    lock.unlock();
}
