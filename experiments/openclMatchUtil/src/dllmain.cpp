#include "file_engine_dllInterface_OpenCLMatchUtil.h"
#include "opencl.hpp"
#include <vector>
#include "constans.h"
#include "file_utils.h"
#include "utf162gbk_val.h"
#ifdef DEBUG_OUTPUT
#include <iostream>
#endif

uint start_kernel(JNIEnv* env,
	jobjectArray records_to_check,
	const std::vector<std::string>& search_case,
	bool is_ignore_case,
	const char* search_text,
	const std::vector<std::string>& keywords,
	const std::vector<std::string>& keywords_lower_case,
	const bool* is_keyword_path,
	jobject result_collector);
uint collect_results(JNIEnv* thread_env, jobject result_collector, const std::vector<std::string>& search_case_vec,
	Memory<char>& output, Memory<char>& records, unsigned int records_num);


Device current_device;
Memory<unsigned short> p_utf162gbk;

/*
 * Class:     file_engine_dllInterface_OpenCLMatchUtil
 * Method:    isOpenCLAvailable
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_file_engine_dllInterface_OpenCLMatchUtil_isOpenCLAvailable
(JNIEnv*, jclass)
{
	if (auto&& devices = get_devices(); devices.empty())
		return JNI_FALSE;
	current_device = Device(select_device_with_most_flops());
	p_utf162gbk = Memory<unsigned short>(current_device, 0x10000);
	const auto utf162gbk = get_p_utf162gbk();
	for (int i = 0; i < 0x10000; ++i)
	{
		p_utf162gbk[i] = utf162gbk[i];
	}
	p_utf162gbk.write_to_device();
	p_utf162gbk.delete_host_buffer();
	return JNI_TRUE;
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

/*
 * Class:     file_engine_dllInterface_OpenCLMatchUtil
 * Method:    check
 * Signature: ([Ljava/lang/Object;[Ljava/lang/String;ZLjava/lang/String;[Ljava/lang/String;[Ljava/lang/String;[ZLjava/util/function/Consumer;)I
 */
JNIEXPORT jint JNICALL Java_file_engine_dllInterface_OpenCLMatchUtil_check
(JNIEnv* env, jclass, jobjectArray records_to_check, jobjectArray search_case, jboolean is_ignore_case,
	jstring search_text, jobjectArray keywords, jobjectArray keywords_lower, jbooleanArray is_keyword_path, jobject result_collector)
{
	//生成搜索条件 search_case_vec
	std::vector<std::string> search_case_vec;
	if (search_case != nullptr)
	{
		generate_search_case(env, search_case_vec, search_case);
	}
	//生成搜索关键字 keywords_vec keywords_lower_vec is_keyword_path_ptr
	const auto keywords_length = env->GetArrayLength(keywords);
	const auto is_keyword_path_ptr_bool_array = env->GetBooleanArrayElements(is_keyword_path, nullptr);

	std::vector<std::string> keywords_vec;
	std::vector<std::string> keywords_lower_vec;
	const auto is_keyword_path_ptr = new bool[keywords_length];

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
#ifdef DEBUG_OUTPUT
		std::cout << "keywords lower case: " << keywords_chars << std::endl;
#endif
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
	//GPU并行计算
	const auto matched_num = start_kernel(env, records_to_check, search_case_vec, is_ignore_case, search_text_chars,
		keywords_vec, keywords_lower_vec, is_keyword_path_ptr, result_collector);
	env->ReleaseStringUTFChars(search_text, search_text_chars);
	delete[] is_keyword_path_ptr;
	return matched_num;
}

uint start_kernel(JNIEnv* env,
	jobjectArray records_to_check,
	const std::vector<std::string>& search_case,
	bool is_ignore_case,
	const char* search_text,
	const std::vector<std::string>& keywords,
	const std::vector<std::string>& keywords_lower_case,
	const bool* is_keyword_path,
	jobject result_collector)
{
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
	strcpy_s(dev_search_text.data(), MAX_PATH_LENGTH, search_text);
	dev_search_text.write_to_device();

	const auto keywords_num = keywords.size();
	// 复制keywords
	Memory<char> dev_keywords(current_device, keywords_num * MAX_PATH_LENGTH);
	for (size_t i = 0; i < keywords_num; ++i)
	{
		strcpy_s(&dev_keywords[i * MAX_PATH_LENGTH], MAX_PATH_LENGTH, keywords[i].c_str());
	}
	dev_keywords.write_to_device();
#ifdef DEBUG_OUTPUT
	std::cout << "keywords check: " << std::endl;
	for (size_t i = 0; i < keywords_num; ++i)
	{
		std::cout << &dev_keywords[i * MAX_PATH_LENGTH] << std::endl;
}
#endif

	// 复制keywords_lower_case
	Memory<char> dev_keywords_lower_case(current_device, keywords_num * MAX_PATH_LENGTH);
	for (size_t i = 0; i < keywords_num; ++i)
	{
		strcpy_s(&dev_keywords_lower_case[i * MAX_PATH_LENGTH], MAX_PATH_LENGTH, keywords_lower_case[i].c_str());
	}
	dev_keywords_lower_case.write_to_device();
#ifdef DEBUG_OUTPUT
	std::cout << "keywords lower case check: " << std::endl;
	for (size_t i = 0; i < keywords_num; ++i)
	{
		std::cout << &dev_keywords_lower_case[i * MAX_PATH_LENGTH] << std::endl;
	}
#endif

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

	const auto size = env->GetArrayLength(records_to_check);
	Memory<char> all_records_dev(current_device, size * MAX_PATH_LENGTH);
	all_records_dev.reset();
	for (jsize i = 0; i < size; ++i)
	{
		const auto record_obj = reinterpret_cast<jstring>(env->GetObjectArrayElement(records_to_check, i));
		if (const auto record = env->GetStringUTFChars(record_obj, nullptr); strlen(record) < MAX_PATH_LENGTH)
		{
			strcpy_s(&all_records_dev[i * MAX_PATH_LENGTH], MAX_PATH_LENGTH, record);
		}
		env->DeleteLocalRef(record_obj);
	}
	all_records_dev.write_to_device();
#ifdef DEBUG_OUTPUT
	std::cout << "records check: " << std::endl;
	for (size_t i = 0; i < size; ++i)
	{
		std::cout << &all_records_dev[i * MAX_PATH_LENGTH] << std::endl;
	}
#endif

	Memory<unsigned int> size_dev(current_device, 1);
	size_dev[0] = size;
	size_dev.write_to_device();

	Memory<char> output(current_device, size);
	output.reset();
	output.write_to_device();

	cl_int ret = 0;
	Kernel search_kernel(current_device, size, "check", &ret,
		all_records_dev,
		size_dev,
		dev_search_case,
		dev_is_ignore_case,
		dev_search_text,
		dev_keywords,
		dev_keywords_lower_case,
		dev_keywords_length,
		dev_is_keyword_path,
		output,
		p_utf162gbk);
	if (ret != CL_SUCCESS)
	{
		fprintf(stderr, "OpenCL: init kernel failed. Error code: %d\n", ret);
	}
	search_kernel(&ret);
	if (ret != CL_SUCCESS)
	{
		fprintf(stderr, "OpenCL: start kernel function failed. Error code: %d\n", ret);
	}
	output.read_from_device();
	return collect_results(env, result_collector, search_case, output, all_records_dev, size);
}

/**
 * \brief 将显卡计算的结果保存到hashmap中
 */
uint collect_results(JNIEnv* thread_env, jobject result_collector, const std::vector<std::string>& search_case_vec,
	Memory<char>& output, Memory<char>& records, const unsigned int records_num)
{
	uint matched_result_num = 0;
	const jclass biconsumer_class = thread_env->GetObjectClass(result_collector);
	const jmethodID collector = thread_env->GetMethodID(biconsumer_class, "accept", "(Ljava/lang/Object;)V");
	auto _collect_func = [&](char _matched_record_str[MAX_PATH_LENGTH])
	{
		auto record_jstring = thread_env->NewStringUTF(_matched_record_str);
		thread_env->CallVoidMethod(result_collector, collector, record_jstring);
		thread_env->DeleteLocalRef(record_jstring);
		++matched_result_num;
	};
	for (size_t i = 0; i < records_num; ++i)
	{
		//dev_cache[i]字符串匹配成功
		if (output[i])
		{
			char matched_record_str[MAX_PATH_LENGTH]{ 0 };
			strcpy_s(matched_record_str, &records[i * MAX_PATH_LENGTH]);
			// 判断文件和文件夹
			if (search_case_vec.empty())
			{
				_collect_func(matched_record_str);
			}
			else
			{
				if (std::find(search_case_vec.begin(), search_case_vec.end(), "f") != search_case_vec.end())
				{
					if (is_dir_or_file(matched_record_str) == 1)
					{
						_collect_func(matched_record_str);
					}
				}
				else if (std::find(search_case_vec.begin(), search_case_vec.end(), "d") != search_case_vec.end())
				{
					if (is_dir_or_file(matched_record_str) == 0)
					{
						_collect_func(matched_record_str);
					}
				}
				else
				{
					_collect_func(matched_record_str);
				}
			}
		}
	}
	thread_env->DeleteLocalRef(biconsumer_class);
	return matched_result_num;
}
