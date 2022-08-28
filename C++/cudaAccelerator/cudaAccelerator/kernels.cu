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
#include "str_utils.cuh"
#include "str_convert.cuh"

inline void gpuAssert(cudaError_t code, const char* file, int line, const char* function, bool is_exit,
                      const char* info)
{
	if (code != cudaSuccess)
	{
		if (info == nullptr)
		{
			fprintf(stderr, "GPU assert: %s %s %d %s\n", cudaGetErrorString(code), file, line, function);
		}
		else
		{
			fprintf(stderr, "GPU assert: %s %s %d %s %s\n", cudaGetErrorString(code), file, line, function, info);
		}
		if (is_exit)
		{
			std::quick_exit(code);
		}
	}
}

__device__ void convert_to_pinyin(const char* chinese_str, char* output_str)
{
	static constexpr int spell_value[] = {
		-20319, -20317, -20304, -20295, -20292, -20283, -20265, -20257, -20242, -20230, -20051, -20036, -20032, -20026,
		-20002, -19990, -19986, -19982, -19976, -19805, -19784, -19775, -19774, -19763, -19756, -19751, -19746, -19741,
		-19739, -19728,
		-19725, -19715, -19540, -19531, -19525, -19515, -19500, -19484, -19479, -19467, -19289, -19288, -19281, -19275,
		-19270, -19263,
		-19261, -19249, -19243, -19242, -19238, -19235, -19227, -19224, -19218, -19212, -19038, -19023, -19018, -19006,
		-19003, -18996,
		-18977, -18961, -18952, -18783, -18774, -18773, -18763, -18756, -18741, -18735, -18731, -18722, -18710, -18697,
		-18696, -18526,
		-18518, -18501, -18490, -18478, -18463, -18448, -18447, -18446, -18239, -18237, -18231, -18220, -18211, -18201,
		-18184, -18183,
		-18181, -18012, -17997, -17988, -17970, -17964, -17961, -17950, -17947, -17931, -17928, -17922, -17759, -17752,
		-17733, -17730,
		-17721, -17703, -17701, -17697, -17692, -17683, -17676, -17496, -17487, -17482, -17468, -17454, -17433, -17427,
		-17417, -17202,
		-17185, -16983, -16970, -16942, -16915, -16733, -16708, -16706, -16689, -16664, -16657, -16647, -16474, -16470,
		-16465, -16459,
		-16452, -16448, -16433, -16429, -16427, -16423, -16419, -16412, -16407, -16403, -16401, -16393, -16220, -16216,
		-16212, -16205,
		-16202, -16187, -16180, -16171, -16169, -16158, -16155, -15959, -15958, -15944, -15933, -15920, -15915, -15903,
		-15889, -15878,
		-15707, -15701, -15681, -15667, -15661, -15659, -15652, -15640, -15631, -15625, -15454, -15448, -15436, -15435,
		-15419, -15416,
		-15408, -15394, -15385, -15377, -15375, -15369, -15363, -15362, -15183, -15180, -15165, -15158, -15153, -15150,
		-15149, -15144,
		-15143, -15141, -15140, -15139, -15128, -15121, -15119, -15117, -15110, -15109, -14941, -14937, -14933, -14930,
		-14929, -14928,
		-14926, -14922, -14921, -14914, -14908, -14902, -14894, -14889, -14882, -14873, -14871, -14857, -14678, -14674,
		-14670, -14668,
		-14663, -14654, -14645, -14630, -14594, -14429, -14407, -14399, -14384, -14379, -14368, -14355, -14353, -14345,
		-14170, -14159,
		-14151, -14149, -14145, -14140, -14137, -14135, -14125, -14123, -14122, -14112, -14109, -14099, -14097, -14094,
		-14092, -14090,
		-14087, -14083, -13917, -13914, -13910, -13907, -13906, -13905, -13896, -13894, -13878, -13870, -13859, -13847,
		-13831, -13658,
		-13611, -13601, -13406, -13404, -13400, -13398, -13395, -13391, -13387, -13383, -13367, -13359, -13356, -13343,
		-13340, -13329,
		-13326, -13318, -13147, -13138, -13120, -13107, -13096, -13095, -13091, -13076, -13068, -13063, -13060, -12888,
		-12875, -12871,
		-12860, -12858, -12852, -12849, -12838, -12831, -12829, -12812, -12802, -12607, -12597, -12594, -12585, -12556,
		-12359, -12346,
		-12320, -12300, -12120, -12099, -12089, -12074, -12067, -12058, -12039, -11867, -11861, -11847, -11831, -11798,
		-11781, -11604,
		-11589, -11536, -11358, -11340, -11339, -11324, -11303, -11097, -11077, -11067, -11055, -11052, -11045, -11041,
		-11038, -11024,
		-11020, -11019, -11018, -11014, -10838, -10832, -10815, -10800, -10790, -10780, -10764, -10587, -10544, -10533,
		-10519, -10331,
		-10329, -10328, -10322, -10315, -10309, -10307, -10296, -10281, -10274, -10270, -10262, -10260, -10256, -10254
	};

	// 395个字符串，每个字符串长度不超过6
	static constexpr char spell_dict[396][7] = {
		"a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng", "bi", "bian", "biao",
		"bie", "bin", "bing", "bo", "bu", "ca", "cai", "can", "cang", "cao", "ce", "ceng", "cha", "chai", "chan",
		"chang", "chao", "che", "chen",
		"cheng", "chi", "chong", "chou", "chu", "chuai", "chuan", "chuang", "chui", "chun", "chuo", "ci", "cong", "cou",
		"cu", "cuan", "cui",
		"cun", "cuo", "da", "dai", "dan", "dang", "dao", "de", "deng", "di", "dian", "diao", "die", "ding", "diu",
		"dong", "dou", "du", "duan",
		"dui", "dun", "duo", "e", "en", "er", "fa", "fan", "fang", "fei", "fen", "feng", "fo", "fou", "fu", "ga", "gai",
		"gan", "gang", "gao",
		"ge", "gei", "gen", "geng", "gong", "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo", "ha",
		"hai", "han", "hang",
		"hao", "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun", "huo",
		"ji", "jia", "jian",
		"jiang", "jiao", "jie", "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun", "ka", "kai", "kan", "kang",
		"kao", "ke", "ken",
		"keng", "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui", "kun", "kuo", "la", "lai", "lan", "lang",
		"lao", "le", "lei",
		"leng", "li", "lia", "lian", "liang", "liao", "lie", "lin", "ling", "liu", "long", "lou", "lu", "lv", "luan",
		"lue", "lun", "luo",
		"ma", "mai", "man", "mang", "mao", "me", "mei", "men", "meng", "mi", "mian", "miao", "mie", "min", "ming",
		"miu", "mo", "mou", "mu",
		"na", "nai", "nan", "nang", "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang", "niao", "nie", "nin",
		"ning", "niu", "nong",
		"nu", "nv", "nuan", "nue", "nuo", "o", "ou", "pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng", "pi",
		"pian", "piao", "pie",
		"pin", "ping", "po", "pu", "qi", "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu", "qu",
		"quan", "que", "qun",
		"ran", "rang", "rao", "re", "ren", "reng", "ri", "rong", "rou", "ru", "ruan", "rui", "run", "ruo", "sa", "sai",
		"san", "sang",
		"sao", "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao", "she", "shen", "sheng", "shi", "shou",
		"shu", "shua",
		"shuai", "shuan", "shuang", "shui", "shun", "shuo", "si", "song", "sou", "su", "suan", "sui", "sun", "suo",
		"ta", "tai",
		"tan", "tang", "tao", "te", "teng", "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui",
		"tun", "tuo",
		"wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia", "xian", "xiang", "xiao", "xie",
		"xin", "xing",
		"xiong", "xiu", "xu", "xuan", "xue", "xun", "ya", "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong",
		"you",
		"yu", "yuan", "yue", "yun", "za", "zai", "zan", "zang", "zao", "ze", "zei", "zen", "zeng", "zha", "zhai",
		"zhan", "zhang",
		"zhao", "zhe", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang", "zhui",
		"zhun", "zhuo",
		"zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo"
	};
	// 循环处理字节数组
	const auto length = strlen_cuda(chinese_str);
	for (size_t j = 0; j < length;)
	{
		// 非汉字处理
		const unsigned char val = chinese_str[j];
		if (val < 128)
		{
			str_add_single(output_str, chinese_str[j]);
			// 偏移下标
			++j;
			continue;
		}

		// 汉字处理
		const int chrasc = chinese_str[j] * 256 + chinese_str[j + 1] + 256;
		if (chrasc > 0 && chrasc < 160)
		{
			// 非汉字
			str_add_single(output_str, chinese_str[j]);
			// 偏移下标
			++j;
		}
		else
		{
			// 汉字
			for (int i = sizeof spell_value / sizeof spell_value[0] - 1; i >= 0; --i)
			{
				// 查找字典
				if (spell_value[i] <= chrasc)
				{
					strcat_cuda(output_str, spell_dict[i]);
					break;
				}
			}
			// 偏移下标 （汉字双字节）
			j += 2;
		}
	}
}

__device__ bool not_matched(const char* path,
                            const bool is_ignore_case,
                            char* keywords,
                            char* keywords_lower_case,
                            const int keywords_length,
                            const bool* is_keyword_path)
{
	for (int i = 0; i < keywords_length; ++i)
	{
		const bool is_keyword_path_val = is_keyword_path[i];
		char match_str[MAX_PATH_LENGTH]{0};
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
			strlwr_cuda(match_str);
		}
		else
		{
			each_keyword = keywords + i * static_cast<unsigned long long>(MAX_PATH_LENGTH);
		}
		if (!each_keyword[0])
		{
			continue;
		}
		if (!match_str[0] || strstr_cuda(match_str, each_keyword) == nullptr)
		{
			if (is_keyword_path_val || !is_str_contains_chinese(match_str))
			{
				return true;
			}
			char gbk_buffer[MAX_PATH_LENGTH * 2];
			char* gbk_buffer_ptr = gbk_buffer;
			unsigned gbk_buffer_size = MAX_PATH_LENGTH * 2;
			// utf-8编码转换gbk
			str_normalize_init();
			utf8_to_gbk(match_str, static_cast<unsigned>(strlen_cuda(match_str)), &gbk_buffer_ptr, &gbk_buffer_size);
			char converted_pinyin[MAX_PATH_LENGTH * 6]{0};
			convert_to_pinyin(gbk_buffer, converted_pinyin);
			if (strstr_cuda(converted_pinyin, each_keyword) == nullptr)
			{
				return true;
			}
		}
	}
	return false;
}

__global__ void check(const char (* str_address_ptr_array)[MAX_PATH_LENGTH],
                      const int* search_case,
                      const bool* is_ignore_case,
                      char* search_text,
                      char* keywords,
                      char* keywords_lower_case,
                      const size_t* keywords_length,
                      const bool* is_keyword_path,
                      char* output,
                      const bool* is_stop_collect_var)
{
	const size_t thread_id = GET_TID();
	const char* path = reinterpret_cast<const char*>(str_address_ptr_array + thread_id);
	if (*is_stop_collect_var)
	{
		return;
	}
#ifdef DEBUG_OUTPUT
	printf("%s\n", path);
#endif
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
	if ((*search_case & 4) == 4)
	{
		// 全字匹配
		strlwr_cuda(search_text);
		char file_name[MAX_PATH_LENGTH];
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
                  const bool* is_keyword_path,
                  cudaStream_t* streams,
                  const size_t stream_count)
{
	int* dev_search_case = nullptr;
	char* dev_search_text = nullptr;
	char* dev_keywords = nullptr;
	char* dev_keywords_lower_case = nullptr;
	size_t* dev_keywords_length = nullptr;
	bool* dev_is_keyword_path = nullptr;
	bool* dev_is_ignore_case = nullptr;

	const auto keywords_num = keywords.size();

	//初始化流
	for (size_t i = 0; i < stream_count; ++i)
	{
		gpuErrchk(cudaStreamCreate(&streams[i]), true, nullptr)
	}
	do
	{
		// 选择第一个GPU
		gpuErrchk(cudaSetDevice(0), true, nullptr)

		// 复制search case
		// 第一位为1表示有F，第二位为1表示有D，第三位为1表示有FULL，CASE由File-Engine主程序进行判断，传入参数is_ignore_case为false表示有CASE
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_search_case), sizeof(int)), true, nullptr)
		int search_case_num = 0;
		for (auto& each_case : search_case)
		{
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
		gpuErrchk(cudaMemcpy(dev_search_case, &search_case_num, sizeof(int), cudaMemcpyHostToDevice), true, nullptr)

		// 复制search text
		const auto search_text_len = strlen(search_text);
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_search_text), (search_text_len + 1) * sizeof(char)), true,
		          nullptr)
		gpuErrchk(cudaMemset(dev_search_text, 0, search_text_len + 1), true, nullptr)
		gpuErrchk(cudaMemcpy(dev_search_text, search_text, search_text_len, cudaMemcpyHostToDevice), true, nullptr)

		// 复制keywords
		gpuErrchk(vector_to_cuda_char_array(keywords, reinterpret_cast<void**>(&dev_keywords)), true, nullptr)

		// 复制keywords_lower_case
		gpuErrchk(vector_to_cuda_char_array(keywords_lower_case, reinterpret_cast<void**>(&dev_keywords_lower_case)),
		          true, nullptr)

		//复制keywords_length
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_keywords_length), sizeof(size_t)), true, nullptr)
		gpuErrchk(cudaMemcpy(dev_keywords_length, &keywords_num, sizeof(size_t), cudaMemcpyHostToDevice), true, nullptr)

		// 复制is_keyword_path
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_is_keyword_path), sizeof(bool) * keywords_num), true,
		          nullptr)
		gpuErrchk(cudaMemcpy(dev_is_keyword_path, is_keyword_path, sizeof(bool) * keywords_num, cudaMemcpyHostToDevice),
		          true, nullptr)

		// 复制is_ignore_case
		gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&dev_is_ignore_case), sizeof(bool)), true, nullptr)
		gpuErrchk(cudaMemcpy(dev_is_ignore_case, &is_ignore_case, sizeof(bool), cudaMemcpyHostToDevice), true, nullptr)
		int count = 0;
		for (auto& each : cache_map)
		{
			const auto& cache = each.second;
			if (!cache->is_cache_valid)
			{
				continue;
			}
			int block_num, thread_num;
			if (cache->str_data.record_num > MAX_THREAD_PER_BLOCK)
			{
				thread_num = MAX_THREAD_PER_BLOCK;
				block_num = static_cast<int>(cache->str_data.record_num / thread_num);
			}
			else
			{
				thread_num = static_cast<int>(cache->str_data.record_num.load());
				block_num = 1;
			}
			//注册回调
			cudaStreamAddCallback(streams[count], set_match_done_flag_callback, cache, 0);

			check<<<block_num, thread_num, 0, streams[count]>>>
			(cache->str_data.dev_cache_str,
			 dev_search_case,
			 dev_is_ignore_case,
			 dev_search_text,
			 dev_keywords,
			 dev_keywords_lower_case,
			 dev_keywords_length,
			 dev_is_keyword_path,
			 cache->dev_output,
			 get_dev_stop_signal());
			++count;
		}

		// 检查启动错误
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "check launch failed: %s\n", cudaGetErrorString(cudaStatus));
			break;
		}
	}
	while (false);
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
