#include "kernel.hpp" 

// TODO 修改constant.h中的宏同时需要修改核函数中的宏定义
string opencl_c_container() {
	// ########################## begin of OpenCL C code ####################################################################
	return string(R"(
#define MAX_PATH_LENGTH 300
#define MAX_KEYWORDS_NUMBER 150

#define COMPBYTE(x, y) ((unsigned char)(x) << 8 | (unsigned char)(y))
#ifndef NULL
#define NULL ((void*) 0)
#endif

)") + R"(

constant const int spell_value[] = {
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
)" + R"(
// 395个字符串，每个字符串长度不超过6
constant const char spell_dict[396][7] = {
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
)" + R"(

// #################################functions#################################

void copy_from_global(char* dst, global const char* src, const unsigned count)
{
	for (unsigned i = 0; i < count; ++i)
	{
		*dst++ = *src++;
	}
}

void copy_from_constant(char* dst, constant const char* src, const unsigned count)
{
	for (unsigned i = 0; i < count; ++i)
	{
		*dst++ = *src++;
	}
}

unsigned strlen_constant(constant const char* str)
{
	unsigned count = 0;
	while (*str != '\0')
	{
		count++;
		++str;
	}
	return count;
}

unsigned strlen_global(global const char* str)
{
	unsigned count = 0;
	while (*str != '\0')
	{
		count++;
		++str;
	}
	return count;
}

char not_matched(global const char* path,
	char is_ignore_case,
	global const char* keywords,
	global const char* keywords_lower_case,
	unsigned keywords_length,
	global const char* is_keyword_path,
	global const unsigned short* p_utf162gbk);
void get_parent_path(global const char* path, char* output);
void get_file_name(global const char* path, char* output);
char is_str_contains_chinese(const char* source);
void convert_to_pinyin(const char* chinese_str, char* output_str, char* pinyin_initials);
void str_add_single(char* dst, char c);
void utf8_to_gbk(global const unsigned short* p_utf162gbk, const char* from, unsigned int from_len, char** to);
char* strcat(char* dst, char const* src);
int strcmp(const char* str1, const char* str2);
unsigned strlen(const char* str);
char* strlwr(char* src);
char* strstr(char* s1, char* s2);

kernel void check(global const char* str_address_ptr_array,
	global const unsigned int* total_num,
	global const int* search_case,
	global const char* is_ignore_case,
	global const char* search_text,
	global const char* keywords,
	global const char* keywords_lower_case,
	global const unsigned* keywords_length,
	global const char* is_keyword_path,
	global char* output,
	global const char* is_stop_collect_var,
	global const unsigned short* p_utf162gbk)
{
	const unsigned long thread_id = get_global_id(0);
	if (thread_id >= *total_num)
	{
		return;
	}
	if (*is_stop_collect_var)
	{
		return;
	}
	global const char* path = str_address_ptr_array + thread_id * MAX_PATH_LENGTH;
	if (path == NULL || !path[0])
	{
		return;
	}
	if (not_matched(path, *is_ignore_case, keywords, keywords_lower_case, *keywords_length, is_keyword_path, p_utf162gbk))
	{
		return;
	}
	if (*search_case == 0)
	{
		output[thread_id] = 1;
		return;
	}
	if (*search_case & 4)
	{
		// 全字匹配
		char search_text_local[MAX_PATH_LENGTH] = { 0 };
		copy_from_global(search_text_local, search_text, strlen_global(search_text));
		strlwr(search_text_local);

		char file_name[MAX_PATH_LENGTH] = { 0 };
		get_file_name(path, file_name);
		strlwr(file_name);

		if (strcmp(search_text_local, file_name) != 0)
		{
			return;
		}
	}
	output[thread_id] = 1;
}

char not_matched(global const char* path,
	const char is_ignore_case,
	global const char* keywords,
	global const char* keywords_lower_case,
	const unsigned keywords_length,
	global const char* is_keyword_path,
	global const unsigned short* p_utf162gbk)
{
	for (unsigned i = 0; i < keywords_length; ++i)
	{
		const char is_keyword_path_val = is_keyword_path[i];
		char match_str[MAX_PATH_LENGTH] = { 0 };
		if (is_keyword_path_val)
		{
			get_parent_path(path, match_str);
		}
		else
		{
			get_file_name(path, match_str);
		}
		global char* each_keyword;
		if (is_ignore_case)
		{
			each_keyword = keywords_lower_case + i * (unsigned)MAX_PATH_LENGTH;
			strlwr(match_str);
		}
		else
		{
			each_keyword = keywords + i * (unsigned)MAX_PATH_LENGTH;
		}
		if (!each_keyword[0])
		{
			continue;
		}
		char each_keyword_local[MAX_PATH_LENGTH] = { 0 };
		copy_from_global(each_keyword_local, each_keyword, strlen_global(each_keyword));
		if (strstr(match_str, each_keyword_local) == NULL)
		{
			if (is_keyword_path_val || !is_str_contains_chinese(match_str))
			{
				return 1;
			}
			char gbk_buffer[MAX_PATH_LENGTH * 2] = { 0 };
			char* gbk_buffer_ptr = gbk_buffer;
			// utf-8编码转换gbk
			utf8_to_gbk(p_utf162gbk, match_str, strlen(match_str), &gbk_buffer_ptr);
			char converted_pinyin[MAX_PATH_LENGTH * 6] = { 0 };
			char converted_pinyin_initials[MAX_PATH_LENGTH] = { 0 };
			convert_to_pinyin(gbk_buffer, converted_pinyin, converted_pinyin_initials);
			if (strstr(converted_pinyin, each_keyword_local) == NULL && 
				strstr(converted_pinyin_initials, each_keyword_local) == NULL)
			{
				return 1;
			}
		}
	}
	return 0;
}

void convert_to_pinyin(const char* chinese_str, char* output_str, char* pinyin_initials)
{
	// 循环处理字节数组
	const unsigned length = strlen(chinese_str);
	for (unsigned j = 0; j < length;)
	{
		// 非汉字处理
		const unsigned char val = chinese_str[j];
		if (val < 128)
		{
			str_add_single(output_str, chinese_str[j]);
			str_add_single(pinyin_initials, chinese_str[j]);
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
			str_add_single(pinyin_initials, chinese_str[j]);
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
					constant const char* p = spell_dict[i];
					char spell_dict_local[7] = { 0 };
					copy_from_constant(spell_dict_local, p, strlen_constant(p));
					strcat(output_str, spell_dict_local);
					str_add_single(pinyin_initials, spell_dict_local[0]);
					break;
				}
			}
			// 偏移下标 （汉字双字节）
			j += 2;
		}
	}
}

int strcmp(const char* str1, const char* str2)
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

unsigned strlen(const char* str)
{
	unsigned count = 0;
	while (*str != '\0')
	{
		count++;
		++str;
	}
	return count;
}

char* strlwr(char* src)
{
	while (*src != '\0')
	{
		if (*src >= 'A' && *src <= 'Z')
		{
			*src += 32;
		}
		++src;
	}
	return src;
}


char* strstr(char* s1, char* s2)
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
		return NULL;
	}
	return s1;
}

char* strrchr(const char* s, int c)
{
	if (s == NULL)
	{
		return NULL;
	}

	char* p_char = NULL;
	while (*s != '\0')
	{
		if (*s == (char)c)
		{
			p_char = (char*)s;
		}
		++s;
	}

	return p_char;
}

char* strcpy(char* dst, const char* src)
{
	char* ret = dst;
	while ((*dst++ = *src++) != '\0')
	{
	}
	return ret;
}

char* strcat(char* dst, char const* src)
{
	if (dst == NULL || src == NULL)
	{
		return NULL;
	}

	char* tmp = dst;

	while (*dst != '\0') //这个循环结束之后，dst指向'\0'
	{
		dst++;
	}

	while (*src != '\0')
	{
		*dst++ = *src++; //把src指向的内容赋值给dst
	}

	*dst = '\0'; //这句一定要加，否则最后一个字符会乱码
	return tmp;
} 

void str_add_single(char* dst, const char c)
{
	while (*dst != '\0')
	{
		dst++;
	}
	*dst++ = c;
	*dst = '\0';
}

char is_str_contains_chinese(const char* source)
{
	int i = 0;
	while (source[i] != 0)
	{
		if (source[i] & 0x80 && source[i] & 0x40 && source[i] & 0x20)
		{
			return 1;
		}
		if (source[i] & 0x80 && source[i] & 0x40)
		{
			i += 2;
		}
		else
		{
			i += 1;
		}
	}
	return 0;
}

void get_file_name(global const char* path, char* output)
{
	char path_local[MAX_PATH_LENGTH] = { 0 };
	copy_from_global(path_local, path, MAX_PATH_LENGTH);
	const char* p = strrchr(path_local, '\\');
	if (p == NULL)
		strcpy(output, path_local);
	else
		strcpy(output, p + 1);
}

void get_parent_path(global const char* path, char* output)
{
	char path_local[MAX_PATH_LENGTH] = { 0 };
	copy_from_global(path_local, path, MAX_PATH_LENGTH);
	strcpy(output, path_local);
	char* p = strrchr(output, '\\');
	if (p != NULL)
		*p = '\0';
}

void utf8_to_gbk(global const unsigned short* p_utf162gbk, const char* from, unsigned int from_len, char** to)
{
	char* result = *to;
	unsigned i_to = 0;

	if (from_len == 0 || from == NULL || to == NULL || result == NULL)
	{
		return;
	}

	for (unsigned i_from = 0; i_from < from_len;)
	{
		if ((unsigned char)from[i_from] < 0x80)
		{
			result[i_to++] = from[i_from++];
		}
		else if ((unsigned char)from[i_from] < 0xC2)
		{
			i_from++;
		}
		else if ((unsigned char)from[i_from] < 0xE0)
		{
			if (i_from >= from_len - 1) break;

			const unsigned short tmp = p_utf162gbk[(from[i_from] & 0x1F) << 6 | from[i_from + 1] & 0x3F];

			if (tmp)
			{
				result[i_to++] = tmp >> 8;
				result[i_to++] = tmp & 0xFF;
			}

			i_from += 2;
		}
		else if ((unsigned char)from[i_from] < 0xF0)
		{
			if (i_from >= from_len - 2) break;

			const unsigned short tmp = p_utf162gbk[(from[i_from] & 0x0F) << 12
				| (from[i_from + 1] & 0x3F) << 6 | from[i_from + 2] & 0x3F];

			if (tmp)
			{
				result[i_to++] = tmp >> 8;
				result[i_to++] = tmp & 0xFF;
			}

			i_from += 3;
		}
		else
		{
			i_from += 4;
		}
	}

	result[i_to] = 0;
}

	)";
} // ############################################################### end of OpenCL C code #####################################################################
