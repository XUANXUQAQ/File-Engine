#include <string>
#include <io.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ctime>
using namespace std;
//#define TEST

vector<string> ignorePathVector;
int searchDepth;
char *searchPath;
string results0_100;
string results100_200;
string results200_300;
string results300_400;
string results400_500;
string results500_600;
string results600_700;
string results700_800;
string results800_900;
string results900_1000;
string results1000_1100;
string results1100_1200;
string results1200_1300;
string results1300_1400;
string results1400_1500;
string results1500_1600;
string results1600_1700;
string results1700_1800;
string results1800_1900;
string results1900_2000;
string results2000_2100;
string results2100_2200;
string results2200_2300;
string results2300_2400;
string results2400_2500;
string results2500_;

void searchFiles(const char *path, const char *exd);
void addIgnorePath(const char *path);
int count(string path, string pattern);
bool isSearchDepthOut(string path);
bool isIgnore(string path);
void search(string path, string exd);
void searchIgnoreSearchDepth(string path, string exd);
void clearResults();

int getAscIISum(string name)
{
    int sum = 0;
    for (int i = 0; i < name.length(); i++)
    {
        sum += name[i];
    }
    return sum;
}

void saveResult(string path, int ascII)
{
    if (0 < ascII && ascII <= 100)
    {
        results0_100.append(path);
        results0_100.append("\n");
    }
    else if (100 < ascII && ascII <= 200)
    {
        results100_200.append(path);
        results100_200.append("\n");
    }
    else if (200 < ascII && ascII <= 300)
    {
        results200_300.append(path);
        results200_300.append("\n");
    }
    else if (300 < ascII && ascII <= 400)
    {
        results300_400.append(path);
        results300_400.append("\n");
    }
    else if (400 < ascII && ascII <= 500)
    {
        results400_500.append(path);
        results400_500.append("\n");
    }
    else if (500 < ascII && ascII <= 600)
    {
        results500_600.append(path);
        results500_600.append("\n");
    }
    else if (600 < ascII && ascII <= 700)
    {
        results600_700.append(path);
        results600_700.append("\n");
    }
    else if (700 < ascII && ascII <= 800)
    {
        results700_800.append(path);
        results700_800.append("\n");
    }
    else if (800 < ascII && ascII <= 900)
    {
        results800_900.append(path);
        results800_900.append("\n");
    }
    else if (900 < ascII && ascII <= 1000)
    {
        results900_1000.append(path);
        results900_1000.append("\n");
    }
    else if (1000 < ascII && ascII <= 1100)
    {
        results1000_1100.append(path);
        results1000_1100.append("\n");
    }
    else if (1100 < ascII && ascII <= 1200)
    {
        results1100_1200.append(path);
        results1100_1200.append("\n");
    }
    else if (1200 < ascII && ascII <= 1300)
    {
        results1200_1300.append(path);
        results1200_1300.append("\n");
    }
    else if (1300 < ascII && ascII <= 1400)
    {
        results1300_1400.append(path);
        results1300_1400.append("\n");
    }
    else if (1400 < ascII && ascII <= 1500)
    {
        results1400_1500.append(path);
        results1400_1500.append("\n");
    }
    else if (1500 < ascII && ascII <= 1600)
    {
        results1500_1600.append(path);
        results1500_1600.append("\n");
    }
    else if (1600 < ascII && ascII <= 1700)
    {
        results1600_1700.append(path);
        results1600_1700.append("\n");
    }
    else if (1700 < ascII && ascII <= 1800)
    {
        results1700_1800.append(path);
        results1700_1800.append("\n");
    }
    else if (1800 < ascII && ascII <= 1900)
    {
        results1800_1900.append(path);
        results1800_1900.append("\n");
    }
    else if (1900 < ascII && ascII <= 2000)
    {
        results1900_2000.append(path);
        results1900_2000.append("\n");
    }
    else if (2000 < ascII && ascII <= 2100)
    {
        results2000_2100.append(path);
        results2000_2100.append("\n");
    }
    else if (2100 < ascII && ascII <= 2200)
    {
        results2100_2200.append(path);
        results2100_2200.append("\n");
    }
    else if (2200 < ascII && ascII <= 2300)
    {
        results2200_2300.append(path);
        results2200_2300.append("\n");
    }
    else if (2300 < ascII && ascII <= 2400)
    {
        results2300_2400.append(path);
        results2300_2400.append("\n");
    }
    else if (2400 < ascII && ascII <= 2500)
    {
        results2400_2500.append(path);
        results2400_2500.append("\n");
    }
    else
    {
        results2500_.append(path);
        results2500_.append("\n");
    }
}

int ChineseConvertPy(const std::string &dest_chinese, std::string &out_py)
{
    const int spell_value[] = {-20319, -20317, -20304, -20295, -20292, -20283, -20265, -20257, -20242, -20230, -20051, -20036, -20032, -20026,
                               -20002, -19990, -19986, -19982, -19976, -19805, -19784, -19775, -19774, -19763, -19756, -19751, -19746, -19741, -19739, -19728,
                               -19725, -19715, -19540, -19531, -19525, -19515, -19500, -19484, -19479, -19467, -19289, -19288, -19281, -19275, -19270, -19263,
                               -19261, -19249, -19243, -19242, -19238, -19235, -19227, -19224, -19218, -19212, -19038, -19023, -19018, -19006, -19003, -18996,
                               -18977, -18961, -18952, -18783, -18774, -18773, -18763, -18756, -18741, -18735, -18731, -18722, -18710, -18697, -18696, -18526,
                               -18518, -18501, -18490, -18478, -18463, -18448, -18447, -18446, -18239, -18237, -18231, -18220, -18211, -18201, -18184, -18183,
                               -18181, -18012, -17997, -17988, -17970, -17964, -17961, -17950, -17947, -17931, -17928, -17922, -17759, -17752, -17733, -17730,
                               -17721, -17703, -17701, -17697, -17692, -17683, -17676, -17496, -17487, -17482, -17468, -17454, -17433, -17427, -17417, -17202,
                               -17185, -16983, -16970, -16942, -16915, -16733, -16708, -16706, -16689, -16664, -16657, -16647, -16474, -16470, -16465, -16459,
                               -16452, -16448, -16433, -16429, -16427, -16423, -16419, -16412, -16407, -16403, -16401, -16393, -16220, -16216, -16212, -16205,
                               -16202, -16187, -16180, -16171, -16169, -16158, -16155, -15959, -15958, -15944, -15933, -15920, -15915, -15903, -15889, -15878,
                               -15707, -15701, -15681, -15667, -15661, -15659, -15652, -15640, -15631, -15625, -15454, -15448, -15436, -15435, -15419, -15416,
                               -15408, -15394, -15385, -15377, -15375, -15369, -15363, -15362, -15183, -15180, -15165, -15158, -15153, -15150, -15149, -15144,
                               -15143, -15141, -15140, -15139, -15128, -15121, -15119, -15117, -15110, -15109, -14941, -14937, -14933, -14930, -14929, -14928,
                               -14926, -14922, -14921, -14914, -14908, -14902, -14894, -14889, -14882, -14873, -14871, -14857, -14678, -14674, -14670, -14668,
                               -14663, -14654, -14645, -14630, -14594, -14429, -14407, -14399, -14384, -14379, -14368, -14355, -14353, -14345, -14170, -14159,
                               -14151, -14149, -14145, -14140, -14137, -14135, -14125, -14123, -14122, -14112, -14109, -14099, -14097, -14094, -14092, -14090,
                               -14087, -14083, -13917, -13914, -13910, -13907, -13906, -13905, -13896, -13894, -13878, -13870, -13859, -13847, -13831, -13658,
                               -13611, -13601, -13406, -13404, -13400, -13398, -13395, -13391, -13387, -13383, -13367, -13359, -13356, -13343, -13340, -13329,
                               -13326, -13318, -13147, -13138, -13120, -13107, -13096, -13095, -13091, -13076, -13068, -13063, -13060, -12888, -12875, -12871,
                               -12860, -12858, -12852, -12849, -12838, -12831, -12829, -12812, -12802, -12607, -12597, -12594, -12585, -12556, -12359, -12346,
                               -12320, -12300, -12120, -12099, -12089, -12074, -12067, -12058, -12039, -11867, -11861, -11847, -11831, -11798, -11781, -11604,
                               -11589, -11536, -11358, -11340, -11339, -11324, -11303, -11097, -11077, -11067, -11055, -11052, -11045, -11041, -11038, -11024,
                               -11020, -11019, -11018, -11014, -10838, -10832, -10815, -10800, -10790, -10780, -10764, -10587, -10544, -10533, -10519, -10331,
                               -10329, -10328, -10322, -10315, -10309, -10307, -10296, -10281, -10274, -10270, -10262, -10260, -10256, -10254};

    // 395个字符串，每个字符串长度不超过6
    const char spell_dict[396][7] = {"a", "ai", "an", "ang", "ao", "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng", "bi", "bian", "biao",
                                     "bie", "bin", "bing", "bo", "bu", "ca", "cai", "can", "cang", "cao", "ce", "ceng", "cha", "chai", "chan", "chang", "chao", "che", "chen",
                                     "cheng", "chi", "chong", "chou", "chu", "chuai", "chuan", "chuang", "chui", "chun", "chuo", "ci", "cong", "cou", "cu", "cuan", "cui",
                                     "cun", "cuo", "da", "dai", "dan", "dang", "dao", "de", "deng", "di", "dian", "diao", "die", "ding", "diu", "dong", "dou", "du", "duan",
                                     "dui", "dun", "duo", "e", "en", "er", "fa", "fan", "fang", "fei", "fen", "feng", "fo", "fou", "fu", "ga", "gai", "gan", "gang", "gao",
                                     "ge", "gei", "gen", "geng", "gong", "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo", "ha", "hai", "han", "hang",
                                     "hao", "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun", "huo", "ji", "jia", "jian",
                                     "jiang", "jiao", "jie", "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun", "ka", "kai", "kan", "kang", "kao", "ke", "ken",
                                     "keng", "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui", "kun", "kuo", "la", "lai", "lan", "lang", "lao", "le", "lei",
                                     "leng", "li", "lia", "lian", "liang", "liao", "lie", "lin", "ling", "liu", "long", "lou", "lu", "lv", "luan", "lue", "lun", "luo",
                                     "ma", "mai", "man", "mang", "mao", "me", "mei", "men", "meng", "mi", "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu",
                                     "na", "nai", "nan", "nang", "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang", "niao", "nie", "nin", "ning", "niu", "nong",
                                     "nu", "nv", "nuan", "nue", "nuo", "o", "ou", "pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng", "pi", "pian", "piao", "pie",
                                     "pin", "ping", "po", "pu", "qi", "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu", "qu", "quan", "que", "qun",
                                     "ran", "rang", "rao", "re", "ren", "reng", "ri", "rong", "rou", "ru", "ruan", "rui", "run", "ruo", "sa", "sai", "san", "sang",
                                     "sao", "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao", "she", "shen", "sheng", "shi", "shou", "shu", "shua",
                                     "shuai", "shuan", "shuang", "shui", "shun", "shuo", "si", "song", "sou", "su", "suan", "sui", "sun", "suo", "ta", "tai",
                                     "tan", "tang", "tao", "te", "teng", "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu", "tuan", "tui", "tun", "tuo",
                                     "wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu", "xi", "xia", "xian", "xiang", "xiao", "xie", "xin", "xing",
                                     "xiong", "xiu", "xu", "xuan", "xue", "xun", "ya", "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong", "you",
                                     "yu", "yuan", "yue", "yun", "za", "zai", "zan", "zang", "zao", "ze", "zei", "zen", "zeng", "zha", "zhai", "zhan", "zhang",
                                     "zhao", "zhe", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang", "zhui", "zhun", "zhuo",
                                     "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo"};

    try
    {
        // 循环处理字节数组
        const int length = dest_chinese.length();
        for (int j = 0, chrasc = 0; j < length;)
        {
            // 非汉字处理
            if (dest_chinese.at(j) >= 0 && dest_chinese.at(j) < 128)
            {
                out_py += dest_chinese.at(j);
                // 偏移下标
                j++;
                continue;
            }

            // 汉字处理
            chrasc = dest_chinese.at(j) * 256 + dest_chinese.at(j + 1) + 256;
            if (chrasc > 0 && chrasc < 160)
            {
                // 非汉字
                out_py += dest_chinese.at(j);
                // 偏移下标
                j++;
            }
            else
            {
                // 汉字
                for (int i = (sizeof(spell_value) / sizeof(spell_value[0]) - 1); i >= 0; --i)
                {
                    // 查找字典
                    if (spell_value[i] <= chrasc)
                    {
                        out_py += spell_dict[i];
                        break;
                    }
                }
                // 偏移下标 （汉字双字节）
                j += 2;
            }
        } // for end
    }
    catch (exception _e)
    {
        std::cout << _e.what() << std::endl;
        return -1;
    }
    return 0;
}

void searchFilesIgnoreSearchDepth(const char *path, const char *exd)
{
    string file(path);
    string suffix(exd);
    cout << "start search without searchDepth" << endl;
    searchIgnoreSearchDepth(file, suffix);
    cout << "end search without searchDepth" << endl;
}

void searchIgnoreSearchDepth(string path, string exd)
{
    //cout << "getFiles()" << path<< endl;
    //文件句柄
    long hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;
    string pathName, exdName;

    if (0 != strcmp(exd.c_str(), ""))
    {
        exdName = "\\*." + exd;
    }
    else
    {
        exdName = "\\*";
    }

    if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
    {
        do
        {
            //cout << fileinfo.name << endl;

            //如果是文件夹中仍有文件夹,加入列表后迭代
            //如果不是,加入列表
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {

                    string name(fileinfo.name);
                    string _name;
                    string _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    ChineseConvertPy(name, _name);
                    transform(_name.begin(), _name.end(), _name.begin(), ::toupper);
                    saveResult(_path, getAscIISum(_name));

                    bool Ignore = isIgnore(path);
                    if (!Ignore)
                    {
                        searchIgnoreSearchDepth(_path, exd);
                        //cout << isResultReady << endl;
                    }
                }
            }
            else
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    string name(fileinfo.name);
                    string _name;
                    string _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    ChineseConvertPy(name, _name);
                    transform(_name.begin(), _name.end(), _name.begin(), ::toupper);
                    saveResult(_path, getAscIISum(_name));
                }
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

void searchFiles(const char *path, const char *exd)
{
    string file(path);
    string suffix(exd);
    cout << "start Search" << endl;
    search(file, exd);
    cout << "end Search" << endl;
#ifdef TEST
    cout << "0-100 : " << results0_100.length() << endl;
    cout << "100-200 : " << results100_200.length() << endl;
    cout << "200-300 : " << results200_300.length() << endl;
    cout << "300-400 : " << results300_400.length() << endl;
    cout << "400-500 : " << results400_500.length() << endl;
    cout << "600-600 : " << results500_600.length() << endl;
    cout << "600-700 : " << results600_700.length() << endl;
    cout << "700-800 : " << results700_800.length() << endl;
    cout << "800-900 : " << results800_900.length() << endl;
    cout << "900-1000 : " << results900_1000.length() << endl;
    cout << "1000-1100 : " << results1000_1100.length() << endl;
    cout << "1100-1200 : " << results1100_1200.length() << endl;
    cout << "1200-1300 : " << results1200_1300.length() << endl;
    cout << "1300-1400 : " << results1300_1400.length() << endl;
    cout << "1400-1500 : " << results1400_1500.length() << endl;
    getchar();
#endif
}

void addIgnorePath(const char *path)
{
    string str(path);
    transform(str.begin(), str.end(), str.begin(), ::tolower);
    cout << "adding ignorePath:" << str << endl;
    ignorePathVector.push_back(str);
}

void setSearchDepth(int i)
{
    searchDepth = i;
}

int count(string path, string pattern)
{
    int begin = -1;
    int count = 0;
    while ((begin = path.find(pattern, begin + 1)) != string::npos)
    {
        count++;
        begin = begin + pattern.length();
    }
    return count;
}

bool isSearchDepthOut(string path)
{
    int num = count(path, "\\");
    if (num > searchDepth - 2)
    {
        return true;
    }
    return false;
}

bool isIgnore(string path)
{
    if (path.find("$") != string::npos)
    {
        return true;
    }
    transform(path.begin(), path.end(), path.begin(), ::tolower);
    int size = ignorePathVector.size();
    for (int i = 0; i < size; i++)
    {
        if (path.find(ignorePathVector[i]) != string::npos)
        {
            return true;
        }
    }
    return false;
}

void search(string path, string exd)
{
    //cout << "getFiles()" << path<< endl;
    //文件句柄
    long hFile = 0;
    //文件信息
    struct _finddata_t fileinfo;
    string pathName, exdName;

    if (0 != strcmp(exd.c_str(), ""))
    {
        exdName = "\\*." + exd;
    }
    else
    {
        exdName = "\\*";
    }

    if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
    {
        do
        {
            //cout << fileinfo.name << endl;

            //如果是文件夹中仍有文件夹,加入列表后迭代
            //如果不是,加入列表
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    string name(fileinfo.name);
                    string _name;
                    string _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    ChineseConvertPy(name, _name);
                    transform(_name.begin(), _name.end(), _name.begin(), ::toupper);
                    saveResult(_path, getAscIISum(_name));

                    bool SearchDepthOut = isSearchDepthOut(path);
                    bool Ignore = isIgnore(path);
                    bool result = !Ignore && !SearchDepthOut;
                    if (result)
                    {
                        search(_path, exd);
                    }
                }
            }
            else
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                {
                    string name(fileinfo.name);
                    string _name;
                    string _path = pathName.assign(path).append("\\").append(fileinfo.name);
                    ChineseConvertPy(name, _name);
                    transform(_name.begin(), _name.end(), _name.begin(), ::toupper);
                    saveResult(_path, getAscIISum(_name));
                }
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

#ifndef TEST
int main(int argc, char *argv[])
{
    char searchPath[260];
    char searchDepth[50];
    char ignorePath[3000];
    char output[260];
    char isIgnoreSearchDepth[2];
    char output0_100[260];
    char output100_200[260];
    char output200_300[260];
    char output300_400[260];
    char output400_500[260];
    char output500_600[260];
    char output600_700[260];
    char output700_800[260];
    char output800_900[260];
    char output900_1000[260];
    char output1000_1100[260];
    char output1100_1200[260];
    char output1200_1300[260];
    char output1300_1400[260];
    char output1400_1500[260];
    char output1500_1600[260];
    char output1600_1700[260];
    char output1700_1800[260];
    char output1800_1900[260];
    char output1900_2000[260];
    char output2000_2100[260];
    char output2100_2200[260];
    char output2200_2300[260];
    char output2300_2400[260];
    char output2400_2500[260];
    char output2500_[260];

    ofstream outfile;
    if (argc == 6)
    {
        strcpy(searchPath, argv[1]);
        strcpy(searchDepth, argv[2]);
        setSearchDepth(atoi(searchDepth));
        strcpy(ignorePath, argv[3]);
        strcpy(output, argv[4]);
        strcpy(isIgnoreSearchDepth, argv[5]);
        strcpy(output0_100, output);
        strcat(output0_100, "\\list0-100.txt");
        strcpy(output100_200, output);
        strcat(output100_200, "\\list100-200.txt");
        strcpy(output200_300, output);
        strcat(output200_300, "\\list200-300.txt");
        strcpy(output300_400, output);
        strcat(output300_400, "\\list300-400.txt");
        strcpy(output400_500, output);
        strcat(output400_500, "\\list400-500.txt");
        strcpy(output500_600, output);
        strcat(output500_600, "\\list500-600.txt");
        strcpy(output600_700, output);
        strcat(output600_700, "\\list600-700.txt");
        strcpy(output700_800, output);
        strcat(output700_800, "\\list700-800.txt");
        strcpy(output800_900, output);
        strcat(output800_900, "\\list800-900.txt");
        strcpy(output900_1000, output);
        strcat(output900_1000, "\\list900-1000.txt");
        strcpy(output1000_1100, output);
        strcat(output1000_1100, "\\list1000-1100.txt");
        strcpy(output1100_1200, output);
        strcat(output1100_1200, "\\list1100-1200.txt");
        strcpy(output1200_1300, output);
        strcat(output1200_1300, "\\list1200-1300.txt");
        strcpy(output1300_1400, output);
        strcat(output1300_1400, "\\list1300-1400.txt");
        strcpy(output1400_1500, output);
        strcat(output1400_1500, "\\list1400-1500.txt");
        strcpy(output1500_1600, output);
        strcat(output1500_1600, "\\list1500-1600.txt");
        strcpy(output1600_1700, output);
        strcat(output1600_1700, "\\list1600-1700.txt");
        strcpy(output1700_1800, output);
        strcat(output1700_1800, "\\list1700-1800.txt");
        strcpy(output1800_1900, output);
        strcat(output1800_1900, "\\list1800-1900.txt");
        strcpy(output1900_2000, output);
        strcat(output2000_2100, "\\list1900-2000.txt");
        strcpy(output2000_2100, output);
        strcat(output2000_2100, "\\list2000-2100.txt");
        strcpy(output2100_2200, output);
        strcat(output2100_2200, "\\list2100-2200.txt");
        strcpy(output2200_2300, output);
        strcat(output2200_2300, "\\list2200-2300.txt");
        strcpy(output2300_2400, output);
        strcat(output2300_2400, "\\list2300-2400.txt");
        strcpy(output2400_2500, output);
        strcat(output2400_2500, "\\list2400-2500.txt");
        strcpy(output2500_, output);
        strcat(output2500_, "\\list2500-.txt");
        cout << "searchPath:" << searchPath << endl;
        cout << "searchDepth:" << searchDepth << endl;
        cout << "ignorePath:" << ignorePath << endl;
        cout << "output:" << output << endl;
        cout << "isIgnoreSearchDepth:" << isIgnoreSearchDepth << endl;
        char *p = NULL;
        char *_ignorepath = ignorePath;
        p = strtok(_ignorepath, ",");
        if (p != NULL)
        {
            addIgnorePath(p);
        }

        while (p != NULL)
        {
            p = strtok(NULL, ",");
            if (p != NULL)
            {
                addIgnorePath(p);
            }
        }

        cout << "ignorePathVector Size:" << ignorePathVector.size() << endl;
        if (atoi(isIgnoreSearchDepth) == 1)
        {
            cout << "ignore searchDepth!!!" << endl;
            searchFilesIgnoreSearchDepth(searchPath, "*");
            //将结果写入文件
            outfile.open(output0_100, ios::app);
            outfile << results0_100 << endl;
            outfile.close();
            outfile.open(output100_200, ios::app);
            outfile << results100_200 << endl;
            outfile.close();
            outfile.open(output200_300, ios::app);
            outfile << results200_300 << endl;
            outfile.close();
            outfile.open(output300_400, ios::app);
            outfile << results300_400 << endl;
            outfile.close();
            outfile.open(output400_500, ios::app);
            outfile << results400_500 << endl;
            outfile.close();
            outfile.open(output500_600, ios::app);
            outfile << results500_600 << endl;
            outfile.close();
            outfile.open(output600_700, ios::app);
            outfile << results600_700 << endl;
            outfile.close();
            outfile.open(output700_800, ios::app);
            outfile << results700_800 << endl;
            outfile.close();
            outfile.open(output800_900, ios::app);
            outfile << results800_900 << endl;
            outfile.close();
            outfile.open(output900_1000, ios::app);
            outfile << results900_1000 << endl;
            outfile.close();
            outfile.open(output1000_1100, ios::app);
            outfile << results1000_1100 << endl;
            outfile.close();
            outfile.open(output1100_1200, ios::app);
            outfile << results1100_1200 << endl;
            outfile.close();
            outfile.open(output1200_1300, ios::app);
            outfile << results1200_1300 << endl;
            outfile.close();
            outfile.open(output1300_1400, ios::app);
            outfile << results1300_1400 << endl;
            outfile.close();
            outfile.open(output1400_1500, ios::app);
            outfile << results1400_1500 << endl;
            outfile.close();
            outfile.open(output1500_1600, ios::app);
            outfile << results1500_1600 << endl;
            outfile.close();
            outfile.open(output1600_1700, ios::app);
            outfile << results1600_1700 << endl;
            outfile.close();
            outfile.open(output1700_1800, ios::app);
            outfile << results1700_1800 << endl;
            outfile.close();
            outfile.open(output1800_1900, ios::app);
            outfile << results1800_1900 << endl;
            outfile.close();
            outfile.open(output1900_2000, ios::app);
            outfile << results1900_2000 << endl;
            outfile.close();
            outfile.open(output2000_2100, ios::app);
            outfile << results2000_2100 << endl;
            outfile.close();
            outfile.open(output2100_2200, ios::app);
            outfile << results2100_2200 << endl;
            outfile.close();
            outfile.open(output2200_2300, ios::app);
            outfile << results2200_2300 << endl;
            outfile.close();
            outfile.open(output2300_2400, ios::app);
            outfile << results2300_2400 << endl;
            outfile.close();
            outfile.open(output2400_2500, ios::app);
            outfile << results2400_2500 << endl;
            outfile.close();
            outfile.open(output2500_, ios::app);
            outfile << results2500_ << endl;
            outfile.close();
        }
        else
        {
            cout << "normal search" << endl;
            searchFiles(searchPath, "*");
            //将结果写入文件
            outfile.open(output0_100, ios::app);
            outfile << results0_100 << endl;
            outfile.close();
            outfile.open(output100_200, ios::app);
            outfile << results100_200 << endl;
            outfile.close();
            outfile.open(output200_300, ios::app);
            outfile << results200_300 << endl;
            outfile.close();
            outfile.open(output300_400, ios::app);
            outfile << results300_400 << endl;
            outfile.close();
            outfile.open(output400_500, ios::app);
            outfile << results400_500 << endl;
            outfile.close();
            outfile.open(output500_600, ios::app);
            outfile << results500_600 << endl;
            outfile.close();
            outfile.open(output600_700, ios::app);
            outfile << results600_700 << endl;
            outfile.close();
            outfile.open(output700_800, ios::app);
            outfile << results700_800 << endl;
            outfile.close();
            outfile.open(output800_900, ios::app);
            outfile << results800_900 << endl;
            outfile.close();
            outfile.open(output900_1000, ios::app);
            outfile << results900_1000 << endl;
            outfile.close();
            outfile.open(output1000_1100, ios::app);
            outfile << results1000_1100 << endl;
            outfile.close();
            outfile.open(output1100_1200, ios::app);
            outfile << results1100_1200 << endl;
            outfile.close();
            outfile.open(output1200_1300, ios::app);
            outfile << results1200_1300 << endl;
            outfile.close();
            outfile.open(output1300_1400, ios::app);
            outfile << results1300_1400 << endl;
            outfile.close();
            outfile.open(output1400_1500, ios::app);
            outfile << results1400_1500 << endl;
            outfile.close();
            outfile.open(output1500_1600, ios::app);
            outfile << results1500_1600 << endl;
            outfile.close();
            outfile.open(output1600_1700, ios::app);
            outfile << results1600_1700 << endl;
            outfile.close();
            outfile.open(output1700_1800, ios::app);
            outfile << results1700_1800 << endl;
            outfile.close();
            outfile.open(output1800_1900, ios::app);
            outfile << results1800_1900 << endl;
            outfile.close();
            outfile.open(output1900_2000, ios::app);
            outfile << results1900_2000 << endl;
            outfile.close();
            outfile.open(output2000_2100, ios::app);
            outfile << results2000_2100 << endl;
            outfile.close();
            outfile.open(output2100_2200, ios::app);
            outfile << results2100_2200 << endl;
            outfile.close();
            outfile.open(output2200_2300, ios::app);
            outfile << results2200_2300 << endl;
            outfile.close();
            outfile.open(output2300_2400, ios::app);
            outfile << results2300_2400 << endl;
            outfile.close();
            outfile.open(output2400_2500, ios::app);
            outfile << results2400_2500 << endl;
            outfile.close();
            outfile.open(output2500_, ios::app);
            outfile << results2500_ << endl;
            outfile.close();
        }

        return 0;
    }
    else
    {
        cout << "args error" << endl;
    }
}
#else
int main()
{
    setSearchDepth(6);
    searchFiles("D:", "*");
}

#endif
