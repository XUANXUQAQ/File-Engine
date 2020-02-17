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
string resultsA;
string resultsB;
string resultsC;
string resultsD;
string resultsE;
string resultsF;
string resultsG;
string resultsH;
string resultsI;
string resultsJ;
string resultsK;
string resultsL;
string resultsM;
string resultsN;
string resultsO;
string resultsP;
string resultsQ;
string resultsR;
string resultsS;
string resultsT;
string resultsU;
string resultsV;
string resultsW;
string resultsX;
string resultsY;
string resultsZ;
string resultsNum;
string resultsPercentSign;
string resultsUnderline;
string resultsUnique;

void searchFiles(const char *path, const char *exd);
void addIgnorePath(const char *path);
int count(string path, string pattern);
bool isSearchDepthOut(string path);
bool isIgnore(string path);
void search(string path, string exd);
void searchIgnoreSearchDepth(string path, string exd);
void clearResults();

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

char toUpper(char i)
{
    if ((i >= 65) && (i <= 90))
    {
        return i;
    }
    else if ((i >= 97) && (i <= 122))
    {
        return i - 32;
    }
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
                    transform(_name.begin(), _name.end(), _name.begin(), ::toUpper);
                    if (_name.find("A") != string::npos)
                    {
                        resultsA.append(_path);
                        resultsA.append("\n");
                    }
                    if (_name.find("B") != string::npos)
                    {
                        resultsB.append(_path);
                        resultsB.append("\n");
                    }
                    if (_name.find("C") != string::npos)
                    {
                        resultsC.append(_path);
                        resultsC.append("\n");
                    }
                    if (_name.find("D") != string::npos)
                    {
                        resultsD.append(_path);
                        resultsD.append("\n");
                    }
                    if (_name.find("E") != string::npos)
                    {
                        resultsE.append(_path);
                        resultsE.append("\n");
                    }
                    if (_name.find("F") != string::npos)
                    {
                        resultsF.append(_path);
                        resultsF.append("\n");
                    }
                    if (_name.find("G") != string::npos)
                    {
                        resultsG.append(_path);
                        resultsG.append("\n");
                    }
                    if (_name.find("H") != string::npos)
                    {
                        resultsH.append(_path);
                        resultsH.append("\n");
                    }
                    if (_name.find("I") != string::npos)
                    {
                        resultsI.append(_path);
                        resultsI.append("\n");
                    }
                    if (_name.find("J") != string::npos)
                    {
                        resultsJ.append(_path);
                        resultsJ.append("\n");
                    }
                    if (_name.find("L") != string::npos)
                    {
                        resultsL.append(_path);
                        resultsL.append("\n");
                    }
                    if (_name.find("M") != string::npos)
                    {
                        resultsM.append(_path);
                        resultsM.append("\n");
                    }
                    if (_name.find("N") != string::npos)
                    {
                        resultsN.append(_path);
                        resultsN.append("\n");
                    }
                    if (_name.find("O") != string::npos)
                    {
                        resultsO.append(_path);
                        resultsO.append("\n");
                    }
                    if (_name.find("P") != string::npos)
                    {
                        resultsP.append(_path);
                        resultsP.append("\n");
                    }
                    if (_name.find("Q") != string::npos)
                    {
                        resultsQ.append(_path);
                        resultsQ.append("\n");
                    }
                    if (_name.find("R") != string::npos)
                    {
                        resultsR.append(_path);
                        resultsR.append("\n");
                    }
                    if (_name.find("S") != string::npos)
                    {
                        resultsS.append(_path);
                        resultsS.append("\n");
                    }
                    if (_name.find("T") != string::npos)
                    {
                        resultsT.append(_path);
                        resultsT.append("\n");
                    }
                    if (_name.find("U") != string::npos)
                    {
                        resultsU.append(_path);
                        resultsU.append("\n");
                    }
                    if (_name.find("V") != string::npos)
                    {
                        resultsV.append(_path);
                        resultsV.append("\n");
                    }
                    if (_name.find("W") != string::npos)
                    {
                        resultsW.append(_path);
                        resultsW.append("\n");
                    }
                    if (_name.find("X") != string::npos)
                    {
                        resultsX.append(_path);
                        resultsX.append("\n");
                    }
                    if (_name.find("Y") != string::npos)
                    {
                        resultsY.append(_path);
                        resultsY.append("\n");
                    }
                    if (_name.find("Z") != string::npos)
                    {
                        resultsZ.append(_path);
                        resultsZ.append("\n");
                    }
                    if (_name.find("%") != string::npos)
                    {
                        resultsPercentSign.append(_path);
                        resultsPercentSign.append("\n");
                    }
                    if (_name.find("_") != string::npos)
                    {
                        resultsUnderline.append(_path);
                        resultsUnderline.append("\n");
                    }
                    if (_name.find("Num") != string::npos)
                    {
                        resultsNum.append(_path);
                        resultsNum.append("\n");
                    }
                    else
                    {
                        resultsUnique.append(_path);
                        resultsUnique.append("\n");
                    }

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
                    transform(_name.begin(), _name.end(), _name.begin(), ::toUpper);
                    if (_name.find("A") != string::npos)
                    {
                        resultsA.append(_path);
                        resultsA.append("\n");
                    }
                    if (_name.find("B") != string::npos)
                    {
                        resultsB.append(_path);
                        resultsB.append("\n");
                    }
                    if (_name.find("C") != string::npos)
                    {
                        resultsC.append(_path);
                        resultsC.append("\n");
                    }
                    if (_name.find("D") != string::npos)
                    {
                        resultsD.append(_path);
                        resultsD.append("\n");
                    }
                    if (_name.find("E") != string::npos)
                    {
                        resultsE.append(_path);
                        resultsE.append("\n");
                    }
                    if (_name.find("F") != string::npos)
                    {
                        resultsF.append(_path);
                        resultsF.append("\n");
                    }
                    if (_name.find("G") != string::npos)
                    {
                        resultsG.append(_path);
                        resultsG.append("\n");
                    }
                    if (_name.find("H") != string::npos)
                    {
                        resultsH.append(_path);
                        resultsH.append("\n");
                    }
                    if (_name.find("I") != string::npos)
                    {
                        resultsI.append(_path);
                        resultsI.append("\n");
                    }
                    if (_name.find("J") != string::npos)
                    {
                        resultsJ.append(_path);
                        resultsJ.append("\n");
                    }
                    if (_name.find("L") != string::npos)
                    {
                        resultsL.append(_path);
                        resultsL.append("\n");
                    }
                    if (_name.find("M") != string::npos)
                    {
                        resultsM.append(_path);
                        resultsM.append("\n");
                    }
                    if (_name.find("N") != string::npos)
                    {
                        resultsN.append(_path);
                        resultsN.append("\n");
                    }
                    if (_name.find("O") != string::npos)
                    {
                        resultsO.append(_path);
                        resultsO.append("\n");
                    }
                    if (_name.find("P") != string::npos)
                    {
                        resultsP.append(_path);
                        resultsP.append("\n");
                    }
                    if (_name.find("Q") != string::npos)
                    {
                        resultsQ.append(_path);
                        resultsQ.append("\n");
                    }
                    if (_name.find("R") != string::npos)
                    {
                        resultsR.append(_path);
                        resultsR.append("\n");
                    }
                    if (_name.find("S") != string::npos)
                    {
                        resultsS.append(_path);
                        resultsS.append("\n");
                    }
                    if (_name.find("T") != string::npos)
                    {
                        resultsT.append(_path);
                        resultsT.append("\n");
                    }
                    if (_name.find("U") != string::npos)
                    {
                        resultsU.append(_path);
                        resultsU.append("\n");
                    }
                    if (_name.find("V") != string::npos)
                    {
                        resultsV.append(_path);
                        resultsV.append("\n");
                    }
                    if (_name.find("W") != string::npos)
                    {
                        resultsW.append(_path);
                        resultsW.append("\n");
                    }
                    if (_name.find("X") != string::npos)
                    {
                        resultsX.append(_path);
                        resultsX.append("\n");
                    }
                    if (_name.find("Y") != string::npos)
                    {
                        resultsY.append(_path);
                        resultsY.append("\n");
                    }
                    if (_name.find("Z") != string::npos)
                    {
                        resultsZ.append(_path);
                        resultsZ.append("\n");
                    }
                    if (_name.find("%") != string::npos)
                    {
                        resultsPercentSign.append(_path);
                        resultsPercentSign.append("\n");
                    }
                    if (_name.find("_") != string::npos)
                    {
                        resultsUnderline.append(_path);
                        resultsUnderline.append("\n");
                    }
                    if (_name.find("Num") != string::npos)
                    {
                        resultsNum.append(_path);
                        resultsNum.append("\n");
                    }
                    else
                    {
                        resultsUnique.append(_path);
                        resultsUnique.append("\n");
                    }
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
    //cout << isResultReady << endl;
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
                    transform(_name.begin(), _name.end(), _name.begin(), ::toUpper);
                    if (_name.find("A") != string::npos)
                    {
                        resultsA.append(_path);
                        resultsA.append("\n");
                    }
                    if (_name.find("B") != string::npos)
                    {
                        resultsB.append(_path);
                        resultsB.append("\n");
                    }
                    if (_name.find("C") != string::npos)
                    {
                        resultsC.append(_path);
                        resultsC.append("\n");
                    }
                    if (_name.find("D") != string::npos)
                    {
                        resultsD.append(_path);
                        resultsD.append("\n");
                    }
                    if (_name.find("E") != string::npos)
                    {
                        resultsE.append(_path);
                        resultsE.append("\n");
                    }
                    if (_name.find("F") != string::npos)
                    {
                        resultsF.append(_path);
                        resultsF.append("\n");
                    }
                    if (_name.find("G") != string::npos)
                    {
                        resultsG.append(_path);
                        resultsG.append("\n");
                    }
                    if (_name.find("H") != string::npos)
                    {
                        resultsH.append(_path);
                        resultsH.append("\n");
                    }
                    if (_name.find("I") != string::npos)
                    {
                        resultsI.append(_path);
                        resultsI.append("\n");
                    }
                    if (_name.find("J") != string::npos)
                    {
                        resultsJ.append(_path);
                        resultsJ.append("\n");
                    }
                    if (_name.find("L") != string::npos)
                    {
                        resultsL.append(_path);
                        resultsL.append("\n");
                    }
                    if (_name.find("M") != string::npos)
                    {
                        resultsM.append(_path);
                        resultsM.append("\n");
                    }
                    if (_name.find("N") != string::npos)
                    {
                        resultsN.append(_path);
                        resultsN.append("\n");
                    }
                    if (_name.find("O") != string::npos)
                    {
                        resultsO.append(_path);
                        resultsO.append("\n");
                    }
                    if (_name.find("P") != string::npos)
                    {
                        resultsP.append(_path);
                        resultsP.append("\n");
                    }
                    if (_name.find("Q") != string::npos)
                    {
                        resultsQ.append(_path);
                        resultsQ.append("\n");
                    }
                    if (_name.find("R") != string::npos)
                    {
                        resultsR.append(_path);
                        resultsR.append("\n");
                    }
                    if (_name.find("S") != string::npos)
                    {
                        resultsS.append(_path);
                        resultsS.append("\n");
                    }
                    if (_name.find("T") != string::npos)
                    {
                        resultsT.append(_path);
                        resultsT.append("\n");
                    }
                    if (_name.find("U") != string::npos)
                    {
                        resultsU.append(_path);
                        resultsU.append("\n");
                    }
                    if (_name.find("V") != string::npos)
                    {
                        resultsV.append(_path);
                        resultsV.append("\n");
                    }
                    if (_name.find("W") != string::npos)
                    {
                        resultsW.append(_path);
                        resultsW.append("\n");
                    }
                    if (_name.find("X") != string::npos)
                    {
                        resultsX.append(_path);
                        resultsX.append("\n");
                    }
                    if (_name.find("Y") != string::npos)
                    {
                        resultsY.append(_path);
                        resultsY.append("\n");
                    }
                    if (_name.find("Z") != string::npos)
                    {
                        resultsZ.append(_path);
                        resultsZ.append("\n");
                    }
                    if (_name.find("%") != string::npos)
                    {
                        resultsPercentSign.append(_path);
                        resultsPercentSign.append("\n");
                    }
                    if (_name.find("_") != string::npos)
                    {
                        resultsUnderline.append(_path);
                        resultsUnderline.append("\n");
                    }
                    if (_name.find("Num") != string::npos)
                    {
                        resultsNum.append(_path);
                        resultsNum.append("\n");
                    }
                    else
                    {
                        resultsUnique.append(_path);
                        resultsUnique.append("\n");
                    }

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
                    transform(_name.begin(), _name.end(), _name.begin(), ::toUpper);
                    if (_name.find("A") != string::npos)
                    {
                        resultsA.append(_path);
                        resultsA.append("\n");
                    }
                    if (_name.find("B") != string::npos)
                    {
                        resultsB.append(_path);
                        resultsB.append("\n");
                    }
                    if (_name.find("C") != string::npos)
                    {
                        resultsC.append(_path);
                        resultsC.append("\n");
                    }
                    if (_name.find("D") != string::npos)
                    {
                        resultsD.append(_path);
                        resultsD.append("\n");
                    }
                    if (_name.find("E") != string::npos)
                    {
                        resultsE.append(_path);
                        resultsE.append("\n");
                    }
                    if (_name.find("F") != string::npos)
                    {
                        resultsF.append(_path);
                        resultsF.append("\n");
                    }
                    if (_name.find("G") != string::npos)
                    {
                        resultsG.append(_path);
                        resultsG.append("\n");
                    }
                    if (_name.find("H") != string::npos)
                    {
                        resultsH.append(_path);
                        resultsH.append("\n");
                    }
                    if (_name.find("I") != string::npos)
                    {
                        resultsI.append(_path);
                        resultsI.append("\n");
                    }
                    if (_name.find("J") != string::npos)
                    {
                        resultsJ.append(_path);
                        resultsJ.append("\n");
                    }
                    if (_name.find("L") != string::npos)
                    {
                        resultsL.append(_path);
                        resultsL.append("\n");
                    }
                    if (_name.find("M") != string::npos)
                    {
                        resultsM.append(_path);
                        resultsM.append("\n");
                    }
                    if (_name.find("N") != string::npos)
                    {
                        resultsN.append(_path);
                        resultsN.append("\n");
                    }
                    if (_name.find("O") != string::npos)
                    {
                        resultsO.append(_path);
                        resultsO.append("\n");
                    }
                    if (_name.find("P") != string::npos)
                    {
                        resultsP.append(_path);
                        resultsP.append("\n");
                    }
                    if (_name.find("Q") != string::npos)
                    {
                        resultsQ.append(_path);
                        resultsQ.append("\n");
                    }
                    if (_name.find("R") != string::npos)
                    {
                        resultsR.append(_path);
                        resultsR.append("\n");
                    }
                    if (_name.find("S") != string::npos)
                    {
                        resultsS.append(_path);
                        resultsS.append("\n");
                    }
                    if (_name.find("T") != string::npos)
                    {
                        resultsT.append(_path);
                        resultsT.append("\n");
                    }
                    if (_name.find("U") != string::npos)
                    {
                        resultsU.append(_path);
                        resultsU.append("\n");
                    }
                    if (_name.find("V") != string::npos)
                    {
                        resultsV.append(_path);
                        resultsV.append("\n");
                    }
                    if (_name.find("W") != string::npos)
                    {
                        resultsW.append(_path);
                        resultsW.append("\n");
                    }
                    if (_name.find("X") != string::npos)
                    {
                        resultsX.append(_path);
                        resultsX.append("\n");
                    }
                    if (_name.find("Y") != string::npos)
                    {
                        resultsY.append(_path);
                        resultsY.append("\n");
                    }
                    if (_name.find("Z") != string::npos)
                    {
                        resultsZ.append(_path);
                        resultsZ.append("\n");
                    }
                    if (_name.find("%") != string::npos)
                    {
                        resultsPercentSign.append(_path);
                        resultsPercentSign.append("\n");
                    }
                    if (_name.find("_") != string::npos)
                    {
                        resultsUnderline.append(_path);
                        resultsUnderline.append("\n");
                    }
                    if (_name.find("Num") != string::npos)
                    {
                        resultsNum.append(_path);
                        resultsNum.append("\n");
                    }
                    else
                    {
                        resultsUnique.append(_path);
                        resultsUnique.append("\n");
                    }
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
    char outputA[260];
    char outputB[260];
    char outputC[260];
    char outputD[260];
    char outputE[260];
    char outputF[260];
    char outputG[260];
    char outputH[260];
    char outputI[260];
    char outputJ[260];
    char outputK[260];
    char outputL[260];
    char outputM[260];
    char outputN[260];
    char outputO[260];
    char outputP[260];
    char outputQ[260];
    char outputR[260];
    char outputS[260];
    char outputT[260];
    char outputU[260];
    char outputV[260];
    char outputW[260];
    char outputX[260];
    char outputY[260];
    char outputZ[260];
    char outputNum[260];
    char outputPercentSign[260];
    char outputUnderline[260];
    char outputUnique[260];

    ofstream outfile;
    if (argc == 6)
    {
        strcpy(searchPath, argv[1]);
        strcpy(searchDepth, argv[2]);
        setSearchDepth(atoi(searchDepth));
        strcpy(ignorePath, argv[3]);
        strcpy(output, argv[4]);
        strcpy(isIgnoreSearchDepth, argv[5]);
        strcpy(outputA, output);
        strcat(outputA, "\\listA.txt");
        strcpy(outputB, output);
        strcat(outputB, "\\listB.txt");
        strcpy(outputC, output);
        strcat(outputC, "\\listC.txt");
        strcpy(outputD, output);
        strcat(outputD, "\\listD.txt");
        strcpy(outputE, output);
        strcat(outputE, "\\listE.txt");
        strcpy(outputF, output);
        strcat(outputF, "\\listF.txt");
        strcpy(outputG, output);
        strcat(outputG, "\\listG.txt");
        strcpy(outputH, output);
        strcat(outputH, "\\listH.txt");
        strcpy(outputI, output);
        strcat(outputI, "\\listI.txt");
        strcpy(outputJ, output);
        strcat(outputJ, "\\listJ.txt");
        strcpy(outputK, output);
        strcat(outputK, "\\listK.txt");
        strcpy(outputL, output);
        strcat(outputL, "\\listL.txt");
        strcpy(outputM, output);
        strcat(outputM, "\\listM.txt");
        strcpy(outputN, output);
        strcat(outputN, "\\listN.txt");
        strcpy(outputO, output);
        strcat(outputO, "\\listO.txt");
        strcpy(outputP, output);
        strcat(outputP, "\\listP.txt");
        strcpy(outputQ, output);
        strcat(outputQ, "\\listQ.txt");
        strcpy(outputR, output);
        strcat(outputR, "\\listR.txt");
        strcpy(outputS, output);
        strcat(outputS, "\\listS.txt");
        strcpy(outputT, output);
        strcat(outputT, "\\listT.txt");
        strcpy(outputU, output);
        strcat(outputU, "\\listU.txt");
        strcpy(outputV, output);
        strcat(outputV, "\\listV.txt");
        strcpy(outputW, output);
        strcat(outputW, "\\listW.txt");
        strcpy(outputX, output);
        strcat(outputX, "\\listX.txt");
        strcpy(outputY, output);
        strcat(outputY, "\\listY.txt");
        strcpy(outputZ, output);
        strcat(outputZ, "\\listZ.txt");
        strcpy(outputNum, output);
        strcat(outputNum, "\\listNum.txt");
        strcpy(outputPercentSign, output);
        strcat(outputPercentSign, "\\listPercentSign.txt");
        strcpy(outputUnderline, output);
        strcat(outputUnderline, "\\listUnderline.txt");
        strcpy(outputUnique, output);
        strcat(outputUnique, "\\listUnique.txt");
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
            outfile.open(outputA, ios::app);
            outfile << resultsA << endl;
            outfile.close();
            outfile.open(outputB, ios::app);
            outfile << resultsB << endl;
            outfile.close();
            outfile.open(outputC, ios::app);
            outfile << resultsC << endl;
            outfile.close();
            outfile.open(outputD, ios::app);
            outfile << resultsD << endl;
            outfile.close();
            outfile.open(outputE, ios::app);
            outfile << resultsE << endl;
            outfile.close();
            outfile.open(outputF, ios::app);
            outfile << resultsF << endl;
            outfile.close();
            outfile.open(outputG, ios::app);
            outfile << resultsG << endl;
            outfile.close();
            outfile.open(outputH, ios::app);
            outfile << resultsH << endl;
            outfile.close();
            outfile.open(outputI, ios::app);
            outfile << resultsI << endl;
            outfile.close();
            outfile.open(outputJ, ios::app);
            outfile << resultsJ << endl;
            outfile.close();
            outfile.open(outputK, ios::app);
            outfile << resultsK << endl;
            outfile.close();
            outfile.open(outputL, ios::app);
            outfile << resultsL << endl;
            outfile.close();
            outfile.open(outputM, ios::app);
            outfile << resultsM << endl;
            outfile.close();
            outfile.open(outputN, ios::app);
            outfile << resultsN << endl;
            outfile.close();
            outfile.open(outputO, ios::app);
            outfile << resultsO << endl;
            outfile.close();
            outfile.open(outputP, ios::app);
            outfile << resultsP << endl;
            outfile.close();
            outfile.open(outputQ, ios::app);
            outfile << resultsQ << endl;
            outfile.close();
            outfile.open(outputR, ios::app);
            outfile << resultsR << endl;
            outfile.close();
            outfile.open(outputS, ios::app);
            outfile << resultsS << endl;
            outfile.close();
            outfile.open(outputT, ios::app);
            outfile << resultsT << endl;
            outfile.close();
            outfile.open(outputU, ios::app);
            outfile << resultsU << endl;
            outfile.close();
            outfile.open(outputV, ios::app);
            outfile << resultsV << endl;
            outfile.close();
            outfile.open(outputW, ios::app);
            outfile << resultsW << endl;
            outfile.close();
            outfile.open(outputX, ios::app);
            outfile << resultsX << endl;
            outfile.close();
            outfile.open(outputY, ios::app);
            outfile << resultsY << endl;
            outfile.close();
            outfile.open(outputZ, ios::app);
            outfile << resultsZ << endl;
            outfile.close();
            outfile.open(outputNum, ios::app);
            outfile << resultsNum << endl;
            outfile.close();
            outfile.open(outputPercentSign, ios::app);
            outfile << resultsPercentSign << endl;
            outfile.close();
            outfile.open(outputUnderline, ios::app);
            outfile << resultsUnderline << endl;
            outfile.close();
            outfile.open(outputUnique, ios::app);
            outfile << resultsUnique << endl;
            outfile.close();
        }
        else
        {
            cout << "normal search" << endl;
            searchFiles(searchPath, "*");
            //将结果写入文件
            outfile.open(outputA, ios::app);
            outfile << resultsA << endl;
            outfile.close();
            outfile.open(outputB, ios::app);
            outfile << resultsB << endl;
            outfile.close();
            outfile.open(outputC, ios::app);
            outfile << resultsC << endl;
            outfile.close();
            outfile.open(outputD, ios::app);
            outfile << resultsD << endl;
            outfile.close();
            outfile.open(outputE, ios::app);
            outfile << resultsE << endl;
            outfile.close();
            outfile.open(outputF, ios::app);
            outfile << resultsF << endl;
            outfile.close();
            outfile.open(outputG, ios::app);
            outfile << resultsG << endl;
            outfile.close();
            outfile.open(outputH, ios::app);
            outfile << resultsH << endl;
            outfile.close();
            outfile.open(outputI, ios::app);
            outfile << resultsI << endl;
            outfile.close();
            outfile.open(outputJ, ios::app);
            outfile << resultsJ << endl;
            outfile.close();
            outfile.open(outputK, ios::app);
            outfile << resultsK << endl;
            outfile.close();
            outfile.open(outputL, ios::app);
            outfile << resultsL << endl;
            outfile.close();
            outfile.open(outputM, ios::app);
            outfile << resultsM << endl;
            outfile.close();
            outfile.open(outputN, ios::app);
            outfile << resultsN << endl;
            outfile.close();
            outfile.open(outputO, ios::app);
            outfile << resultsO << endl;
            outfile.close();
            outfile.open(outputP, ios::app);
            outfile << resultsP << endl;
            outfile.close();
            outfile.open(outputQ, ios::app);
            outfile << resultsQ << endl;
            outfile.close();
            outfile.open(outputR, ios::app);
            outfile << resultsR << endl;
            outfile.close();
            outfile.open(outputS, ios::app);
            outfile << resultsS << endl;
            outfile.close();
            outfile.open(outputT, ios::app);
            outfile << resultsT << endl;
            outfile.close();
            outfile.open(outputU, ios::app);
            outfile << resultsU << endl;
            outfile.close();
            outfile.open(outputV, ios::app);
            outfile << resultsV << endl;
            outfile.close();
            outfile.open(outputW, ios::app);
            outfile << resultsW << endl;
            outfile.close();
            outfile.open(outputX, ios::app);
            outfile << resultsX << endl;
            outfile.close();
            outfile.open(outputY, ios::app);
            outfile << resultsY << endl;
            outfile.close();
            outfile.open(outputZ, ios::app);
            outfile << resultsZ << endl;
            outfile.close();
            outfile.open(outputNum, ios::app);
            outfile << resultsNum << endl;
            outfile.close();
            outfile.open(outputPercentSign, ios::app);
            outfile << resultsPercentSign << endl;
            outfile.close();
            outfile.open(outputUnderline, ios::app);
            outfile << resultsUnderline << endl;
            outfile.close();
            outfile.open(outputUnique, ios::app);
            outfile << resultsUnique << endl;
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
int main(){
    setSearchDepth(6);
    searchFiles("D:", "*");
}

#endif
