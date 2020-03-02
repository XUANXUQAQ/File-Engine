//#define TEST
#include <iostream>
#include <cstring>


using namespace std;
extern "C" __declspec(dllexport) int getAscII(const char *str);


bool includeChinese(string strs)
{
    char _str[260];
    strcpy(_str, strs.c_str());
    char *str = _str;
    char c;
    while (1)
    {
        c = *str++;
        if (c == 0)
            break;    //如果到字符串尾则说明该字符串没有中文字符
        if (c & 0x80) //如果字符高位为1且下一字符高位也是1则有中文字符
            if (*str & 0x80)
                return true;
    }
    return false;
}


__declspec(dllexport) int getAscII(const char *str)
{
    if (includeChinese(str))
    {
        return 1;
    }
    char _str[260];
    strcpy(_str, str);
    char *s = _str;
    int sum = 0;
    int length = strlen(_str);
    for (int i = 0; i < length; i++)
    {
        sum += s[i];
    }
    return sum;
}

#ifdef TEST
int main()
{
    int asc = getAscII("SANDISK备份");
    cout << asc << endl;
    getchar();
}
#endif