//#define TEST
#include <iostream>
#include <cstring>

using namespace std;
extern "C" __declspec(dllexport) int getAscII(const char *str);

__declspec(dllexport) int getAscII(const char *str)
{
    char _str[260];
    strcpy(_str, str);
    char *s = _str;
    int sum = 0;
    int length = strlen(_str);
    for (int i = 0; i < length; i++)
    {
        if (s[i] > 0)
        {
            sum += s[i];
        }
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