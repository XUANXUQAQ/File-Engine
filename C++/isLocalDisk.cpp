#include <iostream>
#include <windows.h>
//#define TEST

using namespace std;

extern "C" __declspec(dllexport) bool isLocalDisk(char *path);

__declspec(dllexport) bool isLocalDisk(char *path)
{
    char disk[10];
    strcpy(disk, path);
    UINT type = GetDriveType((LPCSTR)disk);
    if (type == DRIVE_FIXED)
    {
        return true;
    }
    return false;
}

#ifdef TEST
int main()
{
    cout << isLocalDisk("D:\\") << endl;
    getchar();
}
#endif