#include <Windows.h>
#include <string>
#include "file_utils.h"

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


bool is_file_exist(const char* path)
{
    struct _stat64i32 buffer;
    return _wstat(string2wstring(path).c_str(), &buffer) == 0;
}
