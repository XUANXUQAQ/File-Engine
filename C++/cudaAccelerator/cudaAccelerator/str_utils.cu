#include "pch.h"
#include "cuda_runtime.h"
#include "str_utils.cuh"

__device__ int strcmp_cuda(const char* str1, const char* str2)
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


__device__ char* strlwr_cuda(char* src)
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


__device__ const char* strstr_cuda(const char* s1, const char* s2)
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
        return nullptr;
    }
    return s1;
}

__device__ char* strrchr_cuda(const char* s, int c)
{
    if (s == nullptr)
    {
        return nullptr;
    }

    char* p_char = nullptr;
    while (*s != '\0')
    {
        if (*s == static_cast<char>(c))
        {
            p_char = const_cast<char*>(s);
        }
        ++s;
    }

    return p_char;
}

__device__ char* strcpy_cuda(char* dst, const char* src)
{
    char* ret = dst;
    while ((*dst++ = *src++) != '\0')
    {
    }
    return ret;
}

__device__ size_t strlen_cuda(const char* str)
{
    size_t count = 0;
    while (*str != '\0')
    {
        count++;
        ++str;
    }
    return count;
}

__device__ char* strcat_cuda(char* dst, char const* src)
{
    if (dst == nullptr || src == nullptr)
    {
        return nullptr;
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

__device__ void str_add_single(char* dst, const char c)
{
    while (*dst != '\0')
    {
        dst++;
    }
    *dst++ = c;
    *dst = '\0';
}

__device__ bool is_str_contains_chinese(const char* source)
{
    int i = 0;
    while (source[i] != 0)
    {
        if (source[i] & 0x80 && source[i] & 0x40 && source[i] & 0x20)
        {
            return true;
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
    return false;
}

__device__ void get_file_name(const char* path, char* output)
{
    const char* p = strrchr_cuda(path, '\\');
    strcpy_cuda(output, p + 1);
}

__device__ void get_parent_path(const char* path, char* output)
{
    strcpy_cuda(output, path);
    char* p = strrchr_cuda(output, '\\');
    *p = '\0';
}
