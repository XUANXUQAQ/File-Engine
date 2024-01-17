#include "pch.h"
#include "str_convert.h"

#define COMPBYTE(x, y) ((unsigned char)(x) << 8 | (unsigned char)(y))

unsigned short utf162gbk[0x10000] = {0};

void init_utf162_gbk();
void init_gbk2_utf16_2();
void init_gbk2_utf16_3();

unsigned short gbk2_utf16_3_size;
unsigned short gbk2_utf16_2_size;

void init_str_convert()
{
    init_gbk2_utf16_2();
    init_gbk2_utf16_3();
    // init_gbk2_utf16();
    init_utf162_gbk();
}

void init_gbk2_utf16_2()
{
    gbk2_utf16_2_size = sizeof gbk2_utf16_2_host / sizeof(short);
}

void init_gbk2_utf16_3()
{
    gbk2_utf16_3_size = sizeof gbk2_utf16_3_host / sizeof(short);
}

void init_utf162_gbk()
{
    unsigned short c;
    for (c = 0; c < gbk2_utf16_2_size; c += 2)
        utf162gbk[gbk2_utf16_2_host[c + 1]] = gbk2_utf16_2_host[c];

    for (c = 0; c < gbk2_utf16_3_size; c += 3)
        for (unsigned short d = gbk2_utf16_3_host[c]; d <= gbk2_utf16_3_host[c + 1]; d++)
            utf162gbk[gbk2_utf16_3_host[c + 2] + d - gbk2_utf16_3_host[c]] = d;
}

int utf8_to_gbk(const char* from, unsigned int from_len, char** to, unsigned int* to_len)
{
    char* result = *to;
    unsigned i_to = 0;

    if (from_len == 0 || from == nullptr || to == nullptr || result == nullptr)
    {
        return -1;
    }

    for (unsigned i_from = 0; i_from < from_len;)
    {
        if (static_cast<unsigned char>(from[i_from]) < 0x80)
        {
            result[i_to++] = from[i_from++];
        }
        else if (static_cast<unsigned char>(from[i_from]) < 0xC2)
        {
            i_from++;
        }
        else if (static_cast<unsigned char>(from[i_from]) < 0xE0)
        {
            if (i_from >= from_len - 1) break;

            const unsigned short tmp = utf162gbk[(from[i_from] & 0x1F) << 6 | from[i_from + 1] & 0x3F];

            if (tmp)
            {
                result[i_to++] = tmp >> 8;
                result[i_to++] = tmp & 0xFF;
            }

            i_from += 2;
        }
        else if (static_cast<unsigned char>(from[i_from]) < 0xF0)
        {
            if (i_from >= from_len - 2) break;

            const unsigned short tmp = utf162gbk[(from[i_from] & 0x0F) << 12
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
    if (to_len != nullptr)
    {
        *to_len = i_to;
    }
    return 0;
}
