// #define TEST
#include <corecrt_io.h>
#include <direct.h>
#include <Windows.h>
#include <ShlObj.h>
#include "resource.h"
#include <TlHelp32.h>
#include <fstream>
#include <string>
#include <Psapi.h>
#include <tchar.h>
#include <codecvt>
#include "zip/zip.h"
#include "md5.h"
#include <unordered_set>
#ifdef TEST
#include <iostream>
#endif

#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "Ole32.lib")
#pragma comment(lib, "User32.lib")

// TODO 该变量为File-Engine.zip中的File-Engine.jar的md5值
#define FILE_ENGINE_JAR_MD5 "c0cf09e0b6d069aa2efb62a7834a727d"

constexpr auto CHECK_TIME_THRESHOLD = 1;

const std::string redirect_error_file_option = "-XX:ErrorFile=../logs/hs_err_pid%p.log";
const std::string default_jvm_params =
"-Xms8M "
"-Xmx256M "
"-XX:MaxHeapFreeRatio=20 "
"-XX:MinHeapFreeRatio=10 "
"-Dsun.java2d.noddraw=true "
"-XX:+CompactStrings " +
redirect_error_file_option;

#ifndef TEST
#pragma comment( linker, "/subsystem:windows /entry:mainCRTStartup" )
#endif

constexpr auto* g_file_engine_zip_name = "File-Engine.zip";

wchar_t g_close_signal_file[1000]{ 0 };
wchar_t g_open_from_jar_signal_file[1000]{ 0 };
wchar_t g_file_engine_jar_path[1000]{ 0 };
wchar_t g_file_engine_working_dir[1000]{ 0 };
wchar_t g_jre_path[1000]{ 0 };
wchar_t g_update_signal_file[1000]{ 0 };
wchar_t g_new_file_engine_jar_path[1000]{ 0 };
wchar_t g_log_file_path[1000]{ 0 };
wchar_t g_jvm_parameter_file_path[1000]{ 0 };
short g_restart_count = 0;

bool is_close_exist();
DWORD find_process();
void restart_file_engine(bool);
bool release_resources();
void extract_zip();
bool is_file_exist(const wchar_t* file_path);
void release_all();
bool is_dir_not_exist(const wchar_t* path);
void update();
void init_path();
bool is_launched();
std::wstring get_self_name();
void delete_jre_dir();
bool remove_dir(const wchar_t* szFileDir);
std::wstring init_jvm_parameters();
std::wstring string2wstring(const std::string& str);
std::string& trim(std::string& s);
std::string to_utf8(const wchar_t* buffer, const int len);
std::string to_utf8(const std::wstring& str);

int main()
{
    if (is_launched())
    {
        return 0;
    }
    init_path();
    // check_logs();
#ifdef TEST
    std::cout << "file-engine.jar path :  " << g_file_engine_jar_path << std::endl;
    std::cout << "jre path: " << g_jre_path << std::endl;
    std::cout << "file-engine working dir: " << g_file_engine_working_dir << std::endl;
    std::cout << "new file-engine.jar path: " << g_new_file_engine_jar_path << std::endl;
    std::cout << "update signal file: " << g_update_signal_file << std::endl;
    std::cout << "close signal file : " << g_close_signal_file << std::endl;
    std::cout << "log file path: " << g_log_file_path << std::endl;
#endif
    if (is_dir_not_exist(g_log_file_path))
    {
        if (_wmkdir(g_log_file_path))
        {
            return 0;
        }
    }
    if (is_dir_not_exist(g_file_engine_working_dir))
    {
        if (_wmkdir(g_file_engine_working_dir))
        {
            return 0;
        }
    }
    if (!is_file_exist(g_open_from_jar_signal_file))
    {
        const DWORD pid = find_process();
        if (pid)
        {
            system(("TASKKILL /PID " + std::to_string(pid) + " /F").c_str());
        }
#ifdef TEST
        std::cout << "starting file-engine" << std::endl;
#endif
        restart_file_engine(true);
    }
    else
    {
        _wremove(g_open_from_jar_signal_file);
    }
#ifdef TEST
    std::cout << "start loop" << std::endl;
#endif
    std::time_t start_time = std::time(nullptr);
    while (!is_close_exist())
    {
        Sleep(500);
        const std::time_t tmp = std::time(nullptr) - start_time;
        if (tmp > CHECK_TIME_THRESHOLD)
        {
            start_time = std::time(nullptr);
            if (!find_process())
            {
#ifdef TEST
                std::cout << "File-Engine process not exist" << std::endl;
#endif
                restart_file_engine(false);
            }
        }
    }
    return 0;
}

std::wstring init_jvm_parameters()
{
    // 读取默认jvm启动参数
    std::unordered_set<std::string> default_jvm_parameters_set;
    std::string jvm_parameter(default_jvm_params);
    size_t pos;
    while ((pos = jvm_parameter.find(' ')) != std::string::npos)
    {
        std::string token = jvm_parameter.substr(0, pos);
        default_jvm_parameters_set.insert(trim(token));
        jvm_parameter.erase(0, pos + 1);
    }
    default_jvm_parameters_set.insert(trim(jvm_parameter));
    // 读取自定义jvm启动参数
    std::unordered_set<std::string> custom_vm_options;
    bool is_all_default = true;
    if (is_file_exist(g_jvm_parameter_file_path))
    {
        std::ifstream input_stream(g_jvm_parameter_file_path, std::ios::binary);
        std::string line;
        while (std::getline(input_stream, line))
        {
            auto&& vm_option = trim(line);
            if (is_all_default && default_jvm_parameters_set.find(vm_option) == default_jvm_parameters_set.end())
            {
                is_all_default = false;
            }
            custom_vm_options.insert(vm_option);
        }
        input_stream.close();
        if (custom_vm_options.find(redirect_error_file_option) == custom_vm_options.end())
        {
            custom_vm_options.insert(redirect_error_file_option);
        }
    }
    std::wstring vm_option_str;
    if (is_all_default)
    {
        vm_option_str = string2wstring(default_jvm_params);
    }
    else
    {
        for (const auto& custom_vm_option : custom_vm_options)
        {
            vm_option_str += string2wstring(custom_vm_option);
            vm_option_str += ' ';
        }
    }
    // 保存到文件
    std::ofstream output_stream(g_jvm_parameter_file_path, std::ios::binary);
    for (wchar_t each_char : vm_option_str)
    {
        output_stream.put(each_char == ' ' ? '\n' : each_char);
    }
    output_stream.close();
    return vm_option_str;
}

std::string& trim(std::string& s)
{
    if (s.empty())
    {
        return s;
    }
    s.erase(0, s.find_first_not_of(' '));
    s.erase(s.find_last_not_of(' ') + 1);
    return s;
}

inline void init_path()
{
    wchar_t current_dir[1000]{ 0 };
    GetModuleFileName(nullptr, current_dir, sizeof current_dir / 2);
    const std::wstring tmp_current_dir(current_dir);
    wcscpy_s(current_dir, tmp_current_dir.substr(0, tmp_current_dir.find_last_of(L'\\')).c_str());

    std::wstring _file_engine_log_path(current_dir);
    _file_engine_log_path += L"\\logs\\";
    wcscpy_s(g_log_file_path, _file_engine_log_path.c_str());

    std::wstring file_engine_jar_dir_string(current_dir);
    file_engine_jar_dir_string += L"\\data\\";
    wcscpy_s(g_file_engine_working_dir, file_engine_jar_dir_string.c_str());

    std::wstring jre_path(file_engine_jar_dir_string);
    jre_path += L"jre\\";
    wcscpy_s(g_jre_path, jre_path.c_str());

    file_engine_jar_dir_string += L"File-Engine.jar";
    wcscpy_s(g_file_engine_jar_path, file_engine_jar_dir_string.c_str());

    std::wstring file_engine_directory(g_file_engine_working_dir);
    file_engine_directory += L"tmp\\closeDaemon";
    wcscpy_s(g_close_signal_file, file_engine_directory.c_str());

    std::wstring open_from_jar_signal_file(g_file_engine_working_dir);
    open_from_jar_signal_file += L"tmp\\openFromJar";
    wcscpy_s(g_open_from_jar_signal_file, open_from_jar_signal_file.c_str());

    std::wstring new_file_engine_path(g_file_engine_working_dir);
    new_file_engine_path += L"tmp\\File-Engine.jar";
    wcscpy_s(g_new_file_engine_jar_path, new_file_engine_path.c_str());

    std::wstring update_signal_file(g_file_engine_working_dir);
    update_signal_file += L"user\\update";
    wcscpy_s(g_update_signal_file, update_signal_file.c_str());

    std::wstring jvm_parameter_file(g_file_engine_working_dir);
    jvm_parameter_file += L"jvm.vmoptions";
    wcscpy_s(g_jvm_parameter_file_path, jvm_parameter_file.c_str());
}

inline void delete_jre_dir()
{
    remove_dir(g_jre_path);
}

/**
 * 释放所有文件
 */
void release_all()
{
    // 删除jre文件夹
    delete_jre_dir();
    _wremove(g_jvm_parameter_file_path);
    if (release_resources())
    {
        extract_zip();
#ifndef TEST
        remove(g_file_engine_zip_name);
#endif
    }
}

/**
 * 释放File-Engine.zip文件到当前文件夹
 */
bool release_resources()
{
    HRSRC hRsrc = FindResource(nullptr, MAKEINTRESOURCEW(IDR_ZIP2), L"ZIP");
    if (nullptr == hRsrc)
    {
        return false;
    }
    const DWORD size = SizeofResource(nullptr, hRsrc);
    if (0 == size)
    {
        return false;
    }
    const HGLOBAL hGlobal = LoadResource(nullptr, hRsrc);
    if (nullptr == hGlobal)
    {
        return false;
    }
    const LPVOID pBuffer = LockResource(hGlobal);
    if (nullptr == pBuffer)
    {
        return false;
    }
    FILE* fp;
    if (fopen_s(&fp, g_file_engine_zip_name, "wb"))
    {
        return false;
    }
    if (nullptr != fp)
    {
        if (size != fwrite(pBuffer, 1, size, fp))
        {
            fclose(fp);
            return false;
        }
        fclose(fp);
        return true;
    }
    return false;
}

/**
 * 解压File-Engine.zip
 */
inline void extract_zip()
{
    std::wstring working_dir(g_file_engine_working_dir);
    auto&& w_working_dir = to_utf8(working_dir);
    zip_extract(g_file_engine_zip_name, w_working_dir.c_str(), nullptr, nullptr);
}

/**
 * 重启程序
 */
void restart_file_engine(const bool is_ignore_close_file)
{
    if (is_ignore_close_file)
    {
        _wremove(g_close_signal_file);
    }
    else
    {
        if (is_close_exist())
        {
            return;
        }
    }
    // 更新到新版本
    if (is_file_exist(g_update_signal_file))
    {
        update();
    }
    else if (is_file_exist(g_file_engine_jar_path))
    {
        // 检查File-Engine.jar是否与启动器中的版本一致
        auto&& jar_md5 = GetFileHash(g_file_engine_jar_path);
        if (jar_md5 != FILE_ENGINE_JAR_MD5)
        {
            release_all();
        }
    }
    if (g_restart_count >= 4)
    {
        MessageBoxA(nullptr, "Launch failed", "Error", MB_OK);
        std::quick_exit(-1);
    }
    if (g_restart_count >= 3 || !is_file_exist(g_file_engine_jar_path))
    {
#ifdef TEST
        std::cout << "release File-Engine.zip" << std::endl;
#endif
        release_all();
    }
    g_restart_count++;

    auto&& vm_options = init_jvm_parameters();

    std::wstring command(L"/c ");
    const std::wstring jre(g_jre_path);
    command.append(jre.substr(0, 2));
    command.append(L"\"");
    command.append(jre.substr(2));
    command.append(L"bin\\javaw.exe\" ").append(vm_options).append(L" -jar File-Engine.jar");
    // .append(" 1> normal.log")
    // .append(" 2>> ").append("\"").append(g_log_file_path).append(get_date()).append(".log").append("\"");
#ifdef TEST
    std::cout << "running command: " << command << std::endl;
#endif
    ShellExecute(nullptr, L"open", L"cmd", command.c_str(), g_file_engine_working_dir, SW_HIDE);
}

/**
 * 更新File-Engine
 */
void update()
{
    CopyFile(g_new_file_engine_jar_path, g_file_engine_jar_path, false);
    _wremove(g_update_signal_file);
}

/**
 * 检查关闭标志是否存在
 */
bool is_close_exist()
{
    return is_file_exist(g_close_signal_file);
}

bool is_dir_not_exist(const wchar_t* path)
{
    const int tmp = _waccess(path, 0);
    return -1 == tmp;
}

bool is_file_exist(const wchar_t* file_path)
{
    return GetFileAttributes(file_path) != INVALID_FILE_ATTRIBUTES;
}

BOOL dos_path_to_nt_path(LPTSTR pszDosPath, LPTSTR pszNtPath)
{
    TCHAR szDriveStr[500]{ 0 };

    //检查参数
    if (!pszDosPath || !pszNtPath)
        return FALSE;

    //获取本地磁盘字符串
    if (GetLogicalDriveStrings(500, szDriveStr))
    {
        TCHAR szDrive[3]{ 0 };
        for (int i = 0; szDriveStr[i]; i += 4)
        {
            TCHAR szDevName[100]{ 0 };
            if (!lstrcmpi(&(szDriveStr[i]), TEXT("A:\\")) || !lstrcmpi(&(szDriveStr[i]), TEXT("B:\\")))
                continue;

            szDrive[0] = szDriveStr[i];
            szDrive[1] = szDriveStr[i + 1];
            szDrive[2] = '\0';
            if (!QueryDosDevice(szDrive, szDevName, 100)) //查询 Dos 设备名
                return FALSE;

            const int cchDevName = lstrlen(szDevName);
            if (_tcsnicmp(pszDosPath, szDevName, cchDevName) == 0) //命中
            {
                lstrcpy(pszNtPath, szDrive); //复制驱动器
                lstrcat(pszNtPath, pszDosPath + cchDevName); //复制路径

                return TRUE;
            }
        }
    }

    lstrcpy(pszNtPath, pszDosPath);

    return FALSE;
}

//获取进程完整路径
BOOL get_process_full_path(DWORD dwPID, TCHAR* pszFullPath)
{
    TCHAR szImagePath[MAX_PATH];
    if (!pszFullPath)
        return FALSE;

    pszFullPath[0] = '\0';
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION, 0, dwPID);
    if (!hProcess)
        return FALSE;

    if (!GetProcessImageFileName(hProcess, szImagePath, MAX_PATH))
    {
        CloseHandle(hProcess);
        return FALSE;
    }

    if (!dos_path_to_nt_path(szImagePath, pszFullPath))
    {
        CloseHandle(hProcess);
        return FALSE;
    }

    CloseHandle(hProcess);

    return TRUE;
}

std::wstring get_self_name()
{
    WCHAR _proc_name[MAX_PATH];
    GetModuleFileNameW(nullptr, _proc_name, sizeof _proc_name / 2);
    const std::wstring proc_name(_proc_name);
    return proc_name.substr(proc_name.find_last_of(L'\\') + 1);
}

bool is_launched()
{
    PROCESSENTRY32 pe;
    auto* const hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    pe.dwSize = sizeof(PROCESSENTRY32);
    if (!Process32First(hSnapshot, &pe))
    {
        CloseHandle(hSnapshot);
        return false;
    }
    int proc_num = 0;
    while (true)
    {
        pe.dwSize = sizeof(PROCESSENTRY32);
        if (Process32Next(hSnapshot, &pe) == FALSE)
            break;
        if (wcscmp(pe.szExeFile, get_self_name().c_str()) == 0)
        {
            proc_num++;
            if (proc_num > 1)
            {
                CloseHandle(hSnapshot);
                return true;
            }
        }
    }
    CloseHandle(hSnapshot);
    return false;
}

/**
 * 查找File-Engine进程是否存在
 */
DWORD find_process()
{
    try
    {
        PROCESSENTRY32 pe;
        DWORD ret = 0;
        auto* const hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        pe.dwSize = sizeof(PROCESSENTRY32);
        const std::wstring workingDir(g_file_engine_working_dir);
        if (!Process32First(hSnapshot, &pe))
        {
            CloseHandle(hSnapshot);
            return ret;
        }
        while (true)
        {
            pe.dwSize = sizeof(PROCESSENTRY32);
            if (Process32Next(hSnapshot, &pe) == FALSE)
                break;
            if (wcscmp(pe.szExeFile, L"javaw.exe") == 0)
            {
                const DWORD id = pe.th32ProcessID;
                TCHAR szProcessName[1000] = { 0 };
                get_process_full_path(id, szProcessName);
                std::wstring processName(szProcessName);
                if (processName.find(workingDir) != std::wstring::npos)
                {
                    ret = id;
                    break;
                }
            }
        }
        CloseHandle(hSnapshot);
        return ret;
    }
    catch (std::exception&)
    {
        std::quick_exit(-1);
    }
}

std::string to_utf8(const std::wstring& str)
{
    return to_utf8(str.c_str(), static_cast<int>(str.size()));
}

std::string to_utf8(const wchar_t* buffer, const int len)
{
    const auto n_chars = WideCharToMultiByte(
        CP_UTF8,
        0,
        buffer,
        len,
        nullptr,
        0,
        nullptr,
        nullptr);
    if (n_chars == 0)
    {
        return "";
    }
    std::string new_buffer;
    new_buffer.resize(n_chars);
    WideCharToMultiByte(
        CP_UTF8,
        0,
        buffer,
        len,
        const_cast<char*>(new_buffer.c_str()),
        n_chars,
        nullptr,
        nullptr);

    return new_buffer;
}

std::wstring string2wstring(const std::string& str)
{
    std::wstring result;
    //获取缓冲区大小，并申请空间，缓冲区大小按字符计算
    const int len = MultiByteToWideChar(CP_ACP, 0, str.c_str(), static_cast<int>(str.size()), nullptr, 0);
    const auto buffer = new TCHAR[len + 1];
    //多字节编码转换成宽字节编码
    MultiByteToWideChar(CP_ACP, 0, str.c_str(), static_cast<int>(str.size()), buffer, len);
    buffer[len] = '\0';
    //删除缓冲区并返回值
    result.append(buffer);
    delete[] buffer;
    return result;
}

bool remove_dir(const wchar_t* szFileDir)
{
    std::wstring strDir = szFileDir;
    if (strDir.at(strDir.length() - 1) != L'\\')
        strDir += L'\\';
    WIN32_FIND_DATA wfd;
    HANDLE hFind = FindFirstFile((strDir + L"*.*").c_str(), &wfd);
    if (hFind == INVALID_HANDLE_VALUE)
        return false;
    do
    {
        if (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            if (_wcsicmp(wfd.cFileName, L".") != 0 &&
                _wcsicmp(wfd.cFileName, L"..") != 0)
                remove_dir((strDir + wfd.cFileName).c_str());
        }
        else
        {
            DeleteFile((strDir + wfd.cFileName).c_str());
        }
    } while (FindNextFile(hFind, &wfd));
    FindClose(hFind);
    RemoveDirectory(szFileDir);
    return true;
}
