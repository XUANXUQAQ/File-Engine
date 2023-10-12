#include "NTFSChangesWatcher.h"

#include <chrono>
#include <ranges>
#include "string_wstring_converter.h"

constexpr unsigned MAX_USN_CACHE_SIZE = 1000000;
unsigned remain_cache_number = MAX_USN_CACHE_SIZE;

const int NTFSChangesWatcher::kBufferSize = 1024 * 1024 / 2;
const int NTFSChangesWatcher::FILE_CHANGE_BITMASK = USN_REASON_RENAME_NEW_NAME | USN_REASON_RENAME_OLD_NAME;
const std::string file_monitor_exit_flag = "$$__File-Engine-Exit-Monitor__$$";
const static std::wstring sep_wstr(L"\\");
const static std::u16string sep(sep_wstr.begin(), sep_wstr.end());

inline bool is_file_exist(const std::string& path)
{
    struct _stat64i32 buffer;
    return _wstat(string2wstring(path).c_str(), &buffer) == 0;
}

NTFSChangesWatcher::NTFSChangesWatcher(char drive_letter) :
    drive_letter_(drive_letter)
{
    volume_ = OpenVolume(drive_letter_);

    journal_ = std::make_unique<USN_JOURNAL_DATA>();

    if (const bool res = LoadJournal(volume_, journal_.get()); !res)
    {
        fprintf(stderr, "Failed to load journal, Error code: %lu\n", GetLastError());
        return;
    }
    max_usn_ = journal_->MaxUsn;
    journal_id_ = journal_->UsnJournalID;
    last_usn_ = journal_->NextUsn;
    stop_flag = false;
    is_delete_usn_on_exit_ = false;
    auto&& exit_file = drive_letter_ + file_monitor_exit_flag;
    if (is_file_exist(exit_file))
    {
        if (remove(exit_file.c_str()) != 0)
        {
            fprintf(stderr, "Delete exit monitor file failed.");
        }
    }
}

NTFSChangesWatcher::~NTFSChangesWatcher()
{
    CloseHandle(volume_);
}


HANDLE NTFSChangesWatcher::OpenVolume(const char drive_letter)
{
    wchar_t pattern[10] = L"\\\\?\\a:";

    pattern[4] = static_cast<wchar_t>(drive_letter);

    const HANDLE volume = CreateFile(
        pattern, // lpFileName
        // also could be | FILE_READ_DATA | FILE_READ_ATTRIBUTES | SYNCHRONIZE
        GENERIC_READ | GENERIC_WRITE | SYNCHRONIZE, // dwDesiredAccess
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, // share mode
        nullptr, // default security attributes
        OPEN_EXISTING, // disposition
        // It is always set, no matter whether you explicitly specify it or not. This means, that access
        // must be aligned with sector size so we can only read a number of bytes that is a multiple of the sector size.
        FILE_FLAG_NO_BUFFERING, // file attributes
        nullptr // do not copy file attributes
    );

    if (volume == INVALID_HANDLE_VALUE)
    {
        // An error occurred!
        fprintf(stderr, "Failed to open volume");
        return nullptr;
    }

    return volume;
}


bool NTFSChangesWatcher::CreateJournal(HANDLE volume)
{
    DWORD byte_count;
    CREATE_USN_JOURNAL_DATA create_journal_data{};

    const bool ok = DeviceIoControl(volume, // handle to volume
                                    FSCTL_CREATE_USN_JOURNAL, // dwIoControlCode
                                    &create_journal_data, // input buffer
                                    sizeof(create_journal_data), // size of input buffer
                                    nullptr, // lpOutBuffer
                                    0, // nOutBufferSize
                                    &byte_count, // number of bytes returned
                                    nullptr) != 0; // OVERLAPPED structure

    if (!ok)
    {
        // An error occurred!
    }

    return ok;
}


bool NTFSChangesWatcher::LoadJournal(HANDLE volume, USN_JOURNAL_DATA* journal_data)
{
    DWORD byte_count;

    // Try to open journal.
    if (!DeviceIoControl(volume,
                         FSCTL_QUERY_USN_JOURNAL,
                         nullptr,
                         0,
                         journal_data,
                         sizeof(*journal_data),
                         &byte_count,
                         nullptr))
    {
        // If failed (for example, in case journaling is disabled), create journal and retry.

        if (CreateJournal(volume))
        {
            return LoadJournal(volume, journal_data);
        }

        return false;
    }

    return true;
}

void NTFSChangesWatcher::DeleteUsnOnExit()
{
    is_delete_usn_on_exit_ = true;
}

void NTFSChangesWatcher::StopWatch()
{
    stop_flag = true;
    auto&& exit_file = drive_letter_ + std::string(":\\") + file_monitor_exit_flag;
    if (!is_file_exist(exit_file))
    {
        FILE* fp = nullptr;
        if (const auto ret_val = fopen_s(&fp, exit_file.c_str(), "w"); ret_val != 0)
        {
            fprintf(stderr, "Create exit monitor file failed. Error code: %d\n", ret_val);
        }
        if (fp != nullptr)
        {
            if (fclose(fp) != 0)
            {
                fprintf(stderr, "Close exit monitor file failed.\n");
            }
        }
    }
    if (const auto ret = remove(exit_file.c_str()); ret != 0)
    {
        fprintf(stderr, "Delete exit monitor file failed. Error code: %d\n", ret);
    }
}

bool NTFSChangesWatcher::DeleteJournal() const
{
    DELETE_USN_JOURNAL_DATA dujd;
    dujd.UsnJournalID = journal_id_;
    dujd.DeleteFlags = USN_DELETE_FLAG_DELETE | USN_DELETE_FLAG_NOTIFY;
    DWORD br;

    if (DeviceIoControl(volume_,
                        FSCTL_DELETE_USN_JOURNAL,
                        &dujd,
                        sizeof(dujd),
                        nullptr,
                        0,
                        &br,
                        nullptr)
    )
    {
        return true;
    }
    return false;
}

void NTFSChangesWatcher::WatchChanges(
    void (*file_added_callback_func)(const std::u16string&),
    void (*file_removed_callback_func)(const std::u16string&))
{
    stop_flag = false;
    const auto u_buffer = std::make_unique<char[]>(kBufferSize);

    const auto read_journal_query = GetWaitForNextUsnQuery(last_usn_);

    while (!stop_flag)
    {
        // This function does not return until new USN record created.
        WaitForNextUsn(read_journal_query.get());
        last_usn_ = ReadChangesAndNotify(read_journal_query->StartUsn,
                                         u_buffer.get(),
                                         file_added_callback_func,
                                         file_removed_callback_func);
        read_journal_query->StartUsn = last_usn_;
    }
    if (is_delete_usn_on_exit_)
    {
        if (!DeleteJournal())
        {
            fprintf(stderr, "Failed to delete usn journal. Error code: %lu\n", GetLastError());
        }
    }
}

USN NTFSChangesWatcher::ReadChangesAndNotify(USN low_usn,
                                             char* buffer,
                                             void (*file_added_callback_func)(const std::u16string&),
                                             void (*file_removed_callback_func)(const std::u16string&))
{
    static const std::wstring recycle_bin(L"$RECYCLE.BIN");
    static const std::u16string recycle_bin_u16(recycle_bin.begin(), recycle_bin.end());

    DWORD byte_count;
    const auto journal_query = GetReadJournalQuery(low_usn);
    memset(buffer, 0, kBufferSize);
    if (!ReadJournalRecords(journal_query.get(), buffer, byte_count))
    {
        // An error occurred.
        return low_usn;
    }

    auto record = reinterpret_cast<USN_RECORD*>(reinterpret_cast<USN*>(buffer) + 1);
    const auto record_end = reinterpret_cast<USN_RECORD*>(reinterpret_cast<BYTE*>(buffer) + byte_count);

    std::u16string full_path;
    for (; record < record_end;
           record = reinterpret_cast<USN_RECORD*>(reinterpret_cast<BYTE*>(record) + record->RecordLength))
    {
        const auto reason = record->Reason;
        full_path.clear();
        // It is really strange, but some system files creating and deleting at the same time.
        if ((reason & USN_REASON_FILE_CREATE) && (reason & USN_REASON_FILE_DELETE))
        {
            continue;
        }
        if ((reason & USN_REASON_FILE_CREATE) && (reason & USN_REASON_CLOSE))
        {
            showRecord(full_path, record);
            if (full_path.find(recycle_bin_u16) == std::u16string::npos)
            {
                file_added_callback_func(full_path);
            }
        }
        else if ((reason & USN_REASON_FILE_DELETE) && (reason & USN_REASON_CLOSE))
        {
            showRecord(full_path, record);
            if (full_path.find(recycle_bin_u16) == std::u16string::npos)
            {
                file_removed_callback_func(full_path);
            }
        }
        else if (reason & FILE_CHANGE_BITMASK)
        {
            if (reason & USN_REASON_RENAME_OLD_NAME)
            {
                showRecord(full_path, record);
                if (full_path.find(recycle_bin_u16) == std::u16string::npos)
                {
                    file_removed_callback_func(full_path);
                }
            }
            else if (reason & USN_REASON_RENAME_NEW_NAME)
            {
                showRecord(full_path, record);
                if (full_path.find(recycle_bin_u16) == std::u16string::npos)
                {
                    file_added_callback_func(full_path);
                }
            }
        }
    }
    return *reinterpret_cast<USN*>(buffer);
}

bool NTFSChangesWatcher::WaitForNextUsn(PREAD_USN_JOURNAL_DATA read_journal_data) const
{
    DWORD bytes_read;

    // This function does not return until new USN record created.
    const bool ok = DeviceIoControl(volume_,
                                    FSCTL_READ_USN_JOURNAL,
                                    read_journal_data,
                                    sizeof(*read_journal_data),
                                    &read_journal_data->StartUsn,
                                    sizeof(read_journal_data->StartUsn),
                                    &bytes_read,
                                    nullptr) != 0;
    return ok;
}

std::unique_ptr<READ_USN_JOURNAL_DATA> NTFSChangesWatcher::GetWaitForNextUsnQuery(USN start_usn) const
{
    auto query = std::make_unique<READ_USN_JOURNAL_DATA>();

    query->StartUsn = start_usn;
    query->ReasonMask = 0xFFFFFFFF; // All bits.
    query->ReturnOnlyOnClose = FALSE; // All entries.
    query->Timeout = 0; // No timeout.
    query->BytesToWaitFor = 1; // Wait for this.
    query->UsnJournalID = journal_id_; // The journal.
    query->MinMajorVersion = 2;
    query->MaxMajorVersion = 2;

    return query;
}


bool NTFSChangesWatcher::ReadJournalRecords(PREAD_USN_JOURNAL_DATA journal_query, LPVOID buffer,
                                            DWORD& byte_count) const
{
    return DeviceIoControl(volume_,
                           FSCTL_READ_USN_JOURNAL,
                           journal_query,
                           sizeof(*journal_query),
                           buffer,
                           kBufferSize,
                           &byte_count,
                           nullptr) != 0;
}

std::unique_ptr<READ_USN_JOURNAL_DATA> NTFSChangesWatcher::GetReadJournalQuery(USN low_usn) const
{
    auto query = std::make_unique<READ_USN_JOURNAL_DATA>();

    query->StartUsn = low_usn;
    query->ReasonMask = 0xFFFFFFFF; // All bits.
    query->ReturnOnlyOnClose = FALSE;
    query->Timeout = 0; // No timeout.
    query->BytesToWaitFor = 0;
    query->UsnJournalID = journal_id_;
    query->MinMajorVersion = 2;
    query->MaxMajorVersion = 2;

    return query;
}

std::u16string NTFSChangesWatcher::GetFilename(USN_RECORD* record)
{
    const auto name_length = static_cast<unsigned short>(record->FileNameLength / 2);
    const auto filename = new char16_t[name_length + static_cast<size_t>(1)]();
    const auto* filename_in_record = reinterpret_cast<char16_t*>
        (reinterpret_cast<unsigned char*>(record) + record->FileNameOffset);
    memcpy(filename, filename_in_record, record->FileNameLength);
    *(filename + record->FileNameLength / 2) = L'\0';
    std::u16string file_name(filename);
    delete[] filename;
    return file_name;
}

/**
 * \brief
 * \param full_path 输出完整路径
 * \param record 需要转换的usn record
 */
void NTFSChangesWatcher::showRecord(std::u16string& full_path, USN_RECORD* record)
{
    cache_map_t temp_usn_cache;

    full_path += GetFilename(record);

    const auto start_loop = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    );
    // 清除最不常使用缓存
    if (remain_cache_number <= 0)
    {
        auto&& least_used_cache = std::ranges::min_element(frn_record_pfrn_map_,
                                                           [&](const std::pair<DWORDLONG,
                                                                               std::pair<
                                                                                   std::pair<std::u16string, DWORDLONG>,
                                                                                   DWORDLONG>>& left,
                                                               const std::pair<DWORDLONG,
                                                                               std::pair<
                                                                                   std::pair<std::u16string, DWORDLONG>,
                                                                                   DWORDLONG>>& right)
                                                           {
                                                               return left.second.first.second <
                                                                   right.second.first.second;
                                                           });
        frn_record_pfrn_map_.erase(least_used_cache->first);
        ++remain_cache_number;
    }
    //检查缓存是否已存在
    if (auto&& val = frn_record_pfrn_map_.find(record->ParentFileReferenceNumber);
        val != frn_record_pfrn_map_.end())
    {
        full_path = val->second.first.first + sep + full_path;
        auto& cache_used_timestamp = val->second.first.second;
        cache_used_timestamp = GetTickCount64();
    }
    else
    {
        DWORDLONG file_parent_id = record->ParentFileReferenceNumber;
        const auto usn_buffer = std::make_unique<char[]>(kBufferSize);
        do
        {
            MFT_ENUM_DATA_V0 med;
            med.StartFileReferenceNumber = file_parent_id;
            med.LowUsn = 0;
            med.HighUsn = max_usn_;
            DWORD byte_count = 1;
            if (!DeviceIoControl(volume_,
                                 FSCTL_ENUM_USN_DATA,
                                 &med,
                                 sizeof(med),
                                 usn_buffer.get(),
                                 kBufferSize,
                                 &byte_count,
                                 nullptr))
            {
                return;
            }
            // 防止超时卡死
            if (const auto loop_time = std::chrono::duration_cast<std::chrono::milliseconds>
                (std::chrono::system_clock::now().time_since_epoch()) -
                start_loop; loop_time.count() > 10000)
            {
                return;
            }
            // 读取父级usn record
            const auto parent_record = reinterpret_cast<USN_RECORD*>(reinterpret_cast<USN*>(usn_buffer.get()) + 1);
            if (parent_record->FileReferenceNumber != file_parent_id)
            {
                break;
            }
            const auto file_name = GetFilename(parent_record);
            full_path = file_name + sep + full_path;
            file_parent_id = parent_record->ParentFileReferenceNumber;

            // 添加到temp_usn_cache，递归出完整路径后再转移到frn_record_pfrn_map_
            auto&& path_using_count_pair = std::make_pair(file_name, GetTickCount64());
            auto&& path_usn_record_pair = std::make_pair(path_using_count_pair,
                                                         parent_record->ParentFileReferenceNumber);
            temp_usn_cache.insert(
                std::make_pair(parent_record->FileReferenceNumber, path_usn_record_pair)
            );
        }
        while (true);
    }

    static std::wstring colon_wstr(L":");
    static std::u16string colon(colon_wstr.begin(), colon_wstr.end());
    std::string drive;
    drive += drive_letter_;
    auto&& w_drive = string2wstring(drive);
    const std::u16string drive_u16(w_drive.begin(), w_drive.end());
    full_path = drive_u16 + colon + sep + full_path;

    if (remain_cache_number > temp_usn_cache.size())
    {
        save_usn_cache_to_map(temp_usn_cache);
    }
    if ((record->FileAttributes & FILE_ATTRIBUTE_DIRECTORY) && remain_cache_number > 0)
    {
        auto&& path_using_count_pair = std::make_pair(full_path, GetTickCount64());
        auto&& path_usn_record_pair = std::make_pair(path_using_count_pair,
                                                     record->ParentFileReferenceNumber);
        frn_record_pfrn_map_.insert(
            std::make_pair(record->FileReferenceNumber, path_usn_record_pair)
        );
        --remain_cache_number;
    }
}

/**
 * \brief 将usn_cache_map的usn记录缓存
 * \param usn_cache_map 需要缓存的usn record
 */
void NTFSChangesWatcher::save_usn_cache_to_map(const cache_map_t& usn_cache_map)
{
    for (const auto& [file_ref_number, path_usn_record] : usn_cache_map)
    {
        DWORDLONG file_parent_id = path_usn_record.second;
        std::u16string full_path(path_usn_record.first.first);
        do
        {
            auto&& parent_file = usn_cache_map.find(file_parent_id);
            if (parent_file == usn_cache_map.end())
            {
                break;
            }
            std::u16string file_name = parent_file->second.first.first;
            full_path = file_name.append(sep).append(full_path);
            file_parent_id = parent_file->second.second;
        }
        while (true);
        auto&& path_using_count_pair = std::make_pair(full_path, GetTickCount64());
        auto&& path_usn_record_pair = std::make_pair(path_using_count_pair,
                                                     path_usn_record.second);
        frn_record_pfrn_map_.insert(std::make_pair(file_ref_number, path_usn_record_pair));
        --remain_cache_number;
    }
}
