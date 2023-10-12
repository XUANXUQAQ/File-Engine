#pragma once
#include <memory>
#include <string>
#include <Windows.h>
#include <unordered_map>

using cache_map_t = std::unordered_map<DWORDLONG, std::pair<std::pair<std::u16string, DWORDLONG>, DWORDLONG>>;

class NTFSChangesWatcher
{
public:
    NTFSChangesWatcher(char drive_letter);
    ~NTFSChangesWatcher();

    // Method which runs an infinite loop and waits for new update sequence number in a journal.
    // The thread is blocked till the new USN record created in the journal.
    void WatchChanges(void (*)(const std::u16string&), void (*)(const std::u16string&));

    void StopWatch();

    void DeleteUsnOnExit();

private:
    static HANDLE OpenVolume(char drive_letter);

    static bool CreateJournal(HANDLE volume);

    static bool LoadJournal(HANDLE volume, USN_JOURNAL_DATA* journal_data);

    bool DeleteJournal() const;

    bool WaitForNextUsn(PREAD_USN_JOURNAL_DATA read_journal_data) const;

    std::unique_ptr<READ_USN_JOURNAL_DATA> GetWaitForNextUsnQuery(USN start_usn) const;

    bool ReadJournalRecords(PREAD_USN_JOURNAL_DATA journal_query, LPVOID buffer,
                            DWORD& byte_count) const;

    USN ReadChangesAndNotify(USN low_usn, char* buffer, void (*)(const std::u16string&),
                             void (*)(const std::u16string&));

    std::unique_ptr<READ_USN_JOURNAL_DATA> GetReadJournalQuery(USN low_usn) const;

    void showRecord(std::u16string& full_path, USN_RECORD* record);

    static std::u16string GetFilename(USN_RECORD* record);

    void save_usn_cache_to_map(const cache_map_t& usn_cache_map);

    char drive_letter_;

    volatile bool is_delete_usn_on_exit_;

    HANDLE volume_;

    volatile bool stop_flag;

    cache_map_t frn_record_pfrn_map_;

    std::unique_ptr<USN_JOURNAL_DATA> journal_;

    DWORDLONG journal_id_;

    USN last_usn_;

    USN max_usn_;

    // Flags, which indicate which types of changes you want to listen.
    static const int FILE_CHANGE_BITMASK;

    static const int kBufferSize;
};
