#pragma once
#include <memory>
#include <string>
#include <Windows.h>
#include <unordered_map>

class NTFSChangesWatcher
{
public:
	NTFSChangesWatcher(char drive_letter);
	~NTFSChangesWatcher();

	// Method which runs an infinite loop and waits for new update sequence number in a journal.
	// The thread is blocked till the new USN record created in the journal.
	void WatchChanges(void(*)(const std::u16string&), void(*)(const std::u16string&));

	bool DeleteJournal() const;

	void stopWatch();

	bool isStopWatch() const
	{
		return is_stopped;
	}

private:
	static HANDLE OpenVolume(char drive_letter);

	static bool CreateJournal(HANDLE volume);

	bool LoadJournal(HANDLE volume, USN_JOURNAL_DATA* journal_data);

	bool WaitForNextUsn(PREAD_USN_JOURNAL_DATA read_journal_data) const;

	std::unique_ptr<READ_USN_JOURNAL_DATA> GetWaitForNextUsnQuery(USN start_usn) const;

	bool ReadJournalRecords(PREAD_USN_JOURNAL_DATA journal_query, LPVOID buffer,
		DWORD& byte_count) const;

	USN ReadChangesAndNotify(USN low_usn, char* buffer, void(*)(const std::u16string&), void(*)(const std::u16string&));

	std::unique_ptr<READ_USN_JOURNAL_DATA> GetReadJournalQuery(USN low_usn) const;

	void showRecord(std::u16string& full_path, USN_RECORD* record);

	static std::u16string GetFilename(USN_RECORD* record);

	char drive_letter_;

	HANDLE volume_;

	bool stop_flag;

	bool is_stopped;

	std::unordered_map<DWORDLONG, std::pair<std::pair<std::u16string, DWORDLONG>, DWORDLONG>> frn_record_pfrn_map_;

	std::unique_ptr<USN_JOURNAL_DATA> journal_;

	DWORDLONG journal_id_;

	USN last_usn_;

	USN max_usn_;

	// Flags, which indicate which types of changes you want to listen.
	static const int FILE_CHANGE_BITMASK;

	static const int kBufferSize;
};
