#include "NTFSChangesWatcher.h"

#include "FileInfo.h"
#include "string_wstring_converter.h"

const int NTFSChangesWatcher::kBufferSize = 1024 * 1024 / 2;

const int NTFSChangesWatcher::FILE_CHANGE_BITMASK = USN_REASON_RENAME_NEW_NAME | USN_REASON_RENAME_OLD_NAME;


NTFSChangesWatcher::NTFSChangesWatcher(char drive_letter) :
	drive_letter_(drive_letter)
{
	volume_ = OpenVolume(drive_letter_);

	journal_ = std::make_unique<USN_JOURNAL_DATA>();

	if (const bool res = LoadJournal(volume_, journal_.get()); !res) {
		fprintf(stderr, "Failed to load journal");
		return;
	}
	max_usn_ = journal_->MaxUsn;
	journal_id_ = journal_->UsnJournalID;
	last_usn_ = journal_->NextUsn;
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

	if (volume == INVALID_HANDLE_VALUE) {
		// An error occurred!
		fprintf(stderr, "Failed to open volume");
		return nullptr;
	}

	return volume;
}


bool NTFSChangesWatcher::CreateJournal(HANDLE volume)
{

	DWORD byte_count;
	CREATE_USN_JOURNAL_DATA create_journal_data;

	const bool ok = DeviceIoControl(volume, // handle to volume
		FSCTL_CREATE_USN_JOURNAL,     // dwIoControlCode
		&create_journal_data,         // input buffer
		sizeof(create_journal_data),  // size of input buffer
		nullptr,                         // lpOutBuffer
		0,                            // nOutBufferSize
		&byte_count,                  // number of bytes returned
		nullptr) != 0;                   // OVERLAPPED structure

	if (!ok) {
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

		if (CreateJournal(volume)) {
			return LoadJournal(volume, journal_data);
		}

		return false;
	}

	return true;
}

void NTFSChangesWatcher::WatchChanges(const bool* flag,
	void(*file_added_callback_func)(const std::u16string&),
	void(*file_removed_callback_func)(const std::u16string&))
{
	const auto u_buffer = std::make_unique<char[]>(kBufferSize);

	const auto read_journal_query = GetWaitForNextUsnQuery(last_usn_);

	while (*flag) 
	{
		// This function does not return until new USN record created.
		WaitForNextUsn(read_journal_query.get());
		last_usn_ = ReadChangesAndNotify(read_journal_query->StartUsn,
			u_buffer.get(),
			file_added_callback_func,
			file_removed_callback_func);
		read_journal_query->StartUsn = last_usn_;
	}
	delete flag;
}

USN NTFSChangesWatcher::ReadChangesAndNotify(USN low_usn,
	char* buffer,
	void(*file_added_callback_func)(const std::u16string&),
	void(*file_removed_callback_func)(const std::u16string&))
{

	DWORD byte_count;

	const auto journal_query = GetReadJournalQuery(low_usn);
	memset(buffer, 0, kBufferSize);
	if (!ReadJournalRecords(journal_query.get(), buffer, byte_count))
	{
		// An error occurred.
		fprintf(stderr, "Failed to read journal records");
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
			file_added_callback_func(full_path);
		}
		else if ((reason & USN_REASON_FILE_DELETE) && (reason & USN_REASON_CLOSE)) 
		{
			showRecord(full_path, record);
			file_removed_callback_func(full_path);
		}
		else if (reason & FILE_CHANGE_BITMASK) 
		{
			if (reason & USN_REASON_RENAME_OLD_NAME)
			{
				showRecord(full_path, record);
				file_removed_callback_func(full_path);
			}
			else if (reason & USN_REASON_RENAME_NEW_NAME)
			{
				showRecord(full_path, record);
				file_added_callback_func(full_path);
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

std::unique_ptr<READ_USN_JOURNAL_DATA> NTFSChangesWatcher::GetWaitForNextUsnQuery(USN start_usn)
{

	auto query = std::make_unique<READ_USN_JOURNAL_DATA>();

	query->StartUsn = start_usn;
	query->ReasonMask = 0xFFFFFFFF;     // All bits.
	query->ReturnOnlyOnClose = FALSE;   // All entries.
	query->Timeout = 0;                 // No timeout.
	query->BytesToWaitFor = 1;          // Wait for this.
	query->UsnJournalID = journal_id_;  // The journal.
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

std::unique_ptr<READ_USN_JOURNAL_DATA> NTFSChangesWatcher::GetReadJournalQuery(USN low_usn)
{

	auto query = std::make_unique<READ_USN_JOURNAL_DATA>();

	query->StartUsn = low_usn;
	query->ReasonMask = 0xFFFFFFFF;  // All bits.
	query->ReturnOnlyOnClose = FALSE;
	query->Timeout = 0;  // No timeout.
	query->BytesToWaitFor = 0;
	query->UsnJournalID = journal_id_;
	query->MinMajorVersion = 2;
	query->MaxMajorVersion = 2;

	return query;
}


void NTFSChangesWatcher::showRecord(std::u16string& full_path, USN_RECORD* record)
{
	static std::wstring sep_wstr(L"\\");
	static std::u16string sep(sep_wstr.begin(), sep_wstr.end());

	const indexer_common::FileInfo file_info(*record, drive_letter_);
	if (full_path.empty())
	{
		full_path += file_info.GetName();
	}
	else
	{
		full_path = file_info.GetName() + sep + full_path;
	}
	DWORD byte_count = 1;
	auto buffer = std::make_unique<char[]>(kBufferSize);

	MFT_ENUM_DATA_V0 med;
	med.StartFileReferenceNumber = record->ParentFileReferenceNumber;
	med.LowUsn = 0;
	med.HighUsn = max_usn_;

	if (!DeviceIoControl(volume_,
		FSCTL_ENUM_USN_DATA,
		&med,
		sizeof(med),
		buffer.get(),
		kBufferSize,
		&byte_count,
		nullptr))
	{
		fprintf(stderr, "FSCTL_ENUM_USN_DATA (showRecord): %lu\n", GetLastError());
		return;
	}

	auto* parent_record = reinterpret_cast<USN_RECORD*>(reinterpret_cast<USN*>(buffer.get()) + 1);

	if (parent_record->FileReferenceNumber != record->ParentFileReferenceNumber)
	{
		static std::wstring colon_wstr(L":");
		static std::u16string colon(colon_wstr.begin(), colon_wstr.end());
		std::string drive;
		drive += drive_letter_;
		auto&& w_drive = string2wstring(drive);
		const std::u16string drive_u16(w_drive.begin(), w_drive.end());
		full_path = drive_u16 + colon + sep + full_path;
		return;
	}
	showRecord(full_path, parent_record);
}