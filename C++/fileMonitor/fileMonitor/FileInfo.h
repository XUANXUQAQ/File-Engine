// This file is the part of the Indexer++ project.
// Copyright (C) 2016 Anna Krykora <krykoraanna@gmail.com>. All rights reserved.
// Use of this source code is governed by a MIT-style license that can be found in the LICENSE file.

#pragma once

#include <string>
#include <Windows.h>
#include "Macros.h"

#pragma pack(push, 4)

// Main data structure that represents a file and its properties in NTFS volume. In NTFS there directory
// is also a file, so we do not need a separate abstraction for directories.

namespace indexer_common {

	class FileInfo {
	public:
		explicit FileInfo(char drive_letter);

		FileInfo(USN_RECORD& record, char drive_letter);

		NO_COPY(FileInfo)

			~FileInfo();


		// Updates FileInfo properties, which can be retrieved from USN |record|, such as name, attributes, parent ID.

		void UpdateFromRecord(USN_RECORD& record);


		// Unique file reference number (FRN).

		unsigned int ID;


		// Parent FRN.

		unsigned int ParentID;


		char DriveLetter;


		// The length of the filename.

		unsigned short NameLength;


		// Bitmask of the file attributes (hidden, system, readonly, read/write rights etc.).

		unsigned long FileAttributes;


		// The real size of the file is the size of the unnamed data attribute. This is the number that will appear in a
		// directory listing. Converted and stored from bytes to KB.

		int SizeReal;


		// The allocated size of a file is the amount of disk space the file is taking up. It will be a multiple of the
		// cluster size.
		// Not used right now, so removed to decrease memory consumption.
		// uint64 SizeAllocated;


		// All timestamps stored in the Unix time, defined as the number of seconds that have elapsed since
		// 00:00:00 UTC, Thu, 1 Jan 1970.

		unsigned int CreationTime;

		unsigned int LastAccessTime;

		unsigned int LastWriteTime;


		// Parent in the index tree.

		FileInfo* Parent;


		// First child in the index tree.

		FileInfo* FirstChild;


		// Previous sibling in the index tree.

		FileInfo* PrevSibling;


		//  Next sibling in the index tree.

		FileInfo* NextSibling;


		// Returns file name.

		const char16_t* GetName() const {
			return name_;
		}


		// Sets the name of the file taking the ownership of |name|. If name copy needed call CopyAndSetName function.

		void SetName(const char16_t* name, unsigned short name_length);


		// Makes a copy of |name| string and sets the name of the file with that copy.

		void CopyAndSetName(const char16_t* name, unsigned short name_length);


		// Returns true if the FileInfo object represents a directory.

		bool IsDirectory() const;


		// Returns true if the file is hidden. TODO: determine more precise what hidden means.

		bool IsHiddenOrSystem() const;


	private:
		// The name of the file.
		const char16_t* name_;
	};

#pragma pack(pop)

} // namespace indexer_common