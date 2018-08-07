/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <arrow/io/file.h>

#include "file_reader.h"
#include "file_reader_contents.h"

namespace gdf {
namespace parquet {

std::unique_ptr<FileReader>
FileReader::OpenFile(const std::string &                path,
                     const ::parquet::ReaderProperties &properties) {
    FileReader *const reader = new FileReader();
    reader->parquetFileReader_.reset(new ::parquet::ParquetFileReader());

    std::shared_ptr<::arrow::io::ReadableFile> file;

    PARQUET_THROW_NOT_OK(
      ::arrow::io::ReadableFile::Open(path, properties.memory_pool(), &file));

    std::unique_ptr<::parquet::RandomAccessSource> source(
      new ::parquet::ArrowInputFile(file));

    std::unique_ptr<::parquet::ParquetFileReader::Contents> contents(
      new internal::FileReaderContents(std::move(source), properties));

    static_cast<internal::FileReaderContents *>(contents.get())
      ->ParseMetaData();

    reader->parquetFileReader_->Open(std::move(contents));

    return std::unique_ptr<FileReader>(reader);
}

std::shared_ptr<::parquet::RowGroupReader>
FileReader::RowGroup(int i) {
    return parquetFileReader_->RowGroup(i);
}

std::shared_ptr<::parquet::FileMetaData>
FileReader::metadata() const {
    return parquetFileReader_->metadata();
}

}  // namespace parquet
}  // namespace gdf
