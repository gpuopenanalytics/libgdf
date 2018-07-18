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
#include <arrow/util/logging.h>
#include <parquet/api/reader.h>
#include <parquet/api/writer.h>

#include <gtest/gtest.h>

#include <gdf/parquet/column_reader.h>
#include <gdf/parquet/file_reader.h>

#ifndef PARQUET_FILE_PATH
#error PARQUET_FILE_PATH must be defined for precompiling
#define PARQUET_FILE_PATH "/"
#endif

inline static void
checkMetadata(const std::shared_ptr<const parquet::FileMetaData> &metadata) {
    EXPECT_EQ(1, metadata->num_row_groups());
    EXPECT_EQ(3, metadata->num_columns());
}

inline static void
checkRowGroups(const std::unique_ptr<gdf::parquet::FileReader> &reader) {
    for (int r = 0; r < reader->metadata()->num_row_groups(); ++r) {
        const std::shared_ptr<parquet::RowGroupReader> row_group =
          reader->RowGroup(r);

        std::int64_t                           values_read = 0;
        int                                    i;
        std::shared_ptr<parquet::ColumnReader> column;

        column = row_group->Column(0);
        gdf::parquet::BoolReader *bool_reader =
          static_cast<gdf::parquet::BoolReader *>(column.get());
        i = 0;
        while (bool_reader->HasNext()) {
            bool         value;
            std::int64_t rows_read = bool_reader->ReadBatch(
              1, nullptr, nullptr, &value, &values_read);
            EXPECT_EQ(1, rows_read);
            EXPECT_EQ(1, values_read);
            bool expected = (i % 2) == 0;
            EXPECT_EQ(expected, value);
            i++;
        }

        column = row_group->Column(1);
        gdf::parquet::Int64Reader *int64_reader =
          static_cast<gdf::parquet::Int64Reader *>(column.get());
        i = 0;
        while (int64_reader->HasNext()) {
            std::int64_t value;
            std::int16_t definition_level;
            std::int16_t repetition_level;
            std::int64_t rows_read = int64_reader->ReadBatch(
              1, &definition_level, &repetition_level, &value, &values_read);
            EXPECT_EQ(1, rows_read);
            EXPECT_EQ(1, values_read);
            std::int64_t expected =
              static_cast<std::int64_t>(i) * 1000 * 1000 * 1000 * 1000;
            EXPECT_EQ(expected, value);
            EXPECT_EQ(static_cast<std::int16_t>((i % 2) == 0),
                      repetition_level);
            i++;
        }

        column = row_group->Column(2);
        gdf::parquet::DoubleReader *double_reader =
          static_cast<gdf::parquet::DoubleReader *>(column.get());
        i = 0;
        while (double_reader->HasNext()) {
            double       value;
            std::int64_t rows_read = double_reader->ReadBatch(
              1, nullptr, nullptr, &value, &values_read);
            EXPECT_EQ(1, rows_read);
            EXPECT_EQ(1, values_read);
            double expected = i * 0.001;
            EXPECT_EQ(expected, value);
            i++;
        }
    }
}

TEST(FileReaderTest, Test) {
    std::unique_ptr<gdf::parquet::FileReader> reader =
      gdf::parquet::FileReader::OpenFile(PARQUET_FILE_PATH);

    checkMetadata(reader->metadata());
    checkRowGroups(reader);
}
