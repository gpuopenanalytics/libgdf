/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
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

#include "column_reader.h"
#include "file_reader.h"


static constexpr int NUM_ROWS_PER_ROW_GROUP = 500;

#ifndef PARQUET_FILE_FOR_DECODING_PATH
#error PARQUET_FILE_FOR_DECODING_PATH must be defined for precompiling
#define PARQUET_FILE_FOR_DECODING_PATH "/"
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

        {
            column = row_group->Column(0);
            gdf::parquet::Int32Reader *int32_reader =
                static_cast<gdf::parquet::Int32Reader *>(column.get());

            int64_t amountToRead = NUM_ROWS_PER_ROW_GROUP;
            std::vector<int32_t> valuesBuffer(amountToRead);

            std::vector<int16_t> dresult(amountToRead, -1);
            std::vector<int16_t> rresult(amountToRead,
                                        -1); // repetition levels must not be nullptr in order to avoid skipping null values
            std::vector<uint8_t> valid_bits(amountToRead, 255);

            int32_t val = valuesBuffer[0];

            int64_t rows_read_total = 0;
            while (rows_read_total < amountToRead) {
                int64_t rows_read =
                        int32_reader->ReadBatch(amountToRead,
                                                    dresult.data(),
                                                    rresult.data(),
                                                    (int32_t *) (&(valuesBuffer[rows_read_total])),
                                                    &values_read
                        );
                std::cout << "rows_read: " << rows_read << std::endl;
                rows_read_total += rows_read;
            }
            std::cout << "read values: \n";
            for(size_t i = 0; i < amountToRead; i++)
            {
                std::cout << valuesBuffer[i] << ",";
            }
            std::cout << "\n";
        }

        // {
        //     column = row_group->Column(1);
        //     gdf::parquet::DoubleReader *double_reader =
        //         static_cast<gdf::parquet::DoubleReader *>(column.get());

        //     int64_t amountToRead = NUM_ROWS_PER_ROW_GROUP;
        //     std::vector<double> valuesBuffer(amountToRead);

        //     std::vector<int16_t> dresult(amountToRead, -1);
        //     std::vector<int16_t> rresult(amountToRead,
        //                                 -1); // repetition levels must not be nullptr in order to avoid skipping null values
        //     std::vector<uint8_t> valid_bits(amountToRead, 255);

        //     int64_t rows_read_total = 0;
        //     while (rows_read_total < amountToRead) {
        //         int64_t rows_read =
        //                 double_reader->ReadBatch(amountToRead,
        //                                             dresult.data(),
        //                                             rresult.data(),
        //                                             (double *) (&(valuesBuffer[rows_read_total])),
        //                                             &values_read
        //                 );
        //         std::cout << "rows_read: " << rows_read << std::endl;
        //         rows_read_total += rows_read;
        //     }
        // }
    }
}


inline static std::shared_ptr<parquet::schema::GroupNode> createSchema() {
    parquet::schema::NodeVector fields{
            parquet::schema::PrimitiveNode::Make(
                    "int32_field", parquet::Repetition::REPEATED, parquet::Type::INT32,
                    parquet::LogicalType::NONE),

            parquet::schema::PrimitiveNode::Make(
                    "double_field", parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                    parquet::LogicalType::NONE)};

    return std::static_pointer_cast<parquet::schema::GroupNode>(
            parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED,
                                             fields));
}

TEST(DecodingTest, UsingCustomDecoder) {
    std::string filename = PARQUET_FILE_FOR_DECODING_PATH;
	std::unique_ptr<gdf::parquet::FileReader> reader = gdf::parquet::FileReader::OpenFile(filename);
    checkRowGroups(reader);
}
