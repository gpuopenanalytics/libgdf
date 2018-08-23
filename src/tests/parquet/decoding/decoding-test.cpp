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
checkMetadata(const std::shared_ptr<const parquet::FileMetaData> &metadata)
{
    EXPECT_EQ(1, metadata->num_row_groups());
    EXPECT_EQ(3, metadata->num_columns());
}

void checkBoolValues(const std::shared_ptr<parquet::RowGroupReader> row_group)
{

    std::int64_t levels_read;
    std::int64_t values_read = 0;
    std::int64_t nulls_count;

    std::shared_ptr<parquet::ColumnReader> column;

    column = row_group->Column(0);
    gdf::parquet::BoolReader *_reader =
        static_cast<gdf::parquet::BoolReader *>(column.get());

    int64_t amountToRead = NUM_ROWS_PER_ROW_GROUP;
    std::vector<int8_t> valuesBuffer(amountToRead);

    std::vector<int16_t> dresult(amountToRead, -1);
    std::vector<int16_t> rresult(amountToRead,
                                 -1); // repetition levels must not be nullptr in order to avoid skipping null values
    std::vector<uint8_t> valid_bits(amountToRead, 255);

    int8_t val = valuesBuffer[0];

    int64_t rows_read_total = 0;
    while (rows_read_total < amountToRead)
    {
        int64_t rows_read =
            _reader->ReadBatchSpaced(amountToRead,
                                     dresult.data(),
                                     rresult.data(),
                                     (bool *)(&(valuesBuffer[rows_read_total])),
                                     valid_bits.data(),
                                     0,
                                     &levels_read,
                                     &values_read,
                                     &nulls_count);
        
        rows_read_total += rows_read;
    }
    
    for (size_t i = 0; i < amountToRead; i++)
    {
        bool value = (bool)valuesBuffer[i];
        bool expected_value = ((i % 2) == 0) ? true : false;
        EXPECT_EQ(expected_value, value);
        //std::cout << (bool)valuesBuffer[i] << ",";
    }
    
}

void checkInt32Values(const std::shared_ptr<parquet::RowGroupReader> row_group)
{
    std::int64_t levels_read;
    std::int64_t values_read = 0;
    std::int64_t nulls_count;

    std::shared_ptr<parquet::ColumnReader> column;

    column = row_group->Column(1);
    gdf::parquet::Int32Reader *int32_reader =
        static_cast<gdf::parquet::Int32Reader *>(column.get());

    int64_t amountToRead = 500;
    std::vector<int32_t> valuesBuffer(amountToRead);

    std::vector<int16_t> dresult(amountToRead, -1);
    std::vector<int16_t> rresult(amountToRead,
                                 -1); // repetition levels must not be nullptr in order to avoid skipping null values
    std::vector<uint8_t> valid_bits(amountToRead, 255);

    int32_t val = valuesBuffer[0];

    int64_t rows_read_total = 0;
    while (rows_read_total < amountToRead)
    {
        int64_t rows_read =
            int32_reader->ReadBatchSpaced(amountToRead,
                                          dresult.data(),
                                          rresult.data(),
                                          (int32_t *)(&(valuesBuffer[rows_read_total])),
                                          valid_bits.data(),
                                          0,
                                          &levels_read,
                                          &values_read,
                                          &nulls_count);
        // 
        rows_read_total += rows_read;
    }
    
    for (size_t i = 0; i < amountToRead; i++)
    {
        //std::cout << valuesBuffer[i] << ",";
        int32_t expected_value;
        if (i < 100)
        {
            expected_value = 100;
        }
        else if (i < 200)
        {
            expected_value = i;
        }
        else if (i < 300)
        {
            expected_value = 300;
        }
        else if (i < 400)
        {
            expected_value = i;
        }
        else if (i < 500)
        {
            expected_value = 500;
        }
        EXPECT_EQ(expected_value, valuesBuffer[i]);
    }
    
}

void checkInt64Values(const std::shared_ptr<parquet::RowGroupReader> row_group)
{
    std::int64_t levels_read;
    std::int64_t values_read = 0;
    std::int64_t nulls_count;

    std::shared_ptr<parquet::ColumnReader> column;

    column = row_group->Column(2);
    gdf::parquet::Int64Reader *_reader =
        static_cast<gdf::parquet::Int64Reader *>(column.get());

    int64_t amountToRead = NUM_ROWS_PER_ROW_GROUP;
    std::vector<int64_t> valuesBuffer(amountToRead);

    std::vector<int16_t> dresult(amountToRead, -1);
    std::vector<int16_t> rresult(amountToRead,
                                 -1); // repetition levels must not be nullptr in order to avoid skipping null values
    std::vector<uint8_t> valid_bits(amountToRead, 255);

    int64_t rows_read_total = 0;
    while (rows_read_total < amountToRead)
    {
        int64_t rows_read =
            _reader->ReadBatchSpaced(amountToRead,
                                     dresult.data(),
                                     rresult.data(),
                                     (int64_t *)(&(valuesBuffer[rows_read_total])),
                                     valid_bits.data(),
                                     0,
                                     &levels_read,
                                     &values_read,
                                     &nulls_count);
        
        rows_read_total += rows_read;
    }
    
    for (size_t i = 0; i < amountToRead; i++)
    {
        int64_t value = i * 1000 * 1000;
        value *= 1000 * 1000;
        EXPECT_EQ(value, valuesBuffer[i]);
    }
    
}
void checkFloatValues(const std::shared_ptr<parquet::RowGroupReader> row_group)
{
    std::int64_t levels_read;
    std::int64_t values_read = 0;
    std::int64_t nulls_count;

    std::shared_ptr<parquet::ColumnReader> column;

    column = row_group->Column(3);
    gdf::parquet::FloatReader *_reader =
        static_cast<gdf::parquet::FloatReader *>(column.get());

    int64_t amountToRead = NUM_ROWS_PER_ROW_GROUP;
    std::vector<float> valuesBuffer(amountToRead);

    std::vector<int16_t> dresult(amountToRead, -1);
    std::vector<int16_t> rresult(amountToRead,
                                 -1); // repetition levels must not be nullptr in order to avoid skipping null values
    std::vector<uint8_t> valid_bits(amountToRead, 255);

    int64_t rows_read_total = 0;
    while (rows_read_total < amountToRead)
    {
        int64_t rows_read =
            _reader->ReadBatchSpaced(amountToRead,
                                     dresult.data(),
                                     rresult.data(),
                                     (float *)(&(valuesBuffer[rows_read_total])),
                                     valid_bits.data(),
                                     0,
                                     &levels_read,
                                     &values_read,
                                     &nulls_count);
        
        rows_read_total += rows_read;
    }
    for (size_t i = 0; i < amountToRead; i++)
    {
        float value = i * 1.1f;
        EXPECT_EQ(value, valuesBuffer[i]);
    }
}

void checkDoubleValues(const std::shared_ptr<parquet::RowGroupReader> row_group)
{
    std::int64_t levels_read;
    std::int64_t values_read = 0;
    std::int64_t nulls_count;

    std::shared_ptr<parquet::ColumnReader> column;

    column = row_group->Column(4);
    gdf::parquet::DoubleReader *double_reader =
        static_cast<gdf::parquet::DoubleReader *>(column.get());

    int64_t amountToRead = NUM_ROWS_PER_ROW_GROUP;
    std::vector<double> valuesBuffer(amountToRead);

    std::vector<int16_t> dresult(amountToRead, -1);
    std::vector<int16_t> rresult(amountToRead,
                                 -1); // repetition levels must not be nullptr in order to avoid skipping null values
    std::vector<uint8_t> valid_bits(amountToRead, 255);

    int64_t rows_read_total = 0;
    while (rows_read_total < amountToRead)
    {
        int64_t rows_read =
            double_reader->ReadBatchSpaced(amountToRead,
                                           dresult.data(),
                                           rresult.data(),
                                           (double *)(&(valuesBuffer[rows_read_total])),
                                           valid_bits.data(),
                                           0,
                                           &levels_read,
                                           &values_read,
                                           &nulls_count);
        
        rows_read_total += rows_read;
    }
    
    for (size_t i = 0; i < amountToRead; i++)
    {
        double value = i * 0.001;

        EXPECT_EQ(value, valuesBuffer[i]);
    }
    
}

template<class Functor>
inline static void
checkRowGroups(const std::unique_ptr<gdf::parquet::FileReader> &reader, Functor apply)
{
	int numRowGroups = reader->metadata()->num_row_groups();
    for (int r = 0; r < numRowGroups; ++r)
    {
        const std::shared_ptr<parquet::RowGroupReader> row_group =
            reader->RowGroup(r);

        std::int64_t levels_read;
        std::int64_t values_read = 0;
        std::int64_t nulls_count;

        int i;
        std::shared_ptr<parquet::ColumnReader> column;

        apply(row_group);
     }
}

TEST(DecodingTest, ReadBoolValues)
{
    std::string filename = PARQUET_FILE_FOR_DECODING_PATH;
    std::unique_ptr<gdf::parquet::FileReader> reader = gdf::parquet::FileReader::OpenFile(filename);
    
    checkRowGroups(reader, checkBoolValues);
}

TEST(DecodingTest, ReadInt32Values)
{
    std::string filename = PARQUET_FILE_FOR_DECODING_PATH;
    std::unique_ptr<gdf::parquet::FileReader> reader = gdf::parquet::FileReader::OpenFile(filename);
    checkRowGroups(reader, checkInt32Values);
}

TEST(DecodingTest, ReadInt64Values)
{
    std::string filename = PARQUET_FILE_FOR_DECODING_PATH;
    std::unique_ptr<gdf::parquet::FileReader> reader = gdf::parquet::FileReader::OpenFile(filename);
    checkRowGroups(reader, checkInt64Values);
}

TEST(DecodingTest, ReadFloatValues)
{
    std::string filename = PARQUET_FILE_FOR_DECODING_PATH;
    std::unique_ptr<gdf::parquet::FileReader> reader = gdf::parquet::FileReader::OpenFile(filename);
    checkRowGroups(reader, checkFloatValues);
}

TEST(DecodingTest, ReadDoubleValues)
{
    std::string filename = PARQUET_FILE_FOR_DECODING_PATH;
    std::unique_ptr<gdf::parquet::FileReader> reader = gdf::parquet::FileReader::OpenFile(filename);
    checkRowGroups(reader, checkDoubleValues);   
}
