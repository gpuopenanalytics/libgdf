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

#include <sstream>

#include <boost/filesystem.hpp>

#include <arrow/io/file.h>
#include <arrow/util/logging.h>

#include <parquet/column_writer.h>
#include <parquet/file_writer.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/types.h>

#include <gtest/gtest.h>

#include "../../../parquet/column_reader.h"
#include "../../../parquet/file_reader.h"

#include <gdf/gdf.h>

#include "../../helper/utils.cuh"

template <class DataType>
class NullTest : public ::testing::Test {
protected:
    using TYPE = typename DataType::c_type;

    NullTest();

    void GenerateFile();
    TYPE GenerateValue(std::size_t i);

    virtual void SetUp() override;
    virtual void TearDown() override;

    static constexpr std::size_t kGroups       = 2;
    static constexpr std::size_t kRowsPerGroup = 50;

    const std::string filename;

private:
    std::shared_ptr<::parquet::schema::GroupNode> CreateSchema();
};

using Types = ::testing::Types<::parquet::Int64Type>;
TYPED_TEST_CASE(NullTest, Types);

template <class DataType>
void
NullTest<DataType>::SetUp() {
    GenerateFile();
}

template <class DataType>
void
NullTest<DataType>::TearDown() {
    if (std::remove(filename.c_str())) { FAIL() << "Remove file"; }
}

template <class DataType>
NullTest<DataType>::NullTest()
  : filename(boost::filesystem::unique_path().native()) {}

template <class DataType>
void
NullTest<DataType>::GenerateFile() {
    try {
        std::shared_ptr<::arrow::io::FileOutputStream> stream;
        PARQUET_THROW_NOT_OK(
          ::arrow::io::FileOutputStream::Open(filename, &stream));

        std::shared_ptr<::parquet::schema::GroupNode> schema = CreateSchema();

        ::parquet::WriterProperties::Builder builder;
        builder.compression(::parquet::Compression::SNAPPY);
        std::shared_ptr<::parquet::WriterProperties> properties =
          builder.build();

        std::shared_ptr<::parquet::ParquetFileWriter> file_writer =
          ::parquet::ParquetFileWriter::Open(stream, schema, properties);

        for (std::size_t i = 0; i < kGroups; i++) {
            ::parquet::RowGroupWriter *row_group_writer =
              file_writer->AppendRowGroup(kRowsPerGroup);

            ::parquet::TypedColumnWriter<DataType> *writer =
              static_cast<::parquet::TypedColumnWriter<DataType> *>(
                row_group_writer->NextColumn());
            std::int16_t repetition_level = 0;
            for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                TYPE         value = GenerateValue(i * kRowsPerGroup + j);
                std::int16_t definition_level = j % 2;
                writer->WriteBatch(
                  1, &definition_level, &repetition_level, &value);
            }
        }

        file_writer->Close();

        DCHECK(stream->Close().ok());
    } catch (const std::exception &e) { FAIL() << "Generate file" << e.what(); }
}

template <class DataType>
std::shared_ptr<::parquet::schema::GroupNode>
NullTest<DataType>::CreateSchema() {
    return std::static_pointer_cast<::parquet::schema::GroupNode>(
      ::parquet::schema::GroupNode::Make(
        "schema",
        ::parquet::Repetition::REQUIRED,
        ::parquet::schema::NodeVector{::parquet::schema::PrimitiveNode::Make(
          "field",
          ::parquet::Repetition::OPTIONAL,
          DataType::type_num,
          ::parquet::LogicalType::NONE)}));
}

template <class DataType>
typename NullTest<DataType>::TYPE
NullTest<DataType>::GenerateValue(std::size_t i) {
    return static_cast<TYPE>(i) * 10;
}

TYPED_TEST(NullTest, ReadAll) {
    std::unique_ptr<gdf::parquet::FileReader> reader =
      gdf::parquet::FileReader::OpenFile(this->filename);

    std::shared_ptr<gdf::parquet::ColumnReader<TypeParam>> column_reader =
      std::static_pointer_cast<gdf::parquet::ColumnReader<TypeParam>>(
        reader->RowGroup(0)->Column(0));

    ASSERT_TRUE(column_reader->HasNext());

    using value_type = typename TypeParam::c_type;

    const std::size_t rowsPerGroup = this->kRowsPerGroup;
    const std::size_t groups       = this->kGroups;

    gdf_column column{
      .data       = nullptr,
      .valid      = nullptr,
      .size       = 0,
      .dtype      = GDF_INT64,
      .null_count = 0,
      .dtype_info = {},
    };

    std::size_t valid_size =
      get_number_of_bytes_for_valid(rowsPerGroup * groups);

    cudaMalloc(&column.data, rowsPerGroup * groups * sizeof(value_type));
    cudaMalloc(&column.valid, valid_size);

    const std::size_t total_read = column_reader->ToGdfColumn(column);

    column_reader =
      std::static_pointer_cast<gdf::parquet::ColumnReader<TypeParam>>(
        reader->RowGroup(1)->Column(0));

    ASSERT_TRUE(column_reader->HasNext());
    const std::size_t total_read2 = column_reader->ToGdfColumn(column, 50);

    column.size = static_cast<gdf_size_type>(rowsPerGroup * groups);

    EXPECT_EQ(rowsPerGroup, total_read);

    gdf_column host_column = convert_to_host_gdf_column<value_type>(&column);

    for (std::size_t i = 0; i < groups * rowsPerGroup; i++) {
        value_type   expected = this->GenerateValue(i);
        std::int64_t value    = static_cast<value_type *>(host_column.data)[i];
        if (i % 2) { EXPECT_EQ(expected, value); }
    }

    const std::size_t valid_last = valid_size - 1;
    for (std::size_t i = 0; i < valid_last; i++) {
        std::uint8_t valid = host_column.valid[i];
        EXPECT_EQ(0b10101010, valid);
    }
    EXPECT_EQ(0b00001010, 0b00001010 & host_column.valid[valid_last]);

    delete_gdf_column(&column);
}
