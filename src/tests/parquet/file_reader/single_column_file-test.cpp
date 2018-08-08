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
#include <parquet/file/writer.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/types.h>

#include <gtest/gtest.h>

#include "column_reader.h"
#include "file_reader.h"

#include <gdf/gdf.h>

template <class DataType>
class SingleColumnFileTest : public ::testing::Test {
protected:
    using TYPE = typename DataType::c_type;

    SingleColumnFileTest();

    void GenerateFile();
    TYPE GenerateValue(std::size_t i);

    virtual void SetUp() override;
    virtual void TearDown() override;

    static constexpr std::size_t kRowsPerGroup = 100;

    const std::string filename;

private:
    std::shared_ptr<::parquet::schema::GroupNode> CreateSchema();
};

using Types = ::testing::Types<::parquet::BooleanType,
                               ::parquet::Int32Type,
                               ::parquet::Int64Type,
                               ::parquet::FloatType,
                               ::parquet::DoubleType>;
TYPED_TEST_CASE(SingleColumnFileTest, Types);

template <class DataType>
void
SingleColumnFileTest<DataType>::SetUp() {
    GenerateFile();
}

template <class DataType>
void
SingleColumnFileTest<DataType>::TearDown() {
    if (std::remove(filename.c_str())) { FAIL() << "Remove file"; }
}

template <class DataType>
SingleColumnFileTest<DataType>::SingleColumnFileTest()
  : filename(boost::filesystem::unique_path().native()) {}

template <class DataType>
void
SingleColumnFileTest<DataType>::GenerateFile() {
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

        ::parquet::RowGroupWriter *row_group_writer =
          file_writer->AppendRowGroup(kRowsPerGroup);

        ::parquet::TypedColumnWriter<DataType> *writer =
          static_cast<::parquet::TypedColumnWriter<DataType> *>(
            row_group_writer->NextColumn());
        std::int16_t repetition_level = 0;
        for (std::size_t i = 0; i < kRowsPerGroup; i++) {
            TYPE         value            = GenerateValue(i);
            std::int16_t definition_level = i % 2 ? 1 : 0;
            writer->WriteBatch(
              1, &definition_level, &repetition_level, &value);
        }

        file_writer->Close();

        DCHECK(stream->Close().ok());
    } catch (const std::exception &e) {
        FAIL() << "Generate file" << e.what();
    }
}

template <class DataType>
std::shared_ptr<::parquet::schema::GroupNode>
SingleColumnFileTest<DataType>::CreateSchema() {
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
typename SingleColumnFileTest<DataType>::TYPE
SingleColumnFileTest<DataType>::GenerateValue(std::size_t i) {
    return static_cast<TYPE>(i) * 1000000000000;
}

TYPED_TEST(SingleColumnFileTest, ReadAll) {
    std::unique_ptr<gdf::parquet::FileReader> reader =
      gdf::parquet::FileReader::OpenFile(this->filename);

    std::shared_ptr<gdf::parquet::ColumnReader<TypeParam>> column_reader =
      std::static_pointer_cast<gdf::parquet::ColumnReader<TypeParam>>(
        reader->RowGroup(0)->Column(0));

    ASSERT_TRUE(column_reader->HasNext());

    using value_type = typename TypeParam::c_type;

    const std::size_t rowsPerGroup = this->kRowsPerGroup;

    gdf_column column{
      .data       = new std::uint8_t[rowsPerGroup * sizeof(value_type)],
      .valid      = new std::uint8_t[rowsPerGroup],
      .size       = 0,
      .dtype      = GDF_invalid,
      .null_count = 0,
      .dtype_info = {},
    };
    std::int16_t definition_levels[rowsPerGroup];
    std::int16_t repetition_levels[rowsPerGroup];

    const std::size_t total_read =
      column_reader->ToGdfColumn(definition_levels, repetition_levels, column);

    EXPECT_EQ(rowsPerGroup, total_read);

    for (std::size_t i = 0; i < rowsPerGroup; i++) {
        value_type   expected = this->GenerateValue(i);
        std::int64_t value    = static_cast<value_type *>(column.data)[i];
        if (i % 2) { EXPECT_EQ(expected, value); }
    }

    delete[] static_cast<std::uint8_t *>(column.data);
    delete[] column.valid;
}
