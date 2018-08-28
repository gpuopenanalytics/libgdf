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

#include <gdf/parquet/api.h>

#include <arrow/io/file.h>
#include <arrow/util/logging.h>

#include <boost/filesystem.hpp>

#include <parquet/column_writer.h>
#include <parquet/file/writer.h>
#include <parquet/properties.h>
#include <parquet/schema.h>
#include <parquet/types.h>

#include <gtest/gtest.h>

#include "../../helper/utils.cuh"

class ParquetReaderAPITest : public testing::Test {
protected:
    ParquetReaderAPITest()
      : filename(boost::filesystem::unique_path().native()) {}

    void
    SetUp() final {
        static constexpr std::size_t kGroups       = 2;
        static constexpr std::size_t kRowsPerGroup = 50;
        try {
            std::shared_ptr<::arrow::io::FileOutputStream> stream;
            PARQUET_THROW_NOT_OK(
              ::arrow::io::FileOutputStream::Open(filename, &stream));

            std::shared_ptr<::parquet::schema::GroupNode> schema =
              CreateSchema();

            ::parquet::WriterProperties::Builder builder;
            builder.compression(::parquet::Compression::SNAPPY);
            std::shared_ptr<::parquet::WriterProperties> properties =
              builder.build();

            std::shared_ptr<::parquet::ParquetFileWriter> file_writer =
              ::parquet::ParquetFileWriter::Open(stream, schema, properties);

            std::int16_t repetition_level = 0;

            for (std::size_t i = 0; i < kGroups; i++) {
                ::parquet::RowGroupWriter *row_group_writer =
                  file_writer->AppendRowGroup(kRowsPerGroup);

                ::parquet::BoolWriter *bool_writer =
                  static_cast<::parquet::BoolWriter *>(
                    row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    std::int16_t definition_level = j % 2;
                    bool         bool_value       = true;
                    bool_writer->WriteBatch(
                      1, &definition_level, &repetition_level, &bool_value);
                }

                ::parquet::Int64Writer *int64_writer =
                  static_cast<::parquet::Int64Writer *>(
                    row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    std::int16_t definition_level = j % 2;
                    std::int64_t int64_value      = i * kRowsPerGroup + j;
                    int64_writer->WriteBatch(
                      1, &definition_level, &repetition_level, &int64_value);
                }

                ::parquet::DoubleWriter *double_writer =
                  static_cast<::parquet::DoubleWriter *>(
                    row_group_writer->NextColumn());
                for (std::size_t j = 0; j < kRowsPerGroup; j++) {
                    std::int16_t definition_level = j % 2;
                    double       double_value     = i * kRowsPerGroup + j;
                    double_writer->WriteBatch(
                      1, &definition_level, &repetition_level, &double_value);
                }
            }

            file_writer->Close();

            DCHECK(stream->Close().ok());
        } catch (const std::exception &e) {
            FAIL() << "Generate file" << e.what();
        }
    }

    std ::shared_ptr<::parquet::schema::GroupNode>
    CreateSchema() {
        return std::static_pointer_cast<::parquet::schema::GroupNode>(
          ::parquet::schema::GroupNode::Make(
            "schema",
            ::parquet::Repetition::REQUIRED,
            ::parquet::schema::NodeVector{
              ::parquet::schema::PrimitiveNode::Make(
                "boolean_field",
                ::parquet::Repetition::OPTIONAL,
                ::parquet::Type::BOOLEAN,
                ::parquet::LogicalType::NONE),
              ::parquet::schema::PrimitiveNode::Make(
                "int64_field",
                ::parquet::Repetition::OPTIONAL,
                ::parquet::Type::INT64,
                ::parquet::LogicalType::NONE),
              ::parquet::schema::PrimitiveNode::Make(
                "double_field",
                ::parquet::Repetition::OPTIONAL,
                ::parquet::Type::DOUBLE,
                ::parquet::LogicalType::NONE),
            }));
    }

    void
    TearDown() final {
        if (std::remove(filename.c_str())) { FAIL() << "Remove file"; }
    }

    void
    checkNulls(/*const */ gdf_column &column) {
        const std::size_t valid_size =
          arrow::BitUtil::BytesForBits(column.size);
        const std::size_t valid_last = valid_size - 1;
        for (std::size_t i = 0; i < valid_last; i++) {
            std::uint8_t valid = column.valid[i];
            EXPECT_EQ(0b10101010, valid);
        }
        EXPECT_EQ(0b00001010, 0b00001010 & column.valid[valid_last]);
    }

    void
    checkBoolean(/*const */ gdf_column &column) {
        gdf_column boolean_column =
          convert_to_host_gdf_column<::parquet::BooleanType::c_type>(&column);

        for (std::size_t i = 0; i < boolean_column.size; i++) {
            if (i % 2) {
                bool expected = true;
                bool value    = static_cast<bool *>(boolean_column.data)[i];

                EXPECT_EQ(expected, value);
            }

            checkNulls(boolean_column);
        }
    }

    void
    checkInt64(/*const */ gdf_column &column) {
        gdf_column int64_column =
          convert_to_host_gdf_column<::parquet::Int64Type::c_type>(&column);

        for (std::size_t i = 0; i < int64_column.size; i++) {
            if (i % 2) {
                std::int64_t expected = static_cast<std::int64_t>(i);
                std::int64_t value =
                  static_cast<std::int64_t *>(int64_column.data)[i];

                EXPECT_EQ(expected, value);
            }
        }

        checkNulls(int64_column);
    }

    void
    checkDouble(/*const */ gdf_column &column) {
        gdf_column double_column =
          convert_to_host_gdf_column<::parquet::DoubleType::c_type>(&column);

        for (std::size_t i = 0; i < double_column.size; i++) {
            if (i % 2) {
                double expected = static_cast<double>(i);
                double value    = static_cast<double *>(double_column.data)[i];

                EXPECT_EQ(expected, value);
            }
        }

        checkNulls(double_column);
    }

    const std::string filename;

    gdf_column *columns        = nullptr;
    std::size_t columns_length = 0;
};

TEST_F(ParquetReaderAPITest, ReadAll) {
    gdf_error error_code = gdf::parquet::read_parquet(
      filename.c_str(), nullptr, &columns, &columns_length);

    EXPECT_EQ(GDF_SUCCESS, error_code);

    EXPECT_EQ(3, columns_length);

    EXPECT_EQ(columns[0].size, columns[1].size);
    EXPECT_EQ(columns[1].size, columns[2].size);

    checkBoolean(columns[0]);
    checkInt64(columns[1]);
    checkDouble(columns[2]);
}

TEST_F(ParquetReaderAPITest, ReadSomeColumns) {
    const char *const column_names[] = {
      "double_field", "int64_field", nullptr};

    gdf_error error_code = gdf::parquet::read_parquet(
      filename.c_str(), column_names, &columns, &columns_length);

    EXPECT_EQ(GDF_SUCCESS, error_code);

    EXPECT_EQ(2, columns_length);

    checkDouble(columns[0]);
    checkInt64(columns[1]);
}

TEST_F(ParquetReaderAPITest, ByIdsInOrder) {
    const std::vector<std::size_t> row_group_indices = {0, 1};
    const std::vector<std::size_t> column_indices    = {0, 1, 2};

    std::vector<gdf_column *> columns;

    gdf_error error_code = gdf::parquet::read_parquet_by_ids(
      filename, row_group_indices, column_indices, columns);

    EXPECT_EQ(GDF_SUCCESS, error_code);

    EXPECT_EQ(3, columns.size());

    checkBoolean(*columns[0]);
    checkInt64(*columns[1]);
    checkDouble(*columns[2]);
}
