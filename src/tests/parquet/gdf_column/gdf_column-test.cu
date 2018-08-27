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


 #include <gtest/gtest.h>

 #include <sstream>
 
 #include <boost/filesystem.hpp>
 
 #include <arrow/io/file.h>
 #include <arrow/util/logging.h>
 
 
 #include <parquet/column_writer.h>
 #include <parquet/file/writer.h>
 #include <parquet/properties.h>
 #include <parquet/schema.h>
 #include <parquet/types.h>
 
 #include <cassert>
 
 #include <thrust/functional.h>
 #include <thrust/device_ptr.h>
 
 #include "column_reader.h"
 #include "file_reader.h"
 
 #include "../../helper/utils.cuh"
 
 #ifndef PARQUET_FILE_FOR_DECODING_PATH
 #error PARQUET_FILE_FOR_DECODING_PATH must be defined for precompiling
 #define PARQUET_FILE_FOR_DECODING_PATH "/"
 #endif
 
 template <typename DataType>
 class SingleColumnToGdfTest : public ::testing::Test {
 protected:
     using TYPE = typename DataType::c_type;
 
     SingleColumnToGdfTest();
 
     void GenerateFile();
 
     inline TYPE GenerateValue(size_t i) {
         if (sizeof (TYPE) == 1  ) {
             return i % 2;
         } 
         return static_cast<TYPE>(i) * 10;
     }
 
     virtual void SetUp() override;
 
     virtual void TearDown() override;
 
     static constexpr size_t kRowsPerGroup = 50;
 
     const std::string filename;
 
 private:
     std::shared_ptr<::parquet::schema::GroupNode> CreateSchema();
 };
 
 using Types = ::testing::Types<::parquet::BooleanType,
                                ::parquet::Int32Type>;
 TYPED_TEST_CASE(SingleColumnToGdfTest, Types);
 
 template<typename DataType>
 void SingleColumnToGdfTest<DataType>::SetUp() {
     GenerateFile();
 }
 
 template<typename DataType>
 void SingleColumnToGdfTest<DataType>::TearDown() {
     if ( std::remove(filename.c_str())) {
         FAIL() << "Remove file";
     }
 }
 
 template<typename DataType>
 SingleColumnToGdfTest<DataType>::SingleColumnToGdfTest()
     : filename ( boost::filesystem::unique_path().native())
 {
 }
  
 template <class DataType>
 void SingleColumnToGdfTest<DataType>::GenerateFile() {
     try {
         std::shared_ptr<::arrow::io::FileOutputStream> stream;
         PARQUET_THROW_NOT_OK(
           ::arrow::io::FileOutputStream::Open(filename, &stream));
 
         std::shared_ptr<::parquet::schema::GroupNode> schema = CreateSchema();
 
         ::parquet::WriterProperties::Builder builder;
         builder.compression(::parquet::Compression::SNAPPY);
         std::shared_ptr<::parquet::WriterProperties> properties =
           builder.build();
 
         // Set ColumnDescriptor! =  3
 
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
 SingleColumnToGdfTest<DataType>::CreateSchema() {
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
 
 
 
 TYPED_TEST(SingleColumnToGdfTest, ReadAll) {
     
    using FileReader = gdf::parquet::FileReader;
    using ColumnReader = gdf::parquet::ColumnReader<TypeParam>;
    auto reader = FileReader::OpenFile(this->filename);
    auto row_group = reader->RowGroup(0);
    auto abstract_column_reader = row_group->Column(0);
    std::cout << "column_reader id : " << typeid(abstract_column_reader).name() << std::endl;
 
    auto column_reader = std::static_pointer_cast<ColumnReader>(abstract_column_reader);
 
     ASSERT_TRUE(column_reader->HasNext());
 
     using value_type = typename TypeParam::c_type;
 
     const std::size_t rowsPerGroup = this->kRowsPerGroup;
 
     gdf_column column{
       .data       = nullptr,
       .valid      = nullptr,
       .size       = rowsPerGroup,
       .dtype      = GDF_invalid,
       .null_count = 0,
       .dtype_info = {},
     };
     cudaMalloc(&column.data, rowsPerGroup * sizeof(value_type));
     
     auto n_bytes = get_number_of_bytes_for_valid(this->kRowsPerGroup);
     cudaMalloc((void **)&column.valid, n_bytes);
 
     // std::int16_t definition_levels[rowsPerGroup];
     // std::int16_t repetition_levels[rowsPerGroup];
 
     const std::size_t total_read =
       column_reader->ToGdfColumn(column);
 
     column.size = static_cast<gdf_size_type>(rowsPerGroup); 
    //  column.dtype = ParquetTraits<TypeParam>::gdfDType;
 
     EXPECT_EQ(rowsPerGroup, total_read); // using ReadBatch
 
     print_column<value_type>(&column);
     
     gdf_column host_column = convert_to_host_gdf_column<value_type>(&column);
 
     for (std::size_t i = 0; i < rowsPerGroup; i++) {
         if (i % 2) { 
             value_type   expected = this->GenerateValue(i);
             value_type   value    = static_cast<value_type *>(host_column.data)[i];
             EXPECT_EQ(expected, value); 
         }
     }
 
     delete_gdf_column(&column);
 }
  