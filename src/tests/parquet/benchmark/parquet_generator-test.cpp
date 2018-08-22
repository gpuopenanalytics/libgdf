#include <limits>
#include <iostream>

#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>

#include <arrow/io/file.h>
#include <arrow/util/logging.h>

#include <parquet/api/reader.h>
#include <parquet/api/writer.h>

#include <gtest/gtest.h>

#ifndef PARQUET_FILE_PATH
#error PARQUET_FILE_PATH must be defined for precompiling
#define PARQUET_FILE_PATH "/"
#endif
 
static constexpr int NUM_ROWS_PER_ROW_GROUP = 100000;

static std::shared_ptr<parquet::schema::GroupNode> createSchema() {
    parquet::schema::NodeVector fields;
    // Create a primitive node named 'boolean_field' with type:BOOLEAN,
    // repetition:REQUIRED
    fields.push_back(parquet::schema::PrimitiveNode::Make("boolean_field", parquet::Repetition::REQUIRED,
                                                          parquet::Type::BOOLEAN, parquet::LogicalType::NONE));

    // Create a primitive node named 'int32_field' with type:INT32, repetition:REQUIRED,
    // logical type:TIME_MILLIS
    fields.push_back(
            parquet::schema::PrimitiveNode::Make("int32_field", parquet::Repetition::REQUIRED, parquet::Type::INT32,
                                                 parquet::LogicalType::TIME_MILLIS));

    // Create a primitive node named 'int64_field' with type:INT64, repetition:REPEATED
    fields.push_back(
            parquet::schema::PrimitiveNode::Make("int64_field", parquet::Repetition::REQUIRED, parquet::Type::INT64,
                                                 parquet::LogicalType::NONE));

    fields.push_back(
            parquet::schema::PrimitiveNode::Make("float_field", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
                                                 parquet::LogicalType::NONE));

    fields.push_back(
            parquet::schema::PrimitiveNode::Make("double_field", parquet::Repetition::REQUIRED, parquet::Type::DOUBLE,
                                                 parquet::LogicalType::NONE));

    // Create a GroupNode named 'schema' using the primitive nodes defined above
    // This GroupNode is the root node of the schema tree
    return std::static_pointer_cast<parquet::schema::GroupNode>(
            parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED, fields));
}


inline static void createParquetFile(std::string filename) {
    try {
        std::shared_ptr<arrow::io::FileOutputStream> stream;
        PARQUET_THROW_NOT_OK(
                arrow::io::FileOutputStream::Open(filename, &stream));

        std::shared_ptr<parquet::schema::GroupNode> schema = createSchema();

        parquet::WriterProperties::Builder builder;
        builder.compression(parquet::Compression::GZIP);
        std::shared_ptr<parquet::WriterProperties> props = builder.build();

        std::shared_ptr<parquet::ParquetFileWriter> file_writer =
                parquet::ParquetFileWriter::Open(stream, schema, props);

        parquet::RowGroupWriter *rg_writer =
                file_writer->AppendRowGroup(NUM_ROWS_PER_ROW_GROUP);


        {
            parquet::BoolWriter *bool_writer =
                    static_cast<parquet::BoolWriter *>(rg_writer->NextColumn());
            for (int i = 0; i < NUM_ROWS_PER_ROW_GROUP; i++) {
                bool value = ((i % 2) == 0) ? true : false;
                bool_writer->WriteBatch(1, nullptr, nullptr, &value);
            }
        }
        {
            parquet::Int32Writer *int32_writer =
                    static_cast<parquet::Int32Writer *>(rg_writer->NextColumn());
            for (int i = 0; i < NUM_ROWS_PER_ROW_GROUP; i++) {
                std::int16_t definition_level = 1;
                std::int16_t repetition_level = 0;
                std::int32_t value = i;
                if ((i % 2) == 0) {
                    repetition_level = 1;
                }
                if (i < 100) {
                    value = 100;
                } else if (i < 200) {
                    value = i;
                } else if (i < 300) {
                    value = 300;
                } else if (i < 400) {
                    value = i;
                } else if (i < 500) {
                    value = 500;
                }
                int32_writer->WriteBatch(1, nullptr, nullptr, &value);
            }
        }
        {
            // Write the Int64 column. Each row has repeats twice.
            parquet::Int64Writer *int64_writer =
                    static_cast<parquet::Int64Writer *>(rg_writer->NextColumn());
            for (int i = 0; i < NUM_ROWS_PER_ROW_GROUP; i++) {
                int64_t value = i * 1000 * 1000;
                value *= 1000 * 1000;
                std::int16_t definition_level = 1;
                std::int16_t repetition_level = 0;
                if ((i % 2) == 0) {
                    repetition_level = 1;  // start of a new record
                }
                int64_writer->WriteBatch(1, nullptr, nullptr, &value);
            }
        }
        {
            // Write the Float column
            parquet::FloatWriter *float_writer =
                    static_cast<parquet::FloatWriter *>(rg_writer->NextColumn());
            for (int i = 0; i < NUM_ROWS_PER_ROW_GROUP; i++) {
                float value = i * 1.1f;
                float_writer->WriteBatch(1, nullptr, nullptr, &value);
            }

        }
        {
            parquet::DoubleWriter *double_writer =
                    static_cast<parquet::DoubleWriter *>(rg_writer->NextColumn());
            for (int i = 0; i < NUM_ROWS_PER_ROW_GROUP; i++) {
                double value = i * 0.001;
                double_writer->WriteBatch(1, nullptr, nullptr, &value);
            }
        }

        file_writer->Close();

        DCHECK(stream->Close().ok());
    }
    catch (const std::exception &e) {
        std::cerr << "Parquet write error: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

TEST(ParquetGeneratorTest, GenHugeDataSet) {
    createParquetFile(PARQUET_FILE_PATH);
}