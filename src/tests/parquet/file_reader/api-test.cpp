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

#include <gtest/gtest.h>

#ifndef PARQUET_FILE_PATH
#error PARQUET_FILE_PATH must be defined for precompiling
#define PARQUET_FILE_PATH "/"
#endif

class ParquetReaderAPITest : public testing::Test {
protected:
    void
    TearDown() final {
        for (std::size_t i = 0; i < columns_length; i++) {
            gdf_column *column = &columns[i];

            delete[] static_cast<std::uint8_t *>(column->data);
            delete[] column->valid;
        }
        delete[] columns;
    }

    void
    checkBoolean(const gdf_column &boolean_column) {
        for (std::size_t i = 0; i < boolean_column.size; i++) {
            bool expected = (i % 2) == 0;
            bool value    = static_cast<bool *>(boolean_column.data)[i];

            EXPECT_EQ(expected, value);
        }
    }

    void
    checkInt64(const gdf_column &int64_column) {
        for (std::size_t i = 0; i < int64_column.size; i++) {
            std::int64_t expected =
              static_cast<std::int64_t>(i) * 1000000000000;
            std::int64_t value =
              static_cast<std::int64_t *>(int64_column.data)[i];

            EXPECT_EQ(expected, value);
        }
    }

    void
    checkDouble(const gdf_column &double_column) {
        for (std::size_t i = 0; i < double_column.size; i++) {
            double expected = i * 0.001;
            double value    = static_cast<double *>(double_column.data)[i];

            EXPECT_EQ(expected, value);
        }
    }

    gdf_column *columns        = nullptr;
    std::size_t columns_length = 0;
};

TEST_F(ParquetReaderAPITest, ReadAll) {
    gdf_error error_code = gdf::parquet::read_parquet(
      PARQUET_FILE_PATH, nullptr, nullptr, &columns, &columns_length);

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
      PARQUET_FILE_PATH, nullptr, column_names, &columns, &columns_length);

    EXPECT_EQ(GDF_SUCCESS, error_code);

    EXPECT_EQ(2, columns_length);

    checkDouble(columns[0]);
    checkInt64(columns[1]);
}
