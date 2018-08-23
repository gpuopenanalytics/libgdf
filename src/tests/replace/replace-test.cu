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

#include <gtest/gtest.h>

#include <thrust/device_vector.h>

#include <gdf/gdf.h>

template <class T>
static inline thrust::device_vector<T>
MakeDeviceVector(const std::initializer_list<T> list) {
    const std::vector<T>     column_data(list);
    thrust::device_vector<T> device_data(column_data);
    return device_data;
}

static inline gdf_column
MakeGdfColumn(thrust::device_vector<std::int64_t> &device_vector) {
    return gdf_column{
      .data       = thrust::raw_pointer_cast(device_vector.data()),
      .valid      = nullptr,
      .size       = device_vector.size(),
      .dtype      = GDF_INT64,
      .null_count = 0,
      .dtype_info = {},
    };
}

TEST(ReplaceTest, API) {
    thrust::device_vector<std::int64_t> device_data =
      MakeDeviceVector<std::int64_t>({1, 2, 3, 4, 5, 6, 7, 8});
    gdf_column column = MakeGdfColumn(device_data);

    thrust::device_vector<std::int64_t> to_replace_data =
      MakeDeviceVector<std::int64_t>({2, 4, 6, 8});
    thrust::device_vector<std::int64_t> values_data =
      MakeDeviceVector<std::int64_t>({0, 2, 4, 6});

    gdf_column to_replace = MakeGdfColumn(to_replace_data);
    gdf_column values     = MakeGdfColumn(values_data);

    const gdf_error status = gdf_replace(&column, &to_replace, &values);

    EXPECT_EQ(GDF_SUCCESS, status);

    thrust::device_ptr<std::int64_t> results(
      static_cast<std::int64_t *>(column.data));
    EXPECT_EQ(0, results[1]);
    EXPECT_EQ(2, results[3]);
    EXPECT_EQ(4, results[5]);
    EXPECT_EQ(6, results[7]);
}
