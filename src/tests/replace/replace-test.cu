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

#include <initializer_list>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <gdf/gdf.h>

static gdf_column
CreateGdfColumn(const std::initializer_list<std::int64_t> list) {
    const std::vector<std::int64_t>       host_data(list);
    thrust::device_vector<std::int64_t>   device_data(host_data);
    thrust::device_vector<gdf_valid_type> device_valid(1, 0);

    return gdf_column{
      .data       = thrust::raw_pointer_cast(device_data.data()),
      .valid      = thrust::raw_pointer_cast(device_valid.data()),
      .size       = 0,
      .dtype      = GDF_INT64,
      .null_count = 0,
      .dtype_info = {},
    };
}

TEST(ReplaceTest, API) {
    gdf_column column = CreateGdfColumn({1, 2, 3, 4, 5, 6, 7, 8});

    gdf_column to_replace = CreateGdfColumn({2, 4, 6, 8});
    gdf_column values     = CreateGdfColumn({0, 2, 4, 6});

    const gdf_error status = gdf_replace(&column, &to_replace, &values);

    EXPECT_EQ(GDF_SUCCESS, status);

    const thrust::device_ptr<std::int64_t> data_ptr(
      static_cast<std::int64_t *>(column.data));

    constexpr std::ptrdiff_t ptrdiff = 8;

    const thrust::device_vector<std::int64_t> device_data(data_ptr,
                                                          data_ptr + ptrdiff);

    EXPECT_EQ(0, device_data[1]);
    EXPECT_EQ(2, device_data[3]);
    EXPECT_EQ(4, device_data[5]);
    EXPECT_EQ(6, device_data[7]);
}
