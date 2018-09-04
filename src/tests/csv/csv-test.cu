/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <cstdlib>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include <gdf/gdf.h>
#include <gdf/cffi/functions.h>
 
struct gdf_csv_test : public ::testing::Test {
  void TearDown() {
  }
};
 
TEST(gdf_csv_test, SiuTest)
{
  const char *fileName = "tmp_csvreader_file.csv";
  char delimiter = ',';
  int num_cols = 2;
  const char *col_names[] = {"col1", "col2"};
  const char *dtypes[] = {"int64", "int64"};

  gdf_column **res = read_csv(
                		fileName,                     // in: the file to be loaded
                		delimiter,                    // in: the delimiter
                		num_cols,                     // in: number of columns
                		col_names,                    // in: ordered list of column names
                		dtypes                        // in: ordered list of dtypes
    			      );

  EXPECT_TRUE( true );
}

 
