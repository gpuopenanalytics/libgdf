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

#include <gdf/gdf.h>

#ifdef __cplusplus
#define BEGIN_NAMESPACE_GDF_PARQUET                                           \
    namespace gdf {                                                           \
    namespace parquet {
#define END_NAMESPACE_GDF_PARQUET                                             \
    }                                                                         \
    }
#else
#define BEGIN_NAMESPACE_GDF_PARQUET
#define END_NAMESPACE_GDF_PARQUET
#endif

BEGIN_NAMESPACE_GDF_PARQUET

/// \brief Read parquet file from file path into array of gdf columns
/// \param[in] filename path to parquet file
/// \param[in] columns will be read from the file
/// \param[out] out_gdf_columns array
/// \param[out] out_gdf_columns_length number of columns
extern "C" gdf_error read_parquet(const char *const        filename,
                                  const char *const *const columns,
                                  gdf_column **const       out_gdf_columns,
                                  size_t *const out_gdf_columns_length);

END_NAMESPACE_GDF_PARQUET

#ifdef __cplusplus

#include <string>
#include <vector>
#include <arrow/io/file.h>

namespace gdf {
namespace parquet {

/// \brief Read parquet file from file path into array of gdf columns
/// \param[in] filename path to parquet file
/// \param[in] indices of the rowgroups that will be read from the file
/// \param[in] indices of the columns that will be read from the file
/// \param[out] out_gdf_columns vector of gdf_column pointers. The data read.
gdf_error
read_parquet_by_ids(const std::string &             filename,
                    const std::vector<std::size_t> &row_group_indices,
                    const std::vector<std::size_t> &column_indices,
                    std::vector<gdf_column *> &     out_gdf_columns);

/// \brief Read parquet file from file interface into array of gdf columns
/// \param[in] filename path to parquet file
/// \param[in] indices of the rowgroups that will be read from the file
/// \param[in] indices of the columns that will be read from the file
/// \param[out] out_gdf_columns vector of gdf_column pointers. The data read.
gdf_error
read_parquet_by_ids(std::shared_ptr<::arrow::io::RandomAccessFile> file,
                    const std::vector<std::size_t> &row_group_indices,
                    const std::vector<std::size_t> &column_indices,
                    std::vector<gdf_column *> &     out_gdf_columns);

}  // namespace parquet
}  // namespace gdf

#endif
