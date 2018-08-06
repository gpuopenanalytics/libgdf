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

#include <arrow/util/bit-util.h>
#include <arrow/util/logging.h>
#include <parquet/column_reader.h>
#include <parquet/file/metadata.h>

#include "column_reader.h"
#include "file_reader.h"

#include <gdf/parquet/api.h>

static constexpr gdf_error GDF_BAD_ALLOC =
  static_cast<gdf_error>(std::numeric_limits<std::int64_t>::max());

BEGIN_NAMESPACE_GDF_PARQUET

namespace {

template <::parquet::Type::type TYPE>
struct parquet_physical_traits {};

#define PARQUET_PHYSICAL_TRAITS_FACTORY(TYPE, DTYPE)                          \
    template <>                                                               \
    struct parquet_physical_traits<::parquet::Type::TYPE> {                   \
        static constexpr gdf_dtype dtype = GDF_##DTYPE;                       \
    }

PARQUET_PHYSICAL_TRAITS_FACTORY(BOOLEAN, INT8);
PARQUET_PHYSICAL_TRAITS_FACTORY(INT32, INT32);
PARQUET_PHYSICAL_TRAITS_FACTORY(INT64, INT64);
PARQUET_PHYSICAL_TRAITS_FACTORY(INT96, invalid);
PARQUET_PHYSICAL_TRAITS_FACTORY(FLOAT, FLOAT32);
PARQUET_PHYSICAL_TRAITS_FACTORY(DOUBLE, FLOAT64);
PARQUET_PHYSICAL_TRAITS_FACTORY(BYTE_ARRAY, invalid);
PARQUET_PHYSICAL_TRAITS_FACTORY(FIXED_LEN_BYTE_ARRAY, invalid);

#undef PARQUET_PHYSICAL_TRAITS_FACTORY

template <::parquet::Type::type TYPE>
static inline std::size_t
_ReadBatch(const std::shared_ptr<::parquet::ColumnReader> &column_reader,
           const std::size_t                               num_rows,
           std::int16_t *const                             definition_levels,
           std::int16_t *const                             repetition_levels,
           const gdf_column &                              _gdf_column) {
    const std::shared_ptr<
      gdf::parquet::ColumnReader<::parquet::DataType<TYPE>>> &reader =
      std::static_pointer_cast<
        gdf::parquet::ColumnReader<::parquet::DataType<TYPE>>>(column_reader);

    typedef typename ::parquet::type_traits<TYPE>::value_type value_type;

    value_type *const values =
      static_cast<value_type *const>(_gdf_column.data);
    std::uint8_t *const valid_bits =
      static_cast<std::uint8_t *const>(_gdf_column.valid);

    static std::int64_t levels_read = 0;
    static std::int64_t values_read = 0;
    static std::int64_t nulls_count = 0;

    static const std::size_t min_batch_size = 4096;
    std::size_t              batch          = 0;
    std::size_t              batch_actual   = 0;
    std::size_t              batch_size     = 8;
    std::size_t              total_read     = 0;
    do {
        batch = reader->ReadBatchSpaced(
          batch_size,
          definition_levels,
          repetition_levels,
          values + batch_actual,
          valid_bits + static_cast<std::ptrdiff_t>(batch_actual / 8),
          0,
          &levels_read,
          &values_read,
          &nulls_count);
        total_read += static_cast<std::size_t>(values_read);
        batch_actual += batch;
        batch_size = std::max(batch_size * 2, min_batch_size);
    } while (batch > 0 || levels_read > 0);
    DCHECK_GE(num_rows, total_read);

    return total_read;
}

struct ParquetTypeHash {
    template <class T>
    std::size_t
    operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

const std::unordered_map<::parquet::Type::type, gdf_dtype, ParquetTypeHash>
  dtype_from_physical_type_map{
    {::parquet::Type::BOOLEAN, GDF_INT8},
    {::parquet::Type::INT32, GDF_INT32},
    {::parquet::Type::INT64, GDF_INT64},
    {::parquet::Type::INT96, GDF_invalid},
    {::parquet::Type::FLOAT, GDF_FLOAT32},
    {::parquet::Type::DOUBLE, GDF_FLOAT64},
    {::parquet::Type::BYTE_ARRAY, GDF_invalid},
    {::parquet::Type::FIXED_LEN_BYTE_ARRAY, GDF_invalid},
  };

const std::
  unordered_map<::parquet::LogicalType::type, gdf_dtype, ParquetTypeHash>
    dtype_from_logical_type_map{
      {::parquet::LogicalType::NONE, GDF_invalid},
      {::parquet::LogicalType::UTF8, GDF_invalid},
      {::parquet::LogicalType::MAP, GDF_invalid},
      {::parquet::LogicalType::MAP_KEY_VALUE, GDF_invalid},
      {::parquet::LogicalType::LIST, GDF_invalid},
      {::parquet::LogicalType::ENUM, GDF_invalid},
      {::parquet::LogicalType::DECIMAL, GDF_invalid},
      {::parquet::LogicalType::DATE, GDF_DATE32},
      {::parquet::LogicalType::TIME_MILLIS, GDF_invalid},
      {::parquet::LogicalType::TIME_MICROS, GDF_invalid},
      {::parquet::LogicalType::TIMESTAMP_MILLIS, GDF_TIMESTAMP},
      {::parquet::LogicalType::TIMESTAMP_MICROS, GDF_invalid},
      {::parquet::LogicalType::UINT_8, GDF_invalid},
      {::parquet::LogicalType::UINT_16, GDF_invalid},
      {::parquet::LogicalType::UINT_32, GDF_invalid},
      {::parquet::LogicalType::UINT_64, GDF_invalid},
      {::parquet::LogicalType::INT_8, GDF_INT8},
      {::parquet::LogicalType::INT_16, GDF_INT16},
      {::parquet::LogicalType::INT_32, GDF_INT32},
      {::parquet::LogicalType::INT_64, GDF_INT64},
      {::parquet::LogicalType::JSON, GDF_invalid},
      {::parquet::LogicalType::BSON, GDF_invalid},
      {::parquet::LogicalType::INTERVAL, GDF_invalid},
      {::parquet::LogicalType::NA, GDF_invalid},
    };

static inline gdf_dtype
_DTypeFrom(const ::parquet::ColumnDescriptor *const column_descriptor) {
    const ::parquet::LogicalType::type logical_type =
      column_descriptor->logical_type();

    if (logical_type != ::parquet::LogicalType::NONE) {
        return dtype_from_logical_type_map.at(logical_type);
    }

    const ::parquet::Type::type physical_type =
      column_descriptor->physical_type();

    return dtype_from_physical_type_map.at(physical_type);
}

template <::parquet::Type::type TYPE>
static inline gdf_error
_AllocateGdfColumn(const std::size_t                        num_rows,
                   const ::parquet::ColumnDescriptor *const column_descriptor,
                   gdf_column &                             _gdf_column) {
    const std::size_t value_byte_size =
      static_cast<std::size_t>(::parquet::type_traits<TYPE>::value_byte_size);

    try {
        _gdf_column.data =
          static_cast<void *>(new std::uint8_t[num_rows * value_byte_size]);
    } catch (const std::bad_alloc &e) {
#ifdef GDF_DEBUG
        std::cerr << "Allocation error for data\n" << e.what() << std::endl;
#endif
        return GDF_BAD_ALLOC;
    }

    try {
        _gdf_column.valid = static_cast<gdf_valid_type *>(
          new std::uint8_t[arrow::BitUtil::BytesForBits(num_rows)]);
    } catch (const std::bad_alloc &e) {
#ifdef GDF_DEBUG
        std::cerr << "Allocation error for valid\n" << e.what() << std::endl;
#endif
        return GDF_BAD_ALLOC;
    }

    _gdf_column.size  = num_rows;
    _gdf_column.dtype = _DTypeFrom(column_descriptor);

    return GDF_SUCCESS;
}

static inline gdf_error
_AllocateGdfColumns(
  const std::size_t                                       num_columns,
  const std::size_t                                       num_rows,
  const std::vector<const ::parquet::ColumnDescriptor *> &column_descriptors,
  gdf_column *const                                       gdf_columns) {
#define WHEN(TYPE)                                                            \
    case ::parquet::Type::TYPE:                                               \
        _AllocateGdfColumn<::parquet::Type::TYPE>(                            \
          num_rows, column_descriptor, _gdf_column);                          \
        break

    for (std::size_t i = 0; i < num_columns; i++) {
        gdf_column &                             _gdf_column = gdf_columns[i];
        const ::parquet::ColumnDescriptor *const column_descriptor =
          column_descriptors[i];

        switch (column_descriptor->physical_type()) {
            WHEN(BOOLEAN);
            WHEN(INT32);
            WHEN(INT64);
            WHEN(INT96);
            WHEN(FLOAT);
            WHEN(DOUBLE);
            WHEN(BYTE_ARRAY);
            WHEN(FIXED_LEN_BYTE_ARRAY);
        default:
#ifdef GDF_DEBUG
            std::cerr << "Column type not supported" << std::endl;
#endif
            return GDF_BAD_ALLOC;
        }
    }
#undef WHEN
    return GDF_SUCCESS;
}

static inline gdf_column *
_CreateGdfColumns(const std::size_t num_columns) {
    gdf_column *_gdf_columns = nullptr;
    try {
        _gdf_columns = new gdf_column[num_columns];
    } catch (const std::bad_alloc &e) {
#ifdef GDF_DEBUG
        std::cerr << "Allocation error for gdf columns\n"
                  << e.what() << std::endl;
#endif
        DCHECK_EQ(nullptr, _gdf_columns);
    }
    return _gdf_columns;
}

}  // namespace

extern "C" {

gdf_error
read_parquet_file(const char *const  filename,
                  gdf_column **const out_gdf_columns,
                  std::size_t *const out_gdf_columns_length) {
    const std::unique_ptr<FileReader> file_reader =
      FileReader::OpenFile(filename);

    const std::shared_ptr<const ::parquet::FileMetaData> &metadata =
      file_reader->metadata();

    const std::size_t num_row_groups =
      static_cast<std::size_t>(metadata->num_row_groups());

    if (num_row_groups == 0) { return GDF_BAD_ALLOC; }

    const std::size_t num_rows =
      static_cast<std::size_t>(metadata->num_rows());

    if (num_rows == 0) { return GDF_BAD_ALLOC; }

    const std::size_t num_columns =
      static_cast<std::size_t>(metadata->num_columns());

    gdf_column *const gdf_columns = _CreateGdfColumns(num_columns);

    if (gdf_columns == nullptr) { return GDF_BAD_ALLOC; }

    std::vector<const ::parquet::ColumnDescriptor *> column_descriptors;
    column_descriptors.reserve(num_columns);
    for (std::size_t i = 0; i < num_columns; i++) {
        column_descriptors.emplace_back(
          file_reader->RowGroup(0)->Column(i)->descr());
    }
    if (_AllocateGdfColumns(
          num_columns, num_rows, column_descriptors, gdf_columns)
        != GDF_SUCCESS) {
        return GDF_BAD_ALLOC;
    }

    std::int16_t *const definition_levels = new std::int16_t[num_rows];
    std::int16_t *const repetition_levels = new std::int16_t[num_rows];

    for (std::size_t row_group_index = 0; row_group_index < num_row_groups;
         row_group_index++) {
        const std::shared_ptr<::parquet::RowGroupReader> row_group_reader =
          file_reader->RowGroup(static_cast<int>(row_group_index));

        for (std::size_t column_reader_index = 0;
             column_reader_index < num_columns;
             column_reader_index++) {
            const gdf_column &_gdf_column = gdf_columns[column_reader_index];
            const std::shared_ptr<::parquet::ColumnReader> column_reader =
              row_group_reader->Column(static_cast<int>(column_reader_index));

            switch (column_reader->type()) {
#define WHEN(TYPE)                                                            \
    case ::parquet::Type::TYPE:                                               \
        _ReadBatch<::parquet::Type::TYPE>(column_reader,                      \
                                          num_rows,                           \
                                          definition_levels,                  \
                                          repetition_levels,                  \
                                          _gdf_column);                       \
        break
                WHEN(BOOLEAN);
                WHEN(INT32);
                WHEN(INT64);
                WHEN(INT96);
                WHEN(FLOAT);
                WHEN(DOUBLE);
                WHEN(BYTE_ARRAY);
                WHEN(FIXED_LEN_BYTE_ARRAY);
            default:
#ifdef GDF_DEBUG
                std::cerr << "Column type error from file" << std::endl;
#endif
                return GDF_BAD_ALLOC;
#undef WHEN
            }
        }
    }

    delete[] definition_levels;
    delete[] repetition_levels;

    *out_gdf_columns        = gdf_columns;
    *out_gdf_columns_length = num_columns;

    return GDF_SUCCESS;
}
}

END_NAMESPACE_GDF_PARQUET
